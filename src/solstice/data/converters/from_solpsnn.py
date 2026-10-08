# =========================================================================================
# (C) (or copyright) 2026. UT-Battelle, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract DE-AC05-00OR22725 with
# UT-Battelle, LLC, which manages Oak Ridge National Laboratory (ORNL) for the U.S.
# Department of Energy (DOE). The U.S. Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
# to reproduce, prepare derivative works, distribute copies to the public, perform
# publicly and display publicly, and to permit others to do so. The DOE will provide
# public access to these results in accordance with the DOE Public Access Plan
# (http://energy.gov/downloads/doe-public-access-plan).
# =========================================================================================
# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: Apache-2.0
"""SOLPS-NN Training Data v1 (Dasbach et al.) -> stacked ensemble store.

Source: S. Dasbach, "SOLPS-NN Training Data v1", Zenodo
10.5281/zenodo.19237127 (CC BY 4.0); paper arXiv:2604.19223. A scan of
SOLPS-ITER (B2.5, fluid neutrals, no drifts) over eight scalar inputs on
one JET-shaped 102 x 48 grid whose size is scaled with the major radius
R at fixed aspect ratio. Species D0, D+, N0 .. N7+.

The output follows solstice.data.store (one netcdf per split, canonical
mesh + `case`-stacked fields + `input_*` scalars). The one difference
from a fixed-geometry SOLPS ensemble: the stored mesh is the unscaled
baserun (JET size, R_JET = 3.0007 m); the physical mesh of case k is

    cell_r, cell_z, cell_corners_*, vx_*  x  mesh_scale[k]
    cell_vol                              x  mesh_scale[k]**3

with mesh_scale = R / R_JET stored per case. cell_b is the baserun field
(the release does not ship per-case fields; B enters as an input).

Release conventions used here (verified against the release's own
pwmxap / fnixap): arrays are (n_sim, nx+2, ny+2[, ...]) with guard cells,
face-flux component 0 is the poloidal flux through the west face of the
cell, so the outer-target face is index nx+1 and the inner-target face
index 1. Temperatures are stored in J and converted to eV.

Build:
    python -m solstice.data.converters.from_solpsnn <release_dir> <out_dir> \
        [--splits train test] [--limit N]
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import xarray as xr

from solstice.data.converters import from_solps
from solstice.data.schema import SCHEMA_VERSION

EV = 1.602176634e-19
R_JET = 3.000727179161820          # major radius of the unscaled grid (release example.py)
SPECIES = ("D0", "D1", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7")
CHARGE = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
PARAMS = ("R", "B", "P_in", "D_puff", "N_puff", "D_core", "D_perp", "chi_perp")
# regime classification indices of the paper (guard-inclusive python ix/iy)
IX_OMP, IY_SEP = 64, 25
REGIMES = ("sheath-limited", "attached", "detached", "cold core")
CHUNK = 128                          # cases per read of the memory-mapped release arrays

DOI = "10.5281/zenodo.19237127"
CITATION = ("Dasbach S., Brezinsek S., Liang Y., Reiser D., Wiesen S., "
            "arXiv:2604.19223 (2026); Dasbach S. & Wiesen S., "
            "Nucl. Mater. Energy 34, 101396 (2023)")


def canonical_inputs(X: np.ndarray) -> dict[str, np.ndarray]:
    """Release X_units columns -> canonical input names (docs/specs/data_schema.md).

    P_in is split into pe = pi = P_in / 2 and chi_perp into hci = hce,
    the same convention as the DIII-D store, so the training code's
    merge into ptot / chi applies unchanged. Puffs are in atoms/s
    (fluid neutrals: no molecules), hence puff_D / puff_N."""
    X = np.asarray(X, dtype=float)
    return {
        "rmajor": X[:, 0],
        "btor": X[:, 1],
        "pe": 0.5 * X[:, 2],
        "pi": 0.5 * X[:, 2],
        "puff_D": X[:, 3],
        "puff_N": X[:, 4],
        "core_fueling": X[:, 5],
        "dna": X[:, 6],
        "hci": X[:, 7],
        "hce": X[:, 7],
    }


INPUT_UNITS = {"rmajor": "m", "btor": "T", "pe": "W", "pi": "W", "puff_D": "atoms/s",
               "puff_N": "atoms/s", "core_fueling": "atoms/s", "dna": "m^2/s",
               "hci": "m^2/s", "hce": "m^2/s"}


def classify_regime(te_omp_eV: np.ndarray, te_ot_eV: np.ndarray) -> np.ndarray:
    """Paper Section 2: cold core (Te_omp < 10 eV) > sheath-limited
    (Te_ot/Te_omp >= 0.8) > attached (Te_ot > 5 eV) > detached."""
    reg = np.full(len(te_omp_eV), "detached", dtype=object)
    reg[te_ot_eV > 5] = "attached"
    with np.errstate(divide="ignore", invalid="ignore"):
        reg[te_ot_eV / te_omp_eV >= 0.8] = "sheath-limited"
    reg[te_omp_eV < 10] = "cold core"
    return reg.astype(str)


def interior(a: np.ndarray) -> np.ndarray:
    """(n, nx+2, ny+2) -> (n, nx*ny) real cells, iy fastest (mesh cell order)."""
    a = np.asarray(a)
    return a[:, 1:-1, 1:-1].reshape(a.shape[0], -1)


def poloidal_heat_flux(fht: np.ndarray, gs_x: np.ndarray, scale: np.ndarray):
    """Cell-centred q_pol [W/m^2] and face-centred target profiles from the
    west-face poloidal heat flux fht[..., 0] [W], following
    from_solps.read_case_derived (no thermal-current correction: the
    release ships the total flux only).

    fht: (n, nx+2, ny+2, 2); gs_x: (nx+2, ny+2) unscaled x-face areas;
    scale: (n,) mesh scale, areas go as scale**2."""
    fx = np.asarray(fht[..., 0], dtype=np.float64)
    area = gs_x[None] * (np.asarray(scale)[:, None, None] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_face = np.where(area > 0, fx / area, 0.0)
    q_cc = 0.5 * (q_face + np.roll(q_face, -1, axis=1))
    q_cc[:, -1] = 0.0
    q_cc = np.nan_to_num(q_cc, nan=0.0, posinf=0.0, neginf=0.0)
    q_in = np.nan_to_num(q_face[:, 1, 1:-1], posinf=0.0, neginf=0.0)
    q_out = np.nan_to_num(q_face[:, -1, 1:-1], posinf=0.0, neginf=0.0)
    return interior(q_cc), q_in, q_out


def _read_completed_ids(split_dir: Path) -> np.ndarray:
    return np.load(split_dir / "completed.npy").astype(np.int64)


def _attempted_count(split_dir: Path) -> int | None:
    try:
        con = sqlite3.connect(split_dir / "simulations.db")
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        n = max(con.execute(f"select count(*) from {t}").fetchone()[0] for t in tables)
        con.close()
        return int(n)
    except Exception:  # noqa: BLE001 - metadata only
        return None


def build_split_store(release_dir: str, split: str, out_path: str,
                      limit: int | None = None, progress: bool = True) -> xr.Dataset:
    release_dir = Path(release_dir)
    sdir = release_dir / split
    t0 = time.time()

    geo = from_solps.load_geometry(release_dir / "baserun" / "b2fgmtry")
    mesh = from_solps.build_structured_mesh(geo)
    nx, ny = int(mesh.attrs["nx"]), int(mesh.attrs["ny"])
    crx0 = np.load(release_dir / "geometry" / "crx.npy")
    if not np.allclose(crx0, np.asarray(geo["crx"])):
        raise ValueError("release geometry/crx.npy differs from baserun/b2fgmtry")
    gs_x = np.asarray(geo["gs"])[:, :, 0]
    vol0 = np.asarray(geo["vol"])[1:-1, 1:-1].reshape(-1)

    X = np.load(sdir / "X_units.npy")
    sim_id = _read_completed_ids(sdir)
    n = len(X) if limit is None else min(limit, len(X))
    X, sim_id = X[:n], sim_id[:n]
    scale = X[:, 0] / R_JET
    n_cells = nx * ny

    def mm(name):
        return np.load(sdir / f"{name}.npy", mmap_mode="r")

    fields: dict[str, np.ndarray] = {}
    for name in ("te", "ti"):
        fields[name] = np.empty((n, n_cells), np.float32)
    for sp in SPECIES:
        fields[f"na_{sp}"] = np.empty((n, n_cells), np.float32)
        fields[f"ua_{sp}"] = np.empty((n, n_cells), np.float32)
    for name in ("ne", "prad", "q_pol"):
        fields[name] = np.empty((n, n_cells), np.float32)
    q_in = np.empty((n, ny), np.float32)
    q_out = np.empty((n, ny), np.float32)
    g_in = np.empty((n, ny), np.float32)
    g_out = np.empty((n, ny), np.float32)
    i_d1 = SPECIES.index("D1")
    te_omp, te_ot, ne_omp, ne_ot = (np.empty(n) for _ in range(4))

    te_mm, ti_mm, na_mm, ua_mm, rq_mm, fht_mm, fna_mm = (
        mm(k) for k in ("te", "ti", "na", "ua", "rqrad", "fht", "fna"))
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        sl = slice(a, b)
        te = np.asarray(te_mm[sl]) / EV
        ti = np.asarray(ti_mm[sl]) / EV
        fields["te"][sl] = interior(te)
        fields["ti"][sl] = interior(ti)
        na = np.asarray(na_mm[sl])                                     # (c, nx+2, ny+2, ns)
        ua = np.asarray(ua_mm[sl])
        for j, sp in enumerate(SPECIES):
            fields[f"na_{sp}"][sl] = interior(na[..., j])
            fields[f"ua_{sp}"][sl] = interior(ua[..., j])
        ne = na @ CHARGE
        fields["ne"][sl] = interior(ne)
        rq = np.asarray(rq_mm[sl]).sum(-1)                             # W per cell
        fields["prad"][sl] = interior(rq) / (vol0[None] * scale[sl, None] ** 3)
        qc, qi, qo = poloidal_heat_flux(np.asarray(fht_mm[sl]), gs_x, scale[sl])
        fields["q_pol"][sl], q_in[sl], q_out[sl] = qc, qi, qo
        # D+ particle flux density through the target faces [atoms/(m^2 s)]
        area = gs_x[None] * scale[sl, None, None] ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            g_in[sl] = np.nan_to_num(np.asarray(fna_mm[sl, 1, 1:-1, 0, i_d1]) / area[:, 1, 1:-1])
            g_out[sl] = np.nan_to_num(np.asarray(fna_mm[sl, -1, 1:-1, 0, i_d1]) / area[:, -1, 1:-1])
        te_omp[sl], te_ot[sl] = te[:, IX_OMP, IY_SEP], te[:, nx, IY_SEP]
        ne_omp[sl], ne_ot[sl] = ne[:, IX_OMP, IY_SEP], ne[:, nx, IY_SEP]
        if progress:
            print(f"  {split}: {b}/{n} cases  ({time.time() - t0:.0f} s)", flush=True)

    ds = mesh.copy()
    ds = ds.assign_coords(case=("case", np.array([f"{split}_{i:05d}" for i in sim_id])))
    ds["sim_id"] = ("case", sim_id)
    ds["mesh_scale"] = ("case", scale)
    ds["mesh_scale"].attrs.update(long_name="R / R_JET: multiply cell_r, cell_z, corners and "
                                            "vertices by this, cell_vol by its cube")
    for name, arr in fields.items():
        ds[name] = (("case", "cell"), arr)
    for name, units in (("te", "eV"), ("ti", "eV"), ("ne", "m^-3"), ("prad", "W/m^3"),
                        ("q_pol", "W/m^2")):
        ds[name].attrs["units"] = units
    for sp in SPECIES:
        ds[f"na_{sp}"].attrs["units"] = "m^-3"
        ds[f"ua_{sp}"].attrs["units"] = "m/s"
    ds["prad"].attrs["long_name"] = "line radiation density, all species (release rqrad / cell volume)"
    ds["q_pol"].attrs["long_name"] = ("cell-centred poloidal heat flux density from the release's "
                                      "total face heat flux fht (no thermal-current correction)")
    ds["q_inner_target"] = (("case", "target_iy"), q_in)
    ds["q_outer_target"] = (("case", "target_iy"), q_out)
    for k in ("q_inner_target", "q_outer_target"):
        ds[k].attrs.update(units="W/m^2", long_name="face-centred target heat flux profile along iy")
    ds["gamma_inner_target"] = (("case", "target_iy"), g_in)
    ds["gamma_outer_target"] = (("case", "target_iy"), g_out)
    for k in ("gamma_inner_target", "gamma_outer_target"):
        ds[k].attrs.update(units="atoms/(m^2 s)",
                           long_name="face-centred D+ particle flux density along iy (release fna)")
    ds["inner_target_area"] = ("target_iy", gs_x[1, 1:-1])
    ds["outer_target_area"] = ("target_iy", gs_x[-1, 1:-1])
    for k in ("inner_target_area", "outer_target_area"):
        ds[k].attrs.update(units="m^2", long_name="unscaled target face areas; multiply by mesh_scale**2")

    for name, values in canonical_inputs(X).items():
        ds[f"input_{name}"] = ("case", values)
        ds[f"input_{name}"].attrs["units"] = INPUT_UNITS[name]
    ds["input_pe"].attrs["long_name"] = "P_in / 2 (release input power split equally, pe = pi)"

    for name in ("psol", "pwmxap", "fnixap"):
        ds[name] = ("case", np.load(sdir / f"{name}.npy")[:n].astype(np.float64))
    ds["psol"].attrs.update(units="W", long_name="power crossing the separatrix (release)")
    ds["pwmxap"].attrs.update(units="W/m^2", long_name="peak heat flux at the outer target (release)")
    ds["fnixap"].attrs.update(units="atoms/s", long_name="D+ flux to the outer target (release)")
    ds["te_omp_sep"] = ("case", te_omp)
    ds["te_ot_sep"] = ("case", te_ot)
    ds["ne_omp_sep"] = ("case", ne_omp)
    ds["ne_ot_sep"] = ("case", ne_ot)
    ds["regime"] = ("case", classify_regime(te_omp, te_ot))
    ds["params_converged"] = ("case", np.ones(n, dtype=bool))      # only completed runs ship

    # QC (same columns as solstice.data.store.qc_flags, computed without stacking)
    nan_count = np.zeros(n)
    for arr in fields.values():
        nan_count += (~np.isfinite(arr)).sum(axis=1)
    nan_frac = nan_count / (len(fields) * n_cells)
    ds["qc_max_te"] = ("case", np.nanmax(fields["te"], axis=1).astype(np.float64))
    ds["qc_max_ne"] = ("case", np.nanmax(fields["ne"], axis=1).astype(np.float64))
    ds["qc_nan_frac"] = ("case", nan_frac)
    ds["qc_pass"] = ("case", nan_frac <= 0.02)

    n_attempted = _attempted_count(sdir)
    ds.attrs.update(
        ensemble=f"solpsnn-v1-{split}",
        store_kind="stacked_ensemble",
        machine="scaled-JET",
        topology="structured",
        species=json.dumps(list(SPECIES)),
        n_cases=int(n),
        n_attempted=-1 if n_attempted is None else n_attempted,
        mesh_reference_rmajor=R_JET,
        mesh_scaling=("stored mesh is the unscaled baserun; physical geometry of case k is the "
                      "mesh times mesh_scale[k] (cell_vol times mesh_scale[k]**3); cell_b is "
                      "the baserun field"),
        ix_omp=IX_OMP, iy_sep=IY_SEP, ix_ot=nx,
        regimes=json.dumps(list(REGIMES)),
        source_doi=DOI,
        source_license="CC-BY-4.0",
        citation=CITATION,
        solps_setup="SOLPS-ITER B2.5, fluid neutrals, no drifts, potential not solved",
        schema_version=SCHEMA_VERSION,
        skipped="[]",
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars
                if ds[v].dtype.kind in "fiu"}
    for v in ds.data_vars:
        if ds[v].dtype.kind in "fiu" and "case" in ds[v].dims and "cell" in ds[v].dims:
            encoding[v]["chunksizes"] = (min(64, n), n_cells)
    ds.to_netcdf(out_path, encoding=encoding)
    if progress:
        reg = ds["regime"].values
        mix = ", ".join(f"{r} {100 * (reg == r).mean():.0f}%" for r in REGIMES)
        print(f"wrote {out_path}: {n} cases, {len(fields)} fields ({time.time() - t0:.0f} s)")
        print(f"  regimes: {mix}")
    return ds


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release_dir", help="unzipped Zenodo release (contains baserun/, train/, test/)")
    ap.add_argument("out_dir")
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--limit", type=int, default=None, help="first N cases per split")
    ap.add_argument("--prefix", default="solstice_store_solpsnn_v1")
    args = ap.parse_args(argv)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        out = Path(args.out_dir) / f"{args.prefix}_{split}.nc"
        build_split_store(args.release_dir, split, str(out), limit=args.limit)


if __name__ == "__main__":
    main()
