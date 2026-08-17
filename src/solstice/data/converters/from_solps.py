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
"""SOLPS run directory -> canonical case (docs/specs/data_schema.md).

All raw SOLPS file parsing is imported from the SOLPS-routines package
(`solps_routines.readers`) by Jeremy Lore (ORNL),
https://github.com/ORNL-Fusion/SOLPS-routines — used as an external
library, no code copied here. It provides: b2fgmtry corners and
neighbor maps (leftix/bottomix handle the poloidal cuts exactly as B2
does), region labels and their semantics (geo["region"]/region_ids),
and balance.nc contents. Nothing is re-derived here.

Structured grids only for now; GOAT wide grids raise until a case is
available to validate against.

Conventions fixed here:
  - interior cells only (guard cells dropped); cell order = C-order
    flatten of [1:nx+1, 1:ny+1], i.e. iy fastest; cell_ix/cell_iy are
    guard-inclusive python indices so any order is reconstructible.
  - b2fgmtry corner order is 0=SW, 1=SE, 2=NW, 3=NE (SW = lower ix,
    lower iy); polygon loop is [0, 1, 3, 2].
  - balance.nc arrays arrive (ny+2, nx+2) [or (ns, ny+2, nx+2)] and are
    transposed to match b2fgmtry's (nx+2, ny+2).
  - te/ti are converted J -> eV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from solstice.data.schema import SCHEMA_VERSION

EV = 1.602176634e-19
CORNER_LOOP = (0, 1, 3, 2)
FACE_SETS = {
    "interior": 0,
    "inner_target": 1,
    "outer_target": 2,
    "core_boundary": 3,
    "pfr_boundary": 4,
    "wall": 5,
}


def _readers():
    try:
        from solps_routines import readers
    except ImportError as err:
        raise ImportError(
            "solps_routines is required for SOLPS conversion — the SOLPS-routines "
            "by Jeremy Lore (ORNL); add its src/ to PYTHONPATH"
        ) from err
    return readers


def load_geometry(b2fgmtry_path: str) -> dict:
    geo = _readers().read_b2fgmtry(str(b2fgmtry_path), use_cache=False)
    if geo.get("isUnstructured"):
        raise NotImplementedError(
            "GOAT wide-grid conversion pending a validation case; structured only"
        )
    return geo


def build_structured_mesh(geo: dict) -> xr.Dataset:
    """Canonical mesh from a structured b2fgmtry dict (SOLPS-routines parsed)."""
    nx, ny = int(geo["nx"]), int(geo["ny"])
    crx, cry = np.asarray(geo["crx"]), np.asarray(geo["cry"])
    interior = np.s_[1 : nx + 1, 1 : ny + 1]

    ix2d, iy2d = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing="ij")
    cell_ix, cell_iy = ix2d.reshape(-1), iy2d.reshape(-1)
    n_cells = nx * ny
    corners_r = crx[interior].reshape(n_cells, 4)[:, CORNER_LOOP]
    corners_z = cry[interior].reshape(n_cells, 4)[:, CORNER_LOOP]

    # vertices: dedup corners on rounded coordinates
    pts = np.round(np.stack([corners_r.ravel(), corners_z.ravel()], axis=1), 9)
    uniq, inv = np.unique(pts, axis=0, return_inverse=True)
    cell_corner_vx = inv.reshape(n_cells, 4)  # in CORNER_LOOP order: SW,SE,NE,NW

    # faces from B2's own neighbor maps (cuts handled by leftix/bottomix)
    cell_id = np.full((nx + 2, ny + 2), -1, dtype=np.int64)
    cell_id[interior] = np.arange(n_cells).reshape(nx, ny)
    leftix = np.asarray(geo["leftix_py"], dtype=int)
    leftiy = np.asarray(geo["leftiy_py"], dtype=int)
    botix = np.asarray(geo["bottomix_py"], dtype=int)
    botiy = np.asarray(geo["bottomiy_py"], dtype=int)

    # local vertex ids per side, in CORNER_LOOP order [SW, SE, NE, NW]
    side_vx = {
        "west": (3, 0),   # NW-SW
        "south": (0, 1),  # SW-SE
        "east": (1, 2),   # SE-NE
        "north": (2, 3),  # NE-NW
    }

    face_vertices, face_cells, face_set = [], [], []

    def add_face(c, side, neighbor, boundary_label):
        a, b = side_vx[side]
        face_vertices.append((cell_corner_vx[c, a], cell_corner_vx[c, b]))
        face_cells.append((c, neighbor))
        face_set.append(FACE_SETS[boundary_label if neighbor < 0 else "interior"])

    region_vol = np.asarray(geo["region"])[:, :, 0]  # region_ids['vol'] semantics
    is_core_row = region_vol == geo["region_ids"]["vol"]["is_core"]
    # geometric inner/outer target identification (inner = smaller R)
    mean_r_west = crx[1, 1 : ny + 1, :].mean()
    mean_r_east = crx[nx, 1 : ny + 1, :].mean()
    west_label = "inner_target" if mean_r_west < mean_r_east else "outer_target"
    east_label = "outer_target" if west_label == "inner_target" else "inner_target"

    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            c = cell_id[ix, iy]
            ln = cell_id[leftix[ix, iy], leftiy[ix, iy]]
            add_face(c, "west", ln, west_label if ix == 1 else "interior")
            bn = cell_id[botix[ix, iy], botiy[ix, iy]]
            add_face(
                c, "south", bn,
                ("core_boundary" if is_core_row[ix, iy] else "pfr_boundary")
                if iy == 1 else "interior",
            )
            if ix == nx:
                add_face(c, "east", -1, east_label)
            if iy == ny:
                add_face(c, "north", -1, "wall")

    # interior faces were added once from each side; dedup on the cell pair
    fv = np.asarray(face_vertices)
    fc = np.asarray(face_cells)
    fs = np.asarray(face_set)
    pair_key = np.where(
        fc[:, 1] >= 0,
        np.minimum(fc[:, 0], fc[:, 1]) * (nx * ny + 1) + np.maximum(fc[:, 0], fc[:, 1]),
        -np.arange(1, len(fc) + 1),  # boundary faces are unique
    )
    _, keep = np.unique(pair_key, return_index=True)
    keep.sort()
    fv, fc, fs = fv[keep], fc[keep], fs[keep]

    bb = np.asarray(geo["bb"])[interior].reshape(n_cells, 4)
    ds = xr.Dataset(
        {
            "cell_r": ("cell", corners_r.mean(axis=1)),
            "cell_z": ("cell", corners_z.mean(axis=1)),
            "cell_corners_r": (("cell", "corner"), corners_r),
            "cell_corners_z": (("cell", "corner"), corners_z),
            "cell_vol": ("cell", np.asarray(geo["vol"])[interior].reshape(-1)),
            "cell_b": (("cell", "bcomp"), bb),
            "cell_region": ("cell", region_vol[interior].reshape(-1).astype(np.int8)),
            "cell_region_x": ("cell", np.asarray(geo["region"])[:, :, 1][interior].reshape(-1).astype(np.int8)),
            "cell_region_y": ("cell", np.asarray(geo["region"])[:, :, 2][interior].reshape(-1).astype(np.int8)),
            "cell_ix": ("cell", cell_ix),
            "cell_iy": ("cell", cell_iy),
            "face_vertices": (("face", "two"), fv),
            "face_cells": (("face", "two"), fc),
            "face_set": ("face", fs.astype(np.int8)),
            "vx_r": ("vertex", uniq[:, 0]),
            "vx_z": ("vertex", uniq[:, 1]),
        },
        attrs={
            "topology": "structured",
            "nx": nx,
            "ny": ny,
            "region_ids": json.dumps(geo["region_ids"]),
            "face_sets": json.dumps(FACE_SETS),
            "schema_version": SCHEMA_VERSION,
        },
    )
    return ds


def _species_tags(species_chars: np.ndarray) -> list[str]:
    tags = []
    for row in np.asarray(species_chars):
        s = "".join(c.decode() if isinstance(c, bytes) else str(c) for c in row).strip()
        tags.append(s.replace("+", "").replace(" ", ""))
    return tags


# EIRENE volumetric source groups (mapping from solpex-paper
# build_coupling_dataset.py; species-resolved groups take D+ = index 1)
EIRENE_SOURCES = {
    "sp": (["eirene_mc_papl_sna_bal", "eirene_mc_pmpl_sna_bal",
            "eirene_mc_pipl_sna_bal", "eirene_mc_pppl_sna_bal"], True),
    "sne": (["eirene_mc_pael_sne_bal", "eirene_mc_pmel_sne_bal"], False),
    "qe": (["eirene_mc_eael_she_bal", "eirene_mc_emel_she_bal",
            "eirene_mc_eiel_she_bal", "eirene_mc_epel_she_bal"], False),
    "qi": (["eirene_mc_eapl_shi_bal", "eirene_mc_empl_shi_bal",
            "eirene_mc_eipl_shi_bal", "eirene_mc_eppl_shi_bal"], False),
    "sm": (["eirene_mc_mapl_smo_bal", "eirene_mc_mmpl_smo_bal",
            "eirene_mc_mipl_smo_bal", "eirene_mc_mppl_smo_bal"], True),
}
NEUTRAL_FIELDS = ("dab2", "dmb2", "tab2", "tmb2")  # EIRENE grid, species 0


def read_balance_all(run_dir: str) -> dict:
    bal = _readers().read_balance(str(Path(run_dir) / "balance.nc"))
    if bal is None:
        raise FileNotFoundError(f"no balance.nc in {run_dir}")
    return bal


def _cells(a2d, nx, ny):
    return np.asarray(a2d).T[1 : nx + 1, 1 : ny + 1].reshape(-1)


def read_case_fields(run_dir_or_bal, nx: int, ny: int) -> tuple[dict, list[str]]:
    """Per-cell named field arrays (interior, flattened) from balance.nc."""
    bal = (read_balance_all(run_dir_or_bal)
           if not isinstance(run_dir_or_bal, dict) else run_dir_or_bal)
    tags = _species_tags(bal["species"])
    fields = {
        "te": _cells(bal["te"], nx, ny) / EV,
        "ti": _cells(bal["ti"], nx, ny) / EV,
        "ne": _cells(bal["ne"], nx, ny),
    }
    for i, tag in enumerate(tags):
        fields[f"na_{tag}"] = _cells(bal["na"][i], nx, ny)
        fields[f"ua_{tag}"] = _cells(bal["ua"][i], nx, ny)
    return fields, tags


def read_case_sources(bal: dict, nx: int, ny: int, run_dir: str | None = None) -> dict:
    """EIRENE source terms and neutral fields, per-cell.

    Volumetric source grouping follows the (documented) solpex mapping —
    a candidate for upstreaming into SOLPS-routines. Neutral fields come
    from SOLPS-routines read_ft44 (fort.44, native (nx, ny) B2 layout);
    the balance.nc EIRENE-grid fallback uses columns [1:-3], which
    matches fort.44 exactly (the legacy solpex slice [2:-2] was shifted
    by one poloidal column)."""
    out = {}
    for name, (group, species_dim) in EIRENE_SOURCES.items():
        acc = None
        for vn in group:
            if vn not in bal:
                continue
            a = np.nan_to_num(np.asarray(bal[vn]), nan=0.0)
            a = a[:, 1].sum(axis=0) if species_dim else a.sum(axis=0)
            acc = a if acc is None else acc + a
        if acc is None:
            raise KeyError(f"missing EIRENE source group for {name}")
        out[name] = _cells(acc, nx, ny)

    neut = None
    if run_dir is not None and (Path(run_dir) / "fort.44").exists():
        neut, _wld = _readers().read_ft44(str(Path(run_dir) / "fort.44"))
    for vn in NEUTRAL_FIELDS:
        if neut is not None and vn in neut:
            v = np.asarray(neut[vn])[..., 0].reshape(-1)  # (nx, ny) native
        elif vn in bal:
            v = np.asarray(bal[vn])[0, 1:-1, 1:-3].T.reshape(-1)
        else:
            v = np.zeros(nx * ny)
        out[vn] = v / EV if vn.startswith("t") else v

    # radiated power density [W/m^3]: EIRENE neutral radiation terms
    # (calc_prad.m convention: these enter Prad with a minus sign). The
    # fluid-species rqrad/rqbrm part needs the b2frates machinery
    # (Matlab-only today) and matters once impurities are seeded.
    rad_w = np.zeros(nx * ny)
    if neut is not None:
        for vn in ("eneutrad", "emolrad", "eionrad"):
            if vn in neut:
                rad_w = rad_w - np.asarray(neut[vn]).sum(axis=-1).reshape(-1)
    vol = _cells(bal["vol"], nx, ny)
    out["prad"] = np.where(vol > 0, rad_w / vol, 0.0)
    return out


def read_case_derived(bal: dict, nx: int, ny: int) -> tuple[dict, dict]:
    """Poloidal heat flux density [W/m^2]: cell-centred q_pol plus
    face-centred target profiles (solpex 'no_jv' method: sum fhe_*+fhi_*
    x-face components, subtract the thermal-current part, divide by
    poloidal face area gs[0], then average the two x-faces per cell)."""
    fht = None
    for vn, arr in bal.items():
        if vn.startswith(("fhe_", "fhi_")) and getattr(arr, "ndim", 0) == 3:
            a = np.asarray(arr)[0]
            fht = a if fht is None else fht + a
    if fht is None or "gs" not in bal:
        raise KeyError("missing fhe_*/fhi_*/gs for heat flux")
    if "fhe_thermj" in bal:
        fht = fht - np.asarray(bal["fhe_thermj"])[0]
    sx = np.asarray(bal["gs"])[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        qx_face = np.where(sx > 0, fht / sx, 0.0)
    qx_cc = 0.5 * (qx_face + np.roll(qx_face, -1, axis=1))
    qx_cc[:, -1] = np.nan
    q_pol = np.nan_to_num(qx_cc, nan=0.0, posinf=0.0, neginf=0.0)
    derived = {"q_pol": _cells(q_pol, nx, ny)}
    targets = {
        "q_inner_target": np.nan_to_num(qx_face[1:-1, 1], posinf=0.0, neginf=0.0),
        "q_outer_target": np.nan_to_num(qx_face[1:-1, -1], posinf=0.0, neginf=0.0),
    }
    return derived, targets


def read_case_inputs(run_dir: str) -> tuple[dict, dict]:
    """Scalar control parameters and status metadata from params.json."""
    meta = json.loads((Path(run_dir) / "params.json").read_text())
    p = meta.get("inputs", {})
    core = p.get("core", {})
    inputs = {
        "pe": p.get("power", {}).get("Pe_W"),
        "pi": p.get("power", {}).get("Pi_W"),
        # canonical name is core_fueling (see data_schema.md). Legacy
        # params.json mislabels it "core.density_m-3"; fixed metadata is
        # expected to use "core.fueling" — both are accepted here.
        "core_fueling": core.get("fueling", core.get("density_m-3")),
    }
    for gas, rec in p.get("gas_puffing", {}).get("targets", {}).items():
        inputs[f"puff_{gas}"] = rec.get("value")
    for key in ("dna", "hci", "hce"):
        if key in p.get("transport", {}):
            inputs[key] = p["transport"][key].get("value")
    inputs = {k: float(v) for k, v in inputs.items() if v is not None}
    return inputs, meta


def convert_case(run_dir: str, mesh: xr.Dataset) -> xr.Dataset:
    """One structured SOLPS run -> canonical case on a prebuilt mesh."""
    nx, ny = int(mesh.attrs["nx"]), int(mesh.attrs["ny"])
    fields, tags = read_case_fields(run_dir, nx, ny)
    inputs, meta = read_case_inputs(run_dir)

    ds = mesh.copy()
    for name, values in fields.items():
        ds[name] = ("cell", values)
    for name, value in inputs.items():
        ds[f"input_{name}"] = value
    ds.attrs.update(
        machine=meta.get("machine", "unknown"),
        case_id=meta.get("case", {}).get("case_id", Path(run_dir).name),
        species=" ".join(tags),
        convergence_status="unclassified",
    )
    return ds
