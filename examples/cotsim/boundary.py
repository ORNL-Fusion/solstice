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
# SPDX-License-Identifier: Apache-2.0
"""COTSIM boundary conditions from a SOLSTICE state model.

COTSIM's core transport equations end at the SOLPS core boundary rho_c (~0.95) and need
Te, Ti, ne, ni there. SOLPS-ITER (and so the surrogate) resolves the plasma from rho_c
through the separatrix into the SOL on 2D field-aligned mesh rows, each row a flux
surface inside the separatrix. This script

  1. predicts the 2D state for one operating point,
  2. gives each mesh row a flux coordinate (poloidal flux integrated from the stored
     B_pol along the outer midplane, then normalised; see --gfile / --psi-axis-to-sep /
     the default pin),
  3. reduces each closed flux surface to one value by a volume-weighted flux-surface
     average (what COTSIM, a 1.5D code, wants; the outer-midplane sample is reported
     alongside as a diagnostic — at rho_c the two agree, near the separatrix ne differs
     by several %),
  4. interpolates the profiles to rho_c.

    python examples/cotsim_boundary.py bundles/pepc-diiid-state-v2 --rho-c 0.95 \
        --ptot 6e6 --puff-D2 1e21 --core-fueling 3e20 --dna 0.5 --chi 0.7 \
        [--gfile g123456.01000 | --psi-axis-to-sep 0.45] [--coord psin|rhotor] [--out bc.json]

Normalisation of the flux coordinate. B_pol gives psi - psi_sep for every row, but the
magnetic axis lies outside the SOLPS mesh, so psi_N = 1 + (psi - psi_sep)/(psi_sep - psi_axis)
needs psi_sep - psi_axis from the equilibrium: pass an EQDSK g-file (--gfile; also enables
the toroidal-flux coordinate rho_tor via q(psi_N)) or the number itself (--psi-axis-to-sep,
Wb/rad). With neither, the innermost mesh row is *pinned* to rho_c (the SOLPS core
boundary is where the scan was set up, which the REACT note places at ~0.95); the implied
psi_sep - psi_axis is printed so it can be checked against the real equilibrium.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings

import numpy as np


# -- equilibrium ----------------------------------------------------------------------------
_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def read_gfile(path: str) -> dict:
    """Minimal EQDSK reader: psi on axis / boundary [Wb/rad] and q(psi_N) on nw points."""
    lines = open(path).read().splitlines()
    head = lines[0].split()
    nw, nh = int(head[-2]), int(head[-1])
    vals = [float(x.replace("D", "e").replace("d", "e")) for ln in lines[1:] for x in _FLOAT.findall(ln)]
    # 20 scalars, then fpol, pres, ffprim, pprime (nw each), psirz (nw*nh), qpsi (nw)
    simag, sibry = vals[2], vals[3]
    q0 = 20 + 4 * nw + nw * nh
    qpsi = np.array(vals[q0:q0 + nw])
    return {"psi_axis": simag, "psi_sep": sibry, "qpsi": qpsi, "psin_grid": np.linspace(0, 1, nw)}


def rho_tor_from_q(psin: np.ndarray, qpsi: np.ndarray, psin_grid: np.ndarray) -> np.ndarray:
    """rho_tor = sqrt(Phi/Phi_sep), Phi(psi_N) = int_0^psi_N q dpsi_N. Defined for psi_N <= 1
    (NaN outside the separatrix)."""
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (qpsi[1:] + qpsi[:-1]) * np.diff(psin_grid))])
    out = np.full_like(psin, np.nan, dtype=float)
    inside = psin <= 1.0
    out[inside] = np.sqrt(np.interp(psin[inside], psin_grid, phi) / phi[-1])
    return out


# -- mesh geometry ----------------------------------------------------------------------------
def mesh_rows(mesh) -> dict:
    """Row bookkeeping on the structured SOLPS mesh: closed-surface cells per row, the
    outer-midplane column, the separatrix row, and psi - psi_sep per row from B_pol."""
    reg = json.loads(mesh.attrs["region_ids"])
    iy = mesh.cell_iy.values; ix = mesh.cell_ix.values
    R = mesh.cell_r.values; Z = mesh.cell_z.values; vol = mesh.cell_vol.values
    core = mesh.cell_region.values == reg["vol"]["is_core"]          # closed flux surfaces
    sep_rows = np.unique(iy[mesh.cell_region_y.values == reg["y"]["is_sep"]])
    iy_sep = int(sep_rows[0])                                           # first SOL ring
    iy_core_last = int(iy[core].max())                                  # last closed ring
    ix_omp = int(ix[np.argmax(np.where(core, R, -np.inf))])             # largest R on a closed surface
    col = np.where(ix == ix_omp)[0]
    col = col[np.argsort(iy[col])]                                      # OMP column, core -> wall
    rows = iy[col]
    bpol = mesh.cell_b.values[col, 0]
    r = R[col]
    # psi(row) - psi(row 0) from dpsi = R B_pol dR along the outer midplane (R increases outward)
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] * bpol[1:] + r[:-1] * bpol[:-1]) * np.diff(r))])
    # separatrix is the face between the last closed ring and the first SOL ring
    i_in = np.where(rows == iy_core_last)[0][0]; i_out = np.where(rows == iy_sep)[0][0]
    psi_sep = 0.5 * (psi[i_in] + psi[i_out]); r_sep = 0.5 * (r[i_in] + r[i_out])
    return {"rows": rows, "omp_cells": col, "R_omp": r, "Z_omp": Z[col], "R_sep_omp": float(r_sep),
            "dpsi": psi - psi_sep, "iy_sep": iy_sep, "iy_core_last": iy_core_last,
            "closed_cells": {int(k): np.where(core & (iy == k))[0] for k in rows if k <= iy_core_last},
            "vol": vol}


def flux_coordinate(geom: dict, rho_c: float, coord: str, gfile: dict | None, psi_axis_to_sep: float | None) -> dict:
    """psi_N (and rho_tor with a g-file) per OMP row, plus how it was normalised."""
    dpsi = geom["dpsi"]
    if gfile is not None:
        span = abs(gfile["psi_sep"] - gfile["psi_axis"]); how = "g-file"
    elif psi_axis_to_sep is not None:
        span = abs(float(psi_axis_to_sep)); how = "--psi-axis-to-sep"
    else:
        span = dpsi[0] / (rho_c - 1.0); how = f"pinned: innermost row at rho_c={rho_c}"
        if coord == "rhotor":
            raise SystemExit("rho_tor needs q(psi_N): pass --gfile, or use --coord psin")
    psin = 1.0 + dpsi / span
    out = {"psin": psin, "span": float(span), "how": how}
    if gfile is not None:
        out["rhotor"] = rho_tor_from_q(psin, gfile["qpsi"], gfile["psin_grid"])
    out["x"] = out["rhotor"] if coord == "rhotor" else psin
    return out


# -- reduction ----------------------------------------------------------------------------------
def ion_density(fields: dict, names: dict) -> tuple[np.ndarray, str]:
    """Sum of charged-species densities na_<sp><q>, q >= 1. Falls back to ne (with a note)."""
    ions = [k for k in names if re.fullmatch(r"na_[A-Za-z]+[1-9]\d*", k)]
    if not ions:
        return fields["ne"], "ni := ne (bundle has no charged-species density)"
    return sum(fields[k] for k in ions), "ni = " + " + ".join(ions)


def profiles(fields: dict, geom: dict, x: np.ndarray, ni: np.ndarray) -> dict:
    """Per OMP row: flux-surface average (closed rows) and OMP sample of Te, Ti, ne, ni."""
    rows, col, vol = geom["rows"], geom["omp_cells"], geom["vol"]
    out = {"iy": rows.tolist(), "x": x.tolist(), "R_omp": geom["R_omp"].tolist(),
           "dpsi": geom["dpsi"].tolist(), "closed": [bool(k <= geom["iy_core_last"]) for k in rows]}
    src = {"te": fields["te"], "ti": fields["ti"], "ne": fields["ne"], "ni": ni}
    for name, v in src.items():
        fsa = []
        for k in rows:
            cells = geom["closed_cells"].get(int(k))
            if cells is None:
                fsa.append(np.nan)                                      # open field lines: no FSA
            else:
                w = vol[cells]; fsa.append(float((v[cells] * w).sum() / w.sum()))
        out[f"{name}_fsa"] = fsa
        out[f"{name}_omp"] = v[col].tolist()
    return out


def at_rho(prof: dict, rho_c: float) -> dict:
    """Interpolate the FSA profiles (closed rows only) to rho_c; OMP values for reference."""
    x = np.asarray(prof["x"]); closed = np.asarray(prof["closed"])
    xc = x[closed]
    if not (xc.min() - 1e-9 <= rho_c <= xc.max() + 1e-9):
        raise SystemExit(f"rho_c={rho_c} is outside the closed-surface rows of the mesh "
                         f"({xc.min():.4f} .. {xc.max():.4f} in this coordinate); the SOLPS domain "
                         f"does not reach it. Move rho_c inside, or check the normalisation.")
    bc = {}
    for name in ("te", "ti", "ne", "ni"):
        yf = np.asarray(prof[f"{name}_fsa"])[closed]; yo = np.asarray(prof[f"{name}_omp"])[closed]
        bc[name] = {"fsa": float(np.interp(rho_c, xc, yf)), "omp": float(np.interp(rho_c, xc, yo))}
    return bc


def targets(fields: dict, mesh) -> dict:
    """Peak heat load and strike-point Te / ne at the two divertor targets, read from the
    2D fields on the plate-adjacent cells (q_pol there is the cell-centred poloidal heat
    flux density; on the DIII-D store it is within ~5 % of the face heat flux), plus the
    radiated power. The particle flux across rho_c is NOT a state field (see README)."""
    fs = json.loads(mesh.attrs["face_sets"]); reg = json.loads(mesh.attrs["region_ids"])
    fset = mesh.face_set.values; fcell = mesh.face_cells.values; iy = mesh.cell_iy.values
    iy_sep = int(np.unique(iy[mesh.cell_region_y.values == reg["y"]["is_sep"]])[0])
    out = {"P_rad_W": float(np.sum(fields["prad"] * mesh.cell_vol.values))}
    for which in ("inner_target", "outer_target"):
        cells = np.unique(fcell[fset == fs[which]]); cells = cells[cells >= 0]
        cells = cells[np.argsort(iy[cells])]
        q = np.abs(fields["q_pol"][cells]); k = int(np.argmax(q)); sp = cells[iy[cells] == iy_sep][0]
        out[which] = {"q_peak_W_m2": float(q[k]), "iy_peak": int(iy[cells][k]),
                      "Te_strike_eV": float(fields["te"][sp]), "ne_strike_m3": float(fields["ne"][sp]),
                      "q_profile_W_m2": q.tolist(), "iy": iy[cells].tolist()}
    return out


def json_safe(value):
    """Convert NumPy values and non-finite floats to strict-JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


# -- main ---------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", help="state bundle directory (e.g. bundles/pepc-diiid-state-v2) or hub name")
    p.add_argument("--rho-c", type=float, default=0.95, help="COTSIM / SOLPS core boundary (default 0.95)")
    p.add_argument("--coord", choices=["psin", "rhotor"], default="psin",
                   help="flux coordinate for rho_c: normalised poloidal flux, or toroidal rho (needs --gfile)")
    p.add_argument("--gfile", help="EQDSK g-file of the discharge (psi_axis, psi_sep, q profile)")
    p.add_argument("--psi-axis-to-sep", type=float, help="psi_sep - psi_axis [Wb/rad] if no g-file")
    for k, h in [("ptot", "power across the core boundary pe+pi [W]"), ("puff-D2", "D2 gas puff [atoms/s]"),
                 ("core-fueling", "core fueling rate [atoms/s]"), ("dna", "D_perp [m^2/s]"), ("chi", "chi [m^2/s]")]:
        p.add_argument(f"--{k}", type=float, required=True, help=h)
    p.add_argument("--gamma-b-in", type=float, help="net neutral influx at core surface [1/s]")
    p.add_argument("--gamma-b-out", type=float, help="D+ plasma outflux at core boundary [1/s]")
    p.add_argument("--p-b-in", type=float, help="incoming neutral energy at core surface [W]")
    p.add_argument("--p-b-out", type=float, help="plasma power outflux at core boundary [W]")
    p.add_argument("--out", help="write boundary values and profiles to this JSON file")
    p.add_argument("--plot", help="save a profile figure to this path")
    a = p.parse_args(argv)

    from solstice.hub import load_state_bundle
    try:
        model = load_state_bundle(a.bundle)
    except Exception:
        from solstice import hub
        model = hub.load(a.bundle)

    params = {"ptot": a.ptot, "puff_D2": a.puff_D2, "core_fueling": a.core_fueling, "dna": a.dna, "chi": a.chi}
    for cli in ("gamma_b_in", "gamma_b_out", "p_b_in", "p_b_out"):
        value = getattr(a, cli)
        if value is not None:
            params[cli] = value
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pred = model.predict_batch(params)
    for wi in w:
        print(f"warning: {wi.message}", file=sys.stderr)
    fields = {k: v[0] for k, v in pred["mean"].items()}
    ni, ni_note = ion_density(fields, model.fields)

    geom = mesh_rows(model.mesh)
    gf = read_gfile(a.gfile) if a.gfile else None
    fc = flux_coordinate(geom, a.rho_c, a.coord, gf, a.psi_axis_to_sep)
    prof = profiles(fields, geom, fc["x"], ni)
    if "rhotor" in fc:
        prof["rhotor"] = fc["rhotor"].tolist(); prof["psin"] = fc["psin"].tolist()
    bc = at_rho(prof, a.rho_c)
    tg = targets(fields, model.mesh)

    quality = {k: (pred[k][0].item() if hasattr(pred[k][0], "item") else pred[k][0])
               for k in ("use", "confidence", "novel", "in_box") if k in pred}
    res = {"bundle": model.manifest["name"], "inputs": params, "rho_c": a.rho_c, "coord": a.coord,
           "normalisation": {"how": fc["how"], "psi_sep_minus_psi_axis_Wb_per_rad": fc["span"]},
           "separatrix": {"iy_first_sol_row": geom["iy_sep"], "R_omp_m": geom["R_sep_omp"]},
           "boundary_conditions": bc, "ni_definition": ni_note, "targets": tg,
           "cotsim_boundary_inputs": {
               "gamma_b_in": params.get("gamma_b_in"), "gamma_b_out": params.get("gamma_b_out"),
               "p_b_in": params.get("p_b_in"), "p_b_out": params.get("p_b_out"),
               "sign_convention": "positive neutral influx into SOLPS; positive plasma outflux from the core",
           }, "quality": quality, "profiles": prof}

    print(f"{res['bundle']}  rho_c={a.rho_c} ({a.coord}); normalisation: {fc['how']}, "
          f"psi_sep - psi_axis = {fc['span']:.3f} Wb/rad")
    x = np.asarray(prof["x"]); closed = np.asarray(prof["closed"])
    print(f"closed-surface rows span {a.coord} {x[closed].min():.4f} .. {x[closed].max():.4f}; "
          f"SOL rows to {x.max():.4f}")
    print(f"{ni_note}")
    print(f"boundary conditions at rho_c (flux-surface average | outer-midplane sample):")
    units = {"te": "eV", "ti": "eV", "ne": "m^-3", "ni": "m^-3"}
    for k, v in bc.items():
        print(f"  {k:3s} {v['fsa']:12.4e} | {v['omp']:12.4e}  {units[k]}")
    for which in ("inner_target", "outer_target"):
        t = tg[which]
        print(f"{which:13s} q_peak {t['q_peak_W_m2']:10.3e} W/m^2 (iy {t['iy_peak']})  "
              f"Te_strike {t['Te_strike_eV']:7.2f} eV  ne_strike {t['ne_strike_m3']:9.3e} m^-3")
    print(f"P_rad {tg['P_rad_W']:.3e} W  ({tg['P_rad_W'] / a.ptot:.1%} of ptot)")
    if quality:
        print("model quality flags:", quality)

    if a.out:
        with open(a.out, "w") as f:
            json.dump(json_safe(res), f, indent=1, allow_nan=False)
        print("wrote", a.out)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
        for ax, k in zip(axs, ("te", "ti", "ne", "ni")):
            ax.plot(x, prof[f"{k}_omp"], ".-", color="0.6", label="outer midplane")
            ax.plot(x[closed], np.asarray(prof[f"{k}_fsa"])[closed], "o-", ms=4, label="flux-surface avg")
            ax.axvline(1.0, color="k", lw=0.8); ax.axvline(a.rho_c, color="r", ls="--", lw=0.8)
            ax.set_xlabel(a.coord); ax.set_title(f"{k} [{units[k]}]")
            if k in ("ne", "ni"): ax.set_yscale("log")
        axs[0].legend(fontsize=8); fig.suptitle(f"{res['bundle']}  {params}", fontsize=8)
        fig.tight_layout(); fig.savefig(a.plot, dpi=130)
        print("wrote", a.plot)
    return res


if __name__ == "__main__":
    main()
