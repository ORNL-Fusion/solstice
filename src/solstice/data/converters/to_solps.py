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
"""Prediction -> SOLPS-ITER initial state (b2fstati).

The reverse of `from_solps`: take a SOLSTICE state-model prediction (or a
uniform cold start) and write it as the `b2fstati` of a staged copy of a
reference SOLPS-ITER case, so the case can be restarted from it.

Two initial states:

  nn    SOLSTICE prediction. Control parameters are read from the reference
        case's own namelists so the run keeps identical boundary conditions
        and differs only in its initial guess. te, ti, ne, na_D1, ua_D1 are
        mapped onto the reference (ix, iy) mesh through the cell_ix/cell_iy
        labels bundled with the model mesh and spliced into a copy of the
        reference b2fstate; everything the model does not predict (fluxes,
        po, the D0 fluid block, guard cells) is copied verbatim.

  flat  Uniform cold start, what b2ai produces: uniform ne, na, te, ti in
        every cell including guard cells, po = 3.1 * Te (b2ai's floating
        sheath potential), ua = time = 0, every other real field zeroed.
        Species metadata is copied verbatim.

`check_flatness` reports whether an existing b2fstati is flat or a
structured, pre-converged state (a shipped b2fstati often is).

The b2f* text format (`*cf: type nentry name` headers, whitespace-separated
values, Fortran order over (nx+2, ny+2[, ns])) is parsed here at line level
because the writer splices blocks in place; this is the only SOLPS file
parsing in the package not delegated to SOLPS-routines.

Staging copies the reference directory with outputs, history and the old
initial state removed, b2mn.prt removed so `make` re-runs b2mn, and the cd
line in run.csh pointed at the new directory. Mtimes are preserved; copy to
the cluster with `rsync -a` / `scp -p`, or drop only b2fstati into a `cp -a`
clone of the reference, otherwise make tries to regenerate b2frates and
fails on the existing b2ar.prt.

Console script: `solstice-b2fstati`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

EV_TO_J = 1.602176634e-19  # exact CODATA elementary charge
STATE_MODEL_DEFAULT = "pepc-diiid-state-v1"

# Species metadata blocks copied verbatim in every mode.
KEEP_VERBATIM = frozenset({"zamin", "zamax", "zn", "am"})

# Outputs / history from a completed run, never inputs. Removed when staging.
HISTORY_FILES = (
    "b2fstate", "b2fstate.mat", "b2fstati", "b2fstati.bak",
    "b2fstati_with+ua", "b2fstati_with-ua",
    "b2time.nc", "b2tallies.nc", "balance.nc", "b2batch.nc", "b2batch.nc.i",
    "b2fmovie", "b2ftrace", "b2ftrack", "b2mn.prt", "run.log", "output",
)
HISTORY_GLOBS = ("output.*", "b2mn.exe.dir")


# ====================================================================
# b2f text files
# ====================================================================

def b2f_read_dims(path: Path) -> tuple[int, int, int]:
    """(nx, ny, ns) from a b2f* header."""
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) >= 4 and tok[0] == "*cf:" and "nx,ny" in tok[3]:
                vals = next(f).split()
                return int(vals[0]), int(vals[1]), int(vals[2]) if len(vals) > 2 else 1
    raise RuntimeError(f"no nx,ny header in {path}")


def b2f_extract(variable: str, path: Path, reshape: bool = True) -> np.ndarray:
    """One real block. reshape -> (nx+2, ny+2[, ns]) Fortran order, else flat."""
    nx, ny, _ = b2f_read_dims(path)
    nentry, data, in_body = None, [], False
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not in_body:
                if len(tok) >= 4 and tok[0] == "*cf:" and tok[3] == variable:
                    nentry, in_body = int(tok[2]), True
            else:
                data.extend(tok)
                if len(data) >= nentry:
                    break
    if nentry is None:
        raise KeyError(f"variable '{variable}' not found in {path}")
    arr = np.array(data[:nentry], dtype=np.float64)
    if not reshape:
        return arr
    ns = max(nentry // ((nx + 2) * (ny + 2)), 1)
    arr = arr.reshape(nx + 2, ny + 2, ns, order="F")
    return arr[:, :, 0] if ns == 1 else arr


def list_real_blocks(path: Path) -> dict[str, int]:
    out = {}
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if len(tok) >= 4 and tok[0] == "*cf:" and tok[1] == "real":
            out[tok[3]] = int(tok[2])
    return out


def _format_fortran_real(v: float, prec: int = 17, width: int = 26) -> str:
    mantissa, exp = f"{v:.{prec}E}".split("E")
    return f"{mantissa}E{exp[0]}{exp[1:].zfill(3)}".rjust(width)


def _find_block(lines: list[str], varname: str) -> tuple[int, int, int]:
    """(header_idx, data_start_idx, data_end_idx) of a *cf: block."""
    for i, line in enumerate(lines):
        tok = line.split()
        if len(tok) >= 4 and tok[0] == "*cf:" and tok[3] == varname:
            nentry, j, count = int(tok[2]), i + 1, 0
            while count < nentry:
                count += len(lines[j].split())
                j += 1
            return i, i + 1, j
    raise KeyError(f"variable '{varname}' not found")


def _render_block(flat: np.ndarray, per_line: int = 5) -> list[str]:
    return ["".join(_format_fortran_real(v) for v in flat[k:k + per_line]) + "\n"
            for k in range(0, len(flat), per_line)]


def write_spliced_b2fstate(ref_state: Path, out_path: Path, overwrites: dict) -> None:
    """Copy ref_state, replacing the blocks in `overwrites`
    ({name: array in Fortran order, or flat})."""
    lines = Path(ref_state).read_text().splitlines(keepends=True)
    for varname, arr in overwrites.items():
        _, s, e = _find_block(lines, varname)
        lines[s:e] = _render_block(np.asarray(arr).flatten(order="F"))
    Path(out_path).write_text("".join(lines))


def _interior(arr: np.ndarray) -> np.ndarray:
    return arr[1:-1, 1:-1, ...]


# ====================================================================
# check: is this b2fstati flat?
# ====================================================================

def check_flatness(run: Path, fname: str = "b2fstati", verbose: bool = True) -> bool:
    path = Path(run) / fname
    nx, ny, ns = b2f_read_dims(path)
    rows, flat = [], True

    def cv(a):
        return a.std() / abs(a.mean()) if a.mean() != 0 else np.inf

    for name, scale, unit in (("ne", 1.0, "m^-3"), ("te", 1 / EV_TO_J, "eV"),
                              ("ti", 1 / EV_TO_J, "eV")):
        a = (_interior(b2f_extract(name, path)) * scale).ravel()
        rows.append((name, a.min(), a.mean(), a.max(), cv(a), unit))
        flat &= cv(a) <= 1e-6
    na = b2f_extract("na", path)
    na = na[:, :, None] if na.ndim == 2 else na
    for s in range(ns):
        a = _interior(na[:, :, s]).ravel()
        rows.append((f"na[{s}]", a.min(), a.mean(), a.max(), cv(a), "m^-3"))
        flat &= cv(a) <= 1e-6
    ua = _interior(b2f_extract("ua", path)).ravel()
    rows.append(("ua", ua.min(), ua.mean(), ua.max(), np.abs(ua).max(), "m/s (col5=|max|)"))
    flat &= np.abs(ua).max() == 0

    if verbose:
        print(f"{path}  (nx={nx}, ny={ny}, ns={ns})")
        print(f"  header: {path.read_text().splitlines()[0].strip()}")
        print(f"  {'field':8s} {'min':>11s} {'mean':>11s} {'max':>11s} {'std/mean':>9s}  unit")
        for name, mn, me, mx, c, unit in rows:
            print(f"  {name:8s} {mn:11.3e} {me:11.3e} {mx:11.3e} {c:9.3g}  {unit}")
        final = Path(run) / "b2fstate"
        if final.exists() and fname != "b2fstate":
            te_i = _interior(b2f_extract("te", path)).ravel()
            te_f = _interior(b2f_extract("te", final)).ravel()
            ne_i = _interior(b2f_extract("ne", path)).ravel()
            ne_f = _interior(b2f_extract("ne", final)).ravel()
            if te_i.std() > 0 and ne_i.std() > 0:
                print(f"  corr(init, final):  te={np.corrcoef(te_i, te_f)[0, 1]:.4f}  "
                      f"ne={np.corrcoef(ne_i, ne_f)[0, 1]:.4f}")
        print("  =>", "FLAT (uniform initial state)" if flat
              else "NOT flat (structured / pre-converged initial state)", "\n")
    return bool(flat)


# ====================================================================
# nn: SOLSTICE prediction
# ====================================================================

def _expand_fortran_reals(token_str: str) -> list[float]:
    """'2*0.3, 1.0' -> [0.3, 0.3, 1.0]."""
    vals = []
    for tok in token_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "*" in tok:
            n, v = tok.split("*")
            vals.extend([float(v)] * int(n))
        else:
            vals.append(float(tok))
    return vals


def _find_namelist_value(text: str, key: str) -> list[float] | None:
    m = re.search(rf"\b{re.escape(key)}\s*=\s*(.+?)\s*$", text, re.MULTILINE)
    return None if m is None else _expand_fortran_reals(m.group(1).rstrip(","))


def derive_control_params_from_reference(run_path: Path) -> dict:
    """pe, pi, dna, hci, hce, puff_D2, core_fueling from the reference case's
    own namelists. 'core_fueling' (name inherited from the model bundle) is
    the core density boundary condition, conpar(0,1,1)'s second value."""
    run_path = Path(run_path)
    boundary = (run_path / "b2.boundary.parameters").read_text()
    transport = (run_path / "b2.transport.parameters").read_text()
    neutrals = (run_path / "b2.neutrals.parameters").read_text()
    flux_vals = _find_namelist_value(neutrals, "userfluxparm(1,1)")
    return {
        "pe": _find_namelist_value(boundary, "enepar(1,1)")[0],
        "pi": _find_namelist_value(boundary, "enipar(1,1)")[0],
        "dna": _find_namelist_value(transport, "parm_dna")[0],
        "hci": _find_namelist_value(transport, "parm_hci")[0],
        "hce": _find_namelist_value(transport, "parm_hce")[0],
        "puff_D2": max(flux_vals, key=abs) if flux_vals else 0.0,
        "core_fueling": _find_namelist_value(boundary, "conpar(0,1,1)")[1],
    }


def build_index_map(mesh, ref_run: Path) -> tuple[np.ndarray, np.ndarray]:
    """(b2_ix, b2_iy): guard-inclusive b2fstate indices of each model mesh
    cell, checked against the reference b2fgmtry cell centres."""
    gmtry = Path(ref_run) / "b2fgmtry"
    nx, ny, _ = b2f_read_dims(gmtry)
    if int(mesh.attrs["nx"]) != nx or int(mesh.attrs["ny"]) != ny:
        raise ValueError(f"model mesh is {int(mesh.attrs['nx'])}x{int(mesh.attrs['ny'])}, "
                         f"reference case is {nx}x{ny}")
    b2_ix = mesh["cell_ix"].values.astype(int)
    b2_iy = mesh["cell_iy"].values.astype(int)
    crx = b2f_extract("crx", gmtry, reshape=False).reshape(nx + 2, ny + 2, 4, order="F")
    cry = b2f_extract("cry", gmtry, reshape=False).reshape(nx + 2, ny + 2, 4, order="F")
    ref_r = crx[b2_ix, b2_iy, :].mean(axis=1)
    ref_z = cry[b2_ix, b2_iy, :].mean(axis=1)
    dist = np.hypot(ref_r - mesh["cell_r"].values, ref_z - mesh["cell_z"].values)
    rel_err = dist.max() / np.hypot(np.ptp(ref_r), np.ptp(ref_z))
    if rel_err > 1e-3:
        raise ValueError(f"model cells do not line up with the reference geometry at the "
                         f"mapped (ix,iy) (max relative offset {rel_err:.2%})")
    return b2_ix, b2_iy


def build_nn_state(ref_run: Path, model_name: str = STATE_MODEL_DEFAULT,
                   control_overrides: dict | None = None, verbose: bool = True) -> dict:
    """{ne, te, ti, na, ua}: reference b2fstate arrays with the SOLSTICE
    prediction written over the interior cells."""
    from solstice import hub

    ref_run = Path(ref_run)
    control_params = derive_control_params_from_reference(ref_run)
    if control_overrides:
        control_params.update(control_overrides)
    model = hub.load(model_name)
    pred = model.predict(control_params)
    b2_ix, b2_iy = build_index_map(model.mesh, ref_run)

    ref_state = ref_run / "b2fstate"
    fields = {v: b2f_extract(v, ref_state).copy() for v in ("ne", "te", "ti", "na", "ua")}
    fields["ne"][b2_ix, b2_iy] = pred["ne"]
    fields["te"][b2_ix, b2_iy] = pred["te"] * EV_TO_J
    fields["ti"][b2_ix, b2_iy] = pred["ti"] * EV_TO_J
    fields["na"][b2_ix, b2_iy, 1] = pred["na_D1"]
    fields["ua"][b2_ix, b2_iy, 1] = pred["ua_D1"]
    if verbose:
        print("Control params:", control_params)
        for name in ("te", "ti", "ne"):
            p = pred[name]
            print(f"{name}: model min/mean/max = {p.min():.3g}/{p.mean():.3g}/{p.max():.3g}")
        print("na_D0 is not predicted by this model; left at reference values.")
    return fields


def verify_nn_state(ref_state: Path, out_path: Path, touched: set[str]) -> None:
    """Every untouched real block must be unchanged."""
    ref_vars = set(list_real_blocks(ref_state))
    for var in sorted(ref_vars - touched):
        old = b2f_extract(var, ref_state, reshape=False)
        new = b2f_extract(var, out_path, reshape=False)
        if old.shape != new.shape or not np.array_equal(old, new):
            raise AssertionError(f"round-trip check failed: '{var}' changed")


# ====================================================================
# flat: uniform cold start
# ====================================================================

def neutral_density_default(ref_state: Path, ne0: float, n0_frac: float = 1e-3) -> float:
    """Reuse the reference's uniform fluid-neutral floor if it has one
    (EIRENE-coupled cases pin it, e.g. 1e12); else n0_frac * ne0."""
    zamin = b2f_extract("zamin", ref_state, reshape=False)
    na = b2f_extract("na", ref_state)
    na = na[:, :, None] if na.ndim == 2 else na
    for s in np.where(zamin == 0)[0]:
        a = na[:, :, s].ravel()
        if a.std() <= 1e-9 * abs(a.mean()):
            return float(a.mean())
    return n0_frac * ne0


PO_OVER_TE = 3.1  # b2ai: uniform po = 3.1 * Te [V], the floating sheath potential


def build_flat_state(ref_state: Path, ne0: float, te0_eV: float, ti0_eV: float,
                     n0: float, po0: float | None = None) -> dict:
    """{name: flat array} for every real block except species metadata.
    po0 defaults to b2ai's 3.1 * Te."""
    if po0 is None:
        po0 = PO_OVER_TE * te0_eV
    nx, ny, ns = b2f_read_dims(ref_state)
    ncell = (nx + 2) * (ny + 2)
    zamin = b2f_extract("zamin", ref_state, reshape=False)
    charged = zamin > 0
    if charged.sum() == 0:
        raise ValueError("no charged species in zamin; cannot build a quasi-neutral state")
    na0 = np.where(charged, ne0 / float(zamin[charged].sum()), n0)

    values = {}
    for name, n in list_real_blocks(ref_state).items():
        if name in KEEP_VERBATIM:
            continue
        if name == "ne":
            v = np.full(ncell, ne0)
        elif name == "te":
            v = np.full(ncell, te0_eV * EV_TO_J)
        elif name == "ti":
            v = np.full(ncell, ti0_eV * EV_TO_J)
        elif name == "na":
            v = np.repeat(na0, ncell)  # species-major = Fortran (nx+2, ny+2, ns)
        elif name == "po":
            v = np.full(ncell, po0)
        elif name == "time":
            v = np.zeros(1)
        else:  # ua, po, fluxes, drifts, corrections, kinrgy ...
            v = np.zeros(n)
        if v.size != n:
            raise AssertionError(f"size mismatch for {name}: {v.size} vs {n}")
        values[name] = v
    return values


def verify_flat_state(ref_state: Path, out_path: Path) -> None:
    if list_real_blocks(ref_state) != list_real_blocks(out_path):
        raise AssertionError("block layout changed")
    for name in KEEP_VERBATIM:
        if not np.array_equal(b2f_extract(name, ref_state, reshape=False),
                              b2f_extract(name, out_path, reshape=False)):
            raise AssertionError(f"{name} changed")
    ne = b2f_extract("ne", out_path)
    na = b2f_extract("na", out_path)
    na = na[:, :, None] if na.ndim == 2 else na
    zamin = b2f_extract("zamin", ref_state, reshape=False)
    if not np.allclose((na * zamin[None, None, :]).sum(axis=2), ne):
        raise AssertionError("quasi-neutrality violated")
    if not check_flatness(Path(out_path).parent, Path(out_path).name, verbose=False):
        raise AssertionError("flatness check failed")


# ====================================================================
# plots and staging
# ====================================================================

def save_comparison_plot(ref_run: Path, fields: dict, out_png: Path, label: str) -> None:
    """ne / Te / Ti maps, reference b2fstate vs the new initial state."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    ref_run, out_png = Path(ref_run), Path(out_png)
    gmtry, ref_state = ref_run / "b2fgmtry", ref_run / "b2fstate"
    nx, ny, _ = b2f_read_dims(gmtry)
    shape = (nx + 2, ny + 2)
    crx = b2f_extract("crx", gmtry, reshape=False).reshape(nx + 2, ny + 2, 4, order="F")
    cry = b2f_extract("cry", gmtry, reshape=False).reshape(nx + 2, ny + 2, 4, order="F")
    loop = [0, 1, 3, 2]  # SW, SE, NE, NW: non-self-intersecting quad
    polys = np.stack((crx[:, :, loop], cry[:, :, loop]), axis=-1).reshape(-1, 4, 2)

    def full(name):
        return np.asarray(fields[name]).reshape(shape, order="F")

    panels = [
        ("ne (reference)", b2f_extract("ne", ref_state), "inferno", True),
        (f"ne ({label})", full("ne"), "inferno", True),
        ("Te [eV] (reference)", b2f_extract("te", ref_state) / EV_TO_J, "magma", False),
        (f"Te [eV] ({label})", full("te") / EV_TO_J, "magma", False),
        ("Ti [eV] (reference)", b2f_extract("ti", ref_state) / EV_TO_J, "magma", False),
        (f"Ti [eV] ({label})", full("ti") / EV_TO_J, "magma", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (title, arr, cmap, log10) in zip(axes.flat, panels):
        vals = arr.flatten(order="F")
        if log10:
            vals = np.log10(np.clip(vals, 1e-300, None))
        coll = PolyCollection(polys, array=vals, cmap=cmap, edgecolors="none")
        ax.add_collection(coll)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        fig.colorbar(coll, ax=ax, shrink=0.85)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def stage_run_dir(ref_run: Path, out_dir: Path) -> None:
    """Copy of the reference case ready for a fresh b2fstati (see module doc)."""
    ref_run, out_dir = Path(ref_run), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists")
    shutil.copytree(ref_run, out_dir)  # copy2: mtimes preserved
    targets = [out_dir / n for n in HISTORY_FILES]
    for g in HISTORY_GLOBS:
        targets += list(out_dir.glob(g))
    for p in targets:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    rc = out_dir / "run.csh"
    if rc.exists():
        txt = rc.read_text()
        txt2 = re.sub(rf"(/){re.escape(ref_run.name)}(\s|/|$)",
                      rf"\g<1>{out_dir.name}\2", txt, flags=re.M)
        rc.write_text(txt2)
        if txt2 == txt:
            print(f"warning: run.csh does not mention '{ref_run.name}'; edit its cd line")


# ====================================================================
# CLI
# ====================================================================

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="solstice-b2fstati", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", type=Path, action="append", metavar="RUN_DIR",
                   help="only report whether RUN_DIR/b2fstati is flat (repeatable)")
    p.add_argument("--init", choices=["nn", "flat"], help="how to build the initial state")
    p.add_argument("--reference-run", type=Path, help="completed SOLPS-ITER run directory")
    p.add_argument("--output-dir", type=Path,
                   help="default: <reference-run>_solstice_init or _flat_init alongside it")
    p.add_argument("--plot", type=Path, metavar="PNG",
                   help="save ne/Te/Ti comparison maps (needs matplotlib)")
    p.add_argument("--dry-run", action="store_true",
                   help="build + verify, write b2fstati next to --output-dir, don't stage")
    g = p.add_argument_group("--init nn")
    g.add_argument("--model", default=STATE_MODEL_DEFAULT)
    g.add_argument("--control-params", default=None,
                   help='JSON dict overriding control params, e.g. \'{"core_fueling": 3e20}\'')
    g = p.add_argument_group("--init flat")
    g.add_argument("--ne", type=float, help="uniform ne [m^-3] (default: reference interior mean)")
    g.add_argument("--te", type=float, help="uniform Te [eV] (default: reference interior mean)")
    g.add_argument("--ti", type=float, help="uniform Ti [eV] (default: reference interior mean)")
    g.add_argument("--n0", type=float, help="uniform fluid-neutral density [m^-3] (default: "
                   "reference's uniform floor if any, else --n0-frac * ne)")
    g.add_argument("--n0-frac", type=float, default=1e-3)
    g.add_argument("--po", type=float, help="uniform potential [V] (default: 3.1 * Te, as b2ai)")
    args = p.parse_args(argv)

    if args.check:
        for run in args.check:
            check_flatness(run)
        return
    if args.init is None or args.reference_run is None:
        p.error("give --init {nn,flat} with --reference-run, or --check RUN_DIR")

    ref_run = args.reference_run
    ref_state = ref_run / "b2fstate"
    suffix = {"nn": "solstice_init", "flat": "flat_init"}[args.init]
    out_dir = args.output_dir or ref_run.parent / f"{ref_run.name}_{suffix}"

    if args.init == "nn":
        overrides = json.loads(args.control_params) if args.control_params else None
        fields = build_nn_state(ref_run, args.model, overrides)
        label = "solstice"
    else:
        def mean_of(name, scale=1.0):
            return float(_interior(b2f_extract(name, ref_state)).mean() * scale)
        ne0 = args.ne if args.ne is not None else mean_of("ne")
        te0 = args.te if args.te is not None else mean_of("te", 1 / EV_TO_J)
        ti0 = args.ti if args.ti is not None else mean_of("ti", 1 / EV_TO_J)
        n0 = (args.n0 if args.n0 is not None
              else neutral_density_default(ref_state, ne0, args.n0_frac))
        po0 = args.po if args.po is not None else PO_OVER_TE * te0
        print(f"flat state: ne={ne0:.4g} m^-3, Te={te0:.4g} eV, Ti={ti0:.4g} eV, "
              f"fluid-neutral n0={n0:.3g} m^-3, po={po0:.4g} V")
        fields = build_flat_state(ref_state, ne0, te0, ti0, n0, po0)
        label = "flat"

    if args.plot:
        save_comparison_plot(ref_run, fields, args.plot, label)
        print(f"wrote {args.plot}")

    if args.dry_run:
        out_path = out_dir.parent / f"b2fstati.{label}.dry-run"
    else:
        stage_run_dir(ref_run, out_dir)
        out_path = out_dir / "b2fstati"
    write_spliced_b2fstate(ref_state, out_path, fields)
    if args.init == "nn":
        verify_nn_state(ref_state, out_path, set(fields))
    else:
        verify_flat_state(ref_state, out_path)
    print(f"wrote {out_path}" + ("" if args.dry_run else f"; staged run directory {out_dir}"))


if __name__ == "__main__":
    main()
