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
"""Stacked ensemble training store.

Fixed-geometry ensembles (one mesh, many runs) are stored as a single
netcdf: the canonical mesh variables (dim: cell/face/vertex) plus the
canonical field/input variables with a leading `case` dimension. Same
names and units as the per-case schema (docs/specs/data_schema.md) —
only the extra `case` dim differs. Case ids and per-case metadata ride
along as coordinates.

Build from an ensemble directory of SOLPS runs:
    python -m solstice.data.store <ensemble_dir> <out.nc> [--limit N]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from solstice.data.converters import from_solps


QC_MAX_NAN_FRAC = 0.02


def qc_flags(field_names: list[str], fields_arr: np.ndarray) -> dict:
    """Per-case QC columns from the stacked (case, var, cell) field array.

    qc_pass is data integrity only (finite values). qc_max_te/qc_max_ne
    are diagnostics for users to threshold consciously — no physics
    cut is applied here (local Te well below 1 eV is real detached
    physics, and domain-max thresholds are dataset-dependent).
    """
    i_te, i_ne = field_names.index("te"), field_names.index("ne")
    with np.errstate(invalid="ignore"):
        max_te = np.nanmax(fields_arr[:, i_te, :], axis=1)
        max_ne = np.nanmax(fields_arr[:, i_ne, :], axis=1)
    nan_frac = (~np.isfinite(fields_arr)).mean(axis=(1, 2))
    return {
        "qc_max_te": max_te,
        "qc_max_ne": max_ne,
        "qc_nan_frac": nan_frac,
        "qc_pass": nan_frac <= QC_MAX_NAN_FRAC,
    }


def build_ensemble_store(
    ensemble_dir: str,
    out_path: str,
    run_glob: str = "run_*",
    b2fgmtry: str = "baserun/b2fgmtry",
    limit: int | None = None,
    progress: bool = True,
) -> xr.Dataset:
    ensemble_dir = Path(ensemble_dir)
    geo = from_solps.load_geometry(ensemble_dir / b2fgmtry)
    mesh = from_solps.build_structured_mesh(geo)
    nx, ny = int(mesh.attrs["nx"]), int(mesh.attrs["ny"])

    run_dirs = sorted(d for d in ensemble_dir.glob(run_glob) if d.is_dir())
    if limit:
        run_dirs = run_dirs[:limit]

    case_ids, field_rows, input_rows, converged, puff_missing = [], [], [], [], []
    target_rows = []
    field_names, input_names = None, None
    skipped = []
    for i, run in enumerate(run_dirs):
        try:
            bal = from_solps.read_balance_all(run)
            fields, _tags = from_solps.read_case_fields(bal, nx, ny)
            fields.update(from_solps.read_case_sources(bal, nx, ny, run))
            derived, targets = from_solps.read_case_derived(bal, nx, ny)
            fields.update(derived)
            inputs, meta = from_solps.read_case_inputs(run)
        except Exception as err:  # noqa: BLE001 - collect and report bad runs
            skipped.append((run.name, str(err)))
            continue
        if field_names is None:
            field_names = sorted(fields)
        if sorted(fields) != field_names:
            skipped.append((run.name, "field set differs from first run"))
            continue
        target_rows.append(np.stack([targets["q_inner_target"],
                                     targets["q_outer_target"]]))
        case_ids.append(meta.get("case", {}).get("case_id", run.name))
        converged.append(bool(meta.get("case", {}).get("status", {}).get("converged", False)))
        puff_missing.append("puff_D2" not in inputs)
        field_rows.append(np.stack([fields[k] for k in field_names]))
        input_rows.append(inputs)
        if progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(run_dirs)} runs")

    if not case_ids:
        raise RuntimeError(f"no convertible runs in {ensemble_dir}")

    fields_arr = np.stack(field_rows)   # (case, var, cell)
    # union of input names; absent puff_* entries mean zero puff
    input_names = sorted({k for row in input_rows for k in row})
    inputs_arr = np.array(
        [[row.get(k, 0.0 if k.startswith("puff_") else np.nan) for k in input_names]
         for row in input_rows]
    )
    if np.isnan(inputs_arr).any():
        bad = [input_names[j] for j in np.unique(np.where(np.isnan(inputs_arr))[1])]
        raise RuntimeError(f"non-puff inputs missing in some runs: {bad}")

    ds = mesh.copy()
    ds = ds.assign_coords(case=("case", np.asarray(case_ids)))
    for j, name in enumerate(field_names):
        ds[name] = (("case", "cell"), fields_arr[:, j, :].astype(np.float32))
    for j, name in enumerate(input_names):
        ds[f"input_{name}"] = ("case", inputs_arr[:, j])
    ds["params_converged"] = ("case", np.asarray(converged))
    ds["puff_record_missing"] = ("case", np.asarray(puff_missing))
    # face-centred target heat-flux profiles [W/m^2] along iy (1..ny)
    tq = np.stack(target_rows)
    ds["q_inner_target"] = (("case", "target_iy"), tq[:, 0, :].astype(np.float32))
    ds["q_outer_target"] = (("case", "target_iy"), tq[:, 1, :].astype(np.float32))
    if "prad" in field_names:
        j = field_names.index("prad")
        vol = mesh["cell_vol"].values
        ds["prad_tot"] = ("case", (fields_arr[:, j, :] * vol).sum(axis=1))
        ds["prad_tot"].attrs["units"] = "W"
    for name, values in qc_flags(field_names, fields_arr).items():
        ds[name] = ("case", values)
    ds.attrs.update(
        ensemble=ensemble_dir.name,
        n_cases=len(case_ids),
        skipped=json.dumps(skipped),
        store_kind="stacked_ensemble",
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    if progress:
        print(f"wrote {out_path}: {len(case_ids)} cases, {len(skipped)} skipped")
        for name, err in skipped[:10]:
            print(f"  skipped {name}: {err}")
    return ds


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ensemble_dir")
    ap.add_argument("out_path")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    build_ensemble_store(args.ensemble_dir, args.out_path, limit=args.limit)


if __name__ == "__main__":
    main()
