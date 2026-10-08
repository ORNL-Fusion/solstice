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
"""State-model evaluation on a stacked store.

Per field: R^2 and RMSE in normalized units, and median absolute /
relative errors in physical units over all cells, over the
outer-target cells, and at the outer-midplane separatrix cell (when the
store records ix_omp / iy_sep, as the SOLPS-NN store does). The latter
three are the SOLPS-NN paper's Table 2/3 metrics, so a run on the
SOLPS-NN test split is directly comparable to its numbers (Te: 3.3 eV /
6.5 % all cells, 0.97 eV / 12 % outer target, 7.1 eV / 4.3 % OMP
separatrix). Per-regime breakdowns use the store's `regime` column."""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from solstice.training.data import InputNorm, MeshGraph, OutputNorm, target_cells


def load_export(pt_path: str, ds: xr.Dataset, device="cpu"):
    """best.pt + a store with the model's mesh -> (model, in_norm, out_norm, graph)."""
    from solstice.models import build_model

    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = build_model(pt["model_class"], pt["config"])
    model.load_state_dict(pt["state_dict"])
    model.to(device).eval()
    in_norm, out_norm = InputNorm.from_dict(pt), OutputNorm.from_dict(pt)
    graph = MeshGraph(ds, n_latent=int(pt["n_latent"]), k_nn=int(pt.get("k_nn", 6)),
                      seed=int(pt.get("seed", 0)), node_features=str(pt.get("node_features", "default")),
                      k_enc=int(pt.get("k_enc", 1)), k_dec=pt.get("k_dec"),
                      multimesh=list(pt.get("multimesh") or []))
    return model, in_norm, out_norm, graph


@torch.no_grad()
def predict(model, graph: MeshGraph, Xn: np.ndarray, device, batch: int = 32) -> np.ndarray:
    """Normalized predictions (case, cell, field)."""
    model.eval()
    device = torch.device(device)
    out = []
    X = torch.tensor(Xn, device=device)
    for i in range(0, len(Xn), batch):
        B = min(batch, len(Xn) - i)
        g = graph.batch(B, device)
        kw = {k: v for k, v in g.items() if k != "cell_weight"}
        kw["params"] = X[i:i + B]
        kw["params_latent"] = X[i:i + B].repeat_interleave(graph.n_latent, dim=0)
        out.append(model(**kw).float().cpu().numpy().reshape(B, graph.n_cells, -1))
    return np.concatenate(out)


def _med_errors(p: np.ndarray, t: np.ndarray) -> dict:
    err = np.abs(p - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = err / np.abs(t)
    rel = rel[np.isfinite(rel)]
    return {"mae_median": float(np.median(err)),
            "rel_median": float(np.median(rel)) if rel.size else float("nan")}


def omp_sep_cell(ds: xr.Dataset) -> tuple[np.ndarray | None, int | None]:
    """(cells, iy_sep) of the outer-midplane separatrix cell. From the store's ix_omp /
    iy_sep attributes when present (SOLPS-NN release); otherwise from the SOLPS-routines
    region labels: the separatrix row is `region_ids.y.is_sep`, and the outer midplane
    is the cell of that row with the largest major radius (DIII-D store)."""
    import json
    if "ix_omp" in ds.attrs and "cell_ix" in ds:
        iy_sep = int(ds.attrs["iy_sep"])
        sel = (ds["cell_ix"].values == int(ds.attrs["ix_omp"])) & (ds["cell_iy"].values == iy_sep)
        return np.where(sel)[0], iy_sep
    if "region_ids" in ds.attrs and "cell_region_y" in ds:
        y_sep = json.loads(ds.attrs["region_ids"]).get("y", {}).get("is_sep")
        row = np.where(ds["cell_region_y"].values == y_sep)[0] if y_sep is not None else np.array([], int)
        if len(row):
            omp = row[np.argmax(ds["cell_r"].values[row])]
            return np.array([omp]), int(ds["cell_iy"].values[omp])
    return None, None


def evaluate(ds: xr.Dataset, idx: np.ndarray, out_norm: OutputNorm,
             Yn_pred: np.ndarray, Yn_true: np.ndarray) -> dict:
    """Yn_* are (len(idx), cell, field) normalized arrays for the cases `idx` of ds."""
    outer = target_cells(ds, "outer_target")
    omp, iy_sep = omp_sep_cell(ds)
    regime = ds["regime"].values[idx] if "regime" in ds else None

    res: dict = {"n_cases": int(len(idx)), "fields": {}}
    for j, name in enumerate(out_norm.names):
        pn, tn = Yn_pred[:, :, j], Yn_true[:, :, j]
        p, t = out_norm.inverse(pn, j), out_norm.inverse(tn, j)
        if name in out_norm.aux:                     # supervised only on the masked cells
            keep = out_norm.mask[:, j] > 0
            pn, tn, p, t = pn[:, keep], tn[:, keep], p[:, keep], t[:, keep]
            pos = np.cumsum(keep) - 1
            outer_j = pos[outer]
        else:
            outer_j = outer
        f = {"r2_norm": float(1 - ((pn - tn) ** 2).sum() / ((tn - tn.mean()) ** 2).sum()),
             "rmse_norm": float(np.sqrt(((pn - tn) ** 2).mean())),
             "all_cells": _med_errors(p, t),
             "outer_target": _med_errors(p[:, outer_j], t[:, outer_j])}
        if name in out_norm.aux:
            pk_p, pk_t = np.abs(p[:, outer_j]).max(1), np.abs(t[:, outer_j]).max(1)
            f["peak_outer_target"] = {"rel_median": float(np.median(np.abs(pk_p - pk_t) / pk_t))}
            if regime is not None:
                f["peak_outer_target"]["by_regime"] = {
                    r: float(np.median(np.abs(pk_p - pk_t)[regime == r] / pk_t[regime == r]))
                    for r in np.unique(regime)}
            if "pwmxap" in ds and name.startswith("q"):      # the release's own peak heat flux
                ref = ds["pwmxap"].values[idx]
                f["peak_outer_target"]["rel_median_vs_release"] = \
                    float(np.median(np.abs(pk_p - ref) / ref))
            if "fnixap" in ds and name.startswith("gamma") and "outer_target_area" in ds:
                # integrated D+ flux to the outer target vs the release's fnixap
                area = ds["outer_target_area"].values[None] * ds["mesh_scale"].values[idx][:, None] ** 2
                g_p, g_t = (p[:, outer_j] * area).sum(1), ds["fnixap"].values[idx]
                rel = np.abs(g_p - g_t) / np.abs(g_t)
                f["integrated_outer_target"] = {"rel_median_vs_release": float(np.median(rel))}
                if regime is not None:
                    f["integrated_outer_target"]["by_regime"] = {
                        r: float(np.median(rel[regime == r])) for r in np.unique(regime)}
            res["fields"][name] = f
            continue
        if out_norm.fields[name]:
            lp, lt = np.log10(np.clip(p, 1e-300, None)), np.log10(np.clip(t, 1e-300, None))
            f["rmse_dex"] = float(np.sqrt(((lp - lt) ** 2).mean()))
        if omp is not None and len(omp):
            f["omp_sep"] = _med_errors(p[:, omp], t[:, omp])
            f["ot_sep"] = _med_errors(p[:, outer[iy_sep - 1]], t[:, outer[iy_sep - 1]])
        if regime is not None:
            f["by_regime"] = {r: _med_errors(p[regime == r], t[regime == r])
                              for r in np.unique(regime)}
        if name == "q_pol":
            pk_p, pk_t = np.abs(p[:, outer]).max(1), np.abs(t[:, outer]).max(1)
            f["peak_outer_target"] = {"rel_median": float(np.median(np.abs(pk_p - pk_t) / pk_t))}
        res["fields"][name] = f
    return res


def format_metrics(res: dict) -> str:
    lines = [f"{'field':10s} {'R2':>6s} {'RMSEn':>6s} {'all: med abs':>13s} {'rel':>6s} "
             f"{'OT: med abs':>12s} {'rel':>6s} {'OMPsep: abs':>12s} {'rel':>6s}"]
    for name, f in res["fields"].items():
        a, o = f["all_cells"], f["outer_target"]
        m = f.get("omp_sep", {"mae_median": float("nan"), "rel_median": float("nan")})
        lines.append(f"{name:10s} {f['r2_norm']:6.3f} {f['rmse_norm']:6.3f} "
                     f"{a['mae_median']:13.3g} {100 * a['rel_median']:5.1f}% "
                     f"{o['mae_median']:12.3g} {100 * o['rel_median']:5.1f}% "
                     f"{m['mae_median']:12.3g} {100 * m['rel_median']:5.1f}%")
    if "te" in res["fields"] and "by_regime" in res["fields"]["te"]:
        reg = res["fields"]["te"]["by_regime"]
        lines.append("te rel. error by regime: " + ", ".join(
            f"{r} {100 * v['rel_median']:.1f}%" for r, v in reg.items()))
    for name, f in res["fields"].items():
        if "peak_outer_target" in f:
            pk = f["peak_outer_target"]
            line = f"peak outer-target q from {name}: rel. error (median) {100 * pk['rel_median']:.1f}%"
            if "rel_median_vs_release" in pk:
                line += f", vs release pwmxap {100 * pk['rel_median_vs_release']:.1f}%"
            if "by_regime" in pk:
                line += " [" + ", ".join(f"{r} {100 * v:.0f}%" for r, v in pk["by_regime"].items()) + "]"
            lines.append(line)
        if "integrated_outer_target" in f:
            g = f["integrated_outer_target"]
            line = (f"integrated outer-target flux from {name} vs release fnixap: rel. error "
                    f"(median) {100 * g['rel_median_vs_release']:.1f}%")
            if "by_regime" in g:
                line += " [" + ", ".join(f"{r} {100 * v:.0f}%" for r, v in g["by_regime"].items()) + "]"
            lines.append(line)
    return "\n".join(lines)
