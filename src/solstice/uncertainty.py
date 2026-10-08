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
"""Seed-ensemble uncertainty for the state model, with the error-bar / is-ok protocol of
the earlier nn_learner / rf_learner surrogates (ML-AI glue code):

* the prediction is the member mean, the raw error bar the member standard deviation
  (per cell, per field, in the field's metric: log10 for log fields, else physical);
* `calibrate` fits, on cases no member trained on, one scale per field and region
  (mean |error| / mean std) and records the ensemble RMSE there;
* `iserrok` compares the calibrated error bar of a new case with `fussiness` x that
  RMSE: below it the prediction is as good as the model is on held-out data, above
  it the case is unusual and the caller should fall back to the full solver;
* an input-novelty guard flags requests outside the training box or farther from
  the training set than any training case was from its neighbours. Members share
  all data and architecture, so their spread misses shared bias; novelty catches
  the extrapolations the spread cannot.

Released bundles carry the calibration (hub.load(name).predict_batch(raw) returns the same
flags through apply_calibration); here the training-side version over run directories:

    ens = StateEnsemble(sorted(glob("runs/diiid/x_cal/seed*")), mesh_store, device="cuda")
    ens.calibrate(cal_store, cal_idx)                    # held-out cases
    r = ens.predict({"ptot": ..., "chi": ..., ...})      # one or many cases
    r["mean"]["te"], r["std"]["te"]                      # (case, cell) physical / dex
    r["errbar"]["te"]["outer_target"], r["ok"]["te"]["outer_target"], r["use"]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from solstice.training.data import engineer_inputs, select_cases, target_cells
from solstice.training.evaluate import load_export, predict

REGIONS = ("all", "outer_target", "inner_target")
KNN = 5
BOX_TOL = 0.05


def knn_dist(X: np.ndarray, X_train: np.ndarray, k: int = KNN, loo: bool = False) -> np.ndarray:
    """Mean distance of every row of X to its k nearest rows of X_train (normalized units);
    loo=True when X is X_train itself (a row is not its own neighbour)."""
    D = np.sqrt(((X[:, None, :] - X_train[None, :, :]) ** 2).sum(-1))
    if loo:
        np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, :k].mean(1)


def novelty_check(X: np.ndarray, X_train: np.ndarray, ref: np.ndarray,
                  mean: np.ndarray, std: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Input-novelty guard for standardized inputs X (B, n_in): kNN distance to the training
    inputs, its rank among the training cases' own leave-one-out distances `ref` (sorted),
    and the training box (lo/hi in transformed units) with a BOX_TOL tolerance per input."""
    d = knn_dist(X.astype(np.float64), X_train)
    score = np.searchsorted(ref, d) / len(ref)   # fraction of training cases closer to their neighbours
    # a case marginally past the extreme training sample (as LHS test cases are) is not an
    # extrapolation, a 50 % one is
    Xr = X * std + mean
    tol = BOX_TOL * (hi - lo)
    in_box = np.all((Xr >= lo - tol) & (Xr <= hi + tol), axis=1)
    return {"knn_dist": d, "score": score, "in_box": in_box}


def case_stat(std: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """One error-bar number per case: RMS of the per-cell std over the region."""
    return np.sqrt(np.nanmean(std[:, cells] ** 2, axis=1))


def apply_calibration(cal: dict, std: dict, cells: dict, nov: dict) -> dict:
    """Calibrated std, errbar / ok / fuzzy per field and region, novelty flags, confidence
    and the `use` bit, from the raw member spread `std` {field: (B, cell) metric}, the
    region cell indices and a novelty_check() result. The protocol of StateEnsemble.predict."""
    out = {"std": {}, "errbar": {}, "ok": {}, "fuzzy": {}, "novelty": nov,
           "novel": nov["knn_dist"] > cal["novelty_max"], "in_box": nov["in_box"]}
    use = ~out["novel"] & out["in_box"]
    for n, f in cal["fields"].items():
        if n not in std:
            continue
        out["std"][n] = std[n] * f["all"]["scale"]
        out["errbar"][n], out["ok"][n], out["fuzzy"][n] = {}, {}, {}
        for reg in f:
            if reg not in cells:
                continue
            eb = case_stat(std[n], cells[reg]) * f[reg]["scale"]
            out["errbar"][n][reg] = eb
            out["fuzzy"][n][reg] = eb / f[reg]["thresh"]
            out["ok"][n][reg] = eb < f[reg]["thresh"]
            if n in cal["use_fields"] and reg in cal["use_regions"]:
                use &= out["ok"][n][reg]
    # one dimensionless number per case: the worst fuzzy ratio over the key fields and
    # regions (1 = at the threshold), and confidence = 1 / that, clipped to [0, 1]
    out["fuzzy_max"] = np.max([out["fuzzy"][n][reg] for n in cal["use_fields"] for reg in cal["use_regions"]], axis=0)
    out["confidence"] = np.where(out["novel"] | ~out["in_box"], 0.0,
                                 np.clip(1.0 / np.maximum(out["fuzzy_max"], 1e-12), 0.0, 1.0))
    out["use"] = use
    return out


class StateEnsemble:
    def __init__(self, runs, ds: xr.Dataset, device="cpu", train_store: str | None = None):
        self.runs = [Path(r) for r in runs]
        self.device = device
        self.ds = ds
        self.members = [load_export(r / "best.pt", ds, device) for r in self.runs]
        self.in_norm = self.members[0][1]
        self.names = list(self.members[0][2].names)
        self.log = {n: bool(self.members[0][2].fields[n]) for n in self.names}
        self.cells = {"all": np.arange(ds.sizes["cell"]),
                      "outer_target": target_cells(ds, "outer_target"),
                      "inner_target": target_cells(ds, "inner_target")}
        self.calibration: dict | None = None
        # training inputs (normalized by member 0) for the novelty guard
        import torch
        pt = torch.load(self.runs[0] / "best.pt", map_location="cpu", weights_only=False)
        ts = xr.open_dataset(train_store or pt["train_store"])
        self.X_train = self.in_norm.transform(engineer_inputs(ts))[select_cases(ts)]
        ts.close()
        d = self._knn_dist(self.X_train, loo=True)
        self.novelty_ref = np.sort(d)

    # -- raw ensemble ---------------------------------------------------------------------
    def _metric(self, y: np.ndarray, name: str) -> np.ndarray:
        return np.log10(np.clip(np.abs(y), 1e-30, None)) if self.log[name] else y

    def _knn_dist(self, X: np.ndarray, loo: bool = False) -> np.ndarray:
        """Mean distance to the KNN nearest training inputs (normalized units)."""
        return knn_dist(X, self.X_train, KNN, loo)

    def predict_raw(self, raw: dict[str, np.ndarray], batch: int = 32) -> dict:
        """Member mean / std per field: {'mean': phys (case, cell), 'std': metric (case, cell),
        'mean_metric': (case, cell)} plus the per-member metric stack."""
        raw = {k: np.atleast_1d(np.asarray(v, np.float64)) for k, v in raw.items()}
        stack = {n: [] for n in self.names}
        for model, in_norm, out_norm, graph in self.members:
            Pr = predict(model, graph, in_norm.transform(raw), self.device, batch)
            for j, n in enumerate(self.names):
                stack[n].append(self._metric(out_norm.inverse(Pr[..., j], j), n))
        stack = {n: np.stack(v) for n, v in stack.items()}                      # (member, case, cell)
        mean_m = {n: v.mean(0) for n, v in stack.items()}
        std = {n: v.std(0) for n, v in stack.items()}
        mean = {n: (10.0 ** mean_m[n] if self.log[n] else mean_m[n]) for n in self.names}
        return {"mean": mean, "mean_metric": mean_m, "std": std, "members": stack}

    def novelty(self, raw: dict[str, np.ndarray]) -> dict:
        X = self.in_norm.transform({k: np.atleast_1d(np.asarray(v, np.float64)) for k, v in raw.items()})
        n = self.in_norm
        return novelty_check(X.astype(np.float64), self.X_train, self.novelty_ref, n.mean, n.std, n.lo, n.hi)

    # -- calibration (nn_learner.Model.calibrate, per field and region) --------------------
    @staticmethod
    def _case_stat(std: np.ndarray, cells: np.ndarray) -> np.ndarray:
        return case_stat(std, cells)

    def calibrate(self, ds: xr.Dataset, idx: np.ndarray, fussiness: float = 1.0,
                  scale_method: str = "quantile", coverage: float = 0.68,
                  novelty_quantile: float = 0.99, use_fields: list[str] | None = None,
                  use_regions: list[str] | None = None) -> dict:
        """scale_method "mean": scale = mean |err| / mean std (nn_learner.calibrate);
        "quantile": scale = the `coverage` quantile of |err| / std over the held-out cells, so
        that fraction of errors falls inside one calibrated std there (split conformal).
        `use_fields` / `use_regions`: which ok flags the `use` bit combines (default all)."""
        raw = {k: v[idx] for k, v in engineer_inputs(ds).items()}
        r = self.predict_raw(raw)
        cal = {"fussiness": float(fussiness), "scale_method": scale_method, "coverage": float(coverage),
               "n_cases": int(len(idx)), "knn": KNN,
               "novelty_max": float(np.quantile(self.novelty_ref, novelty_quantile)),
               "use_fields": list(use_fields or self.names), "use_regions": list(use_regions or REGIONS),
               "fields": {}}
        for n in self.names:
            if n not in ds:        # auxiliary target channels have no per-cell truth in the store
                continue
            truth = self._metric(ds[n].values[idx], n)
            err = np.abs(r["mean_metric"][n] - truth)
            cal["fields"][n] = {}
            for reg in REGIONS:
                c = self.cells[reg]
                sd, e = r["std"][n][:, c], err[:, c]
                if scale_method == "mean":
                    scale = float(e.mean() / max(sd.mean(), 1e-12))
                elif scale_method == "quantile":
                    ok = sd > 0
                    scale = float(np.quantile(e[ok] / sd[ok], coverage))
                else:
                    raise ValueError(f"scale_method must be mean or quantile, got {scale_method!r}")
                rmse = float(np.sqrt((e ** 2).mean()))
                # case-level reference: RMS over the region of the calibrated std, on the held-out cases
                eb = self._case_stat(sd, np.arange(len(c))) * scale
                cal["fields"][n][reg] = {"scale": scale, "rmse": rmse, "thresh": fussiness * rmse,
                                         "errbar_median_heldout": float(np.median(eb)),
                                         "unit": "dex" if self.log[n] else "phys"}
        self.calibration = cal
        return cal

    def save_calibration(self, path):
        Path(path).write_text(json.dumps(self.calibration, indent=1))

    def load_calibration(self, path):
        self.calibration = json.loads(Path(path).read_text())
        return self.calibration

    # -- calibrated prediction with the is-ok flag ------------------------------------------
    def predict(self, raw: dict[str, np.ndarray], batch: int = 32) -> dict:
        """mean (phys), std (calibrated, metric), errbar / ok / fuzzy per field and region,
        novelty, and `use` (every field ok in every region, and not novel)."""
        if self.calibration is None:
            raise RuntimeError("call calibrate() or load_calibration() first")
        r = self.predict_raw(raw, batch)
        out = {"mean": r["mean"], "mean_metric": r["mean_metric"]}
        out.update(apply_calibration(self.calibration, r["std"], self.cells, self.novelty(raw)))
        return out

    # -- validation on cases with truth ------------------------------------------------------
    def evaluate_flags(self, ds: xr.Dataset, idx: np.ndarray) -> dict:
        """Does the flag separate good from bad predictions? Per field and region: the case
        RMSE of flagged-ok vs flagged-bad cases, the fraction flagged, the rank correlation
        of errbar with the actual case RMSE, and the coverage of the calibrated per-cell std."""
        from scipy.stats import spearmanr
        raw = {k: v[idx] for k, v in engineer_inputs(ds).items()}
        p = self.predict(raw)
        res = {"n_cases": int(len(idx)), "use_fraction": float(p["use"].mean()),
               "novel_fraction": float(p["novel"].mean()), "fields": {}}
        for n in self.names:
            if n not in ds:
                continue
            truth = self._metric(ds[n].values[idx], n)
            err = np.abs(self._metric(p["mean"][n], n) - truth)
            res["fields"][n] = {}
            for reg in REGIONS:
                c = self.cells[reg]
                case_rmse = np.sqrt((err[:, c] ** 2).mean(1))
                ok = p["ok"][n][reg]
                sd = p["std"][n][:, c]; e = err[:, c]
                res["fields"][n][reg] = {
                    "ok_fraction": float(ok.mean()),
                    "rmse_ok": float(np.sqrt((case_rmse[ok] ** 2).mean())) if ok.any() else None,
                    "rmse_bad": float(np.sqrt((case_rmse[~ok] ** 2).mean())) if (~ok).any() else None,
                    "spearman_errbar_vs_rmse": float(spearmanr(p["errbar"][n][reg], case_rmse).correlation),
                    "coverage_1std": float((e < sd).mean()), "coverage_2std": float((e < 2 * sd).mean()),
                }
        return res
