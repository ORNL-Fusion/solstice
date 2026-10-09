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
"""Stacked store -> tensors for the state task (params -> fields).

Same feature engineering as the DIII-D training notebook: `input_*`
scalars with constant columns dropped, pe + pi merged into ptot and
hci = hce into chi when the data has them equal, log10 on the
log-uniformly sampled inputs; per-cell, per-field standardization of
the targets (log10 for positive fields). Normalization statistics are
fitted on the training subset only and applied to any store with the
same mesh (e.g. the held-out test split)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from solstice.graphs import build_latent_graph, default_node_features, extended_node_features

LOG_CLIP = 1e-6
QUANTILE_EPS = 1e-7        # clip of the uniform variable before the normal map (as scikit-learn)


def engineer_inputs(ds: xr.Dataset, verbose: bool = False) -> dict[str, np.ndarray]:
    """Raw `input_*` columns -> model degrees of freedom (dict name -> (case,))."""
    raw = {v[6:]: ds[v].values.astype(np.float64) for v in ds.data_vars if v.startswith("input_")}
    for name in list(raw):
        if np.unique(raw[name]).size == 1:
            if verbose:
                print(f"dropping constant input: {name}")
            del raw[name]
    if "pe" in raw and "pi" in raw and np.allclose(raw["pe"], raw["pi"]):
        raw["ptot"] = raw.pop("pe") + raw.pop("pi")
    if "hci" in raw and "hce" in raw and np.allclose(raw["hci"], raw["hce"]):
        raw["chi"] = raw.pop("hci")
        del raw["hce"]
    return raw


@dataclass
class InputNorm:
    names: list[str]
    log_inputs: list[str]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    lo: np.ndarray | None = None      # training box in transformed units
    hi: np.ndarray | None = None

    def _matrix(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        missing = [n for n in self.names if n not in raw]
        if missing:
            raise KeyError(f"store lacks model inputs {missing}; has {sorted(raw)}")
        X = np.stack([np.asarray(raw[n], dtype=np.float64) for n in self.names], axis=1)
        for j, n in enumerate(self.names):
            if n in self.log_inputs:
                X[:, j] = np.log10(np.clip(X[:, j], 1e-30, None))
        return X

    def fit(self, raw: dict[str, np.ndarray], idx: np.ndarray) -> "InputNorm":
        X = self._matrix(raw)[idx]
        self.mean, self.std = X.mean(0), X.std(0) + 1e-12
        self.lo, self.hi = X.min(0), X.max(0)
        return self

    def transform(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        return ((self._matrix(raw) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"inputs": list(self.names), "log_inputs": list(self.log_inputs),
                "x_mean": self.mean, "x_std": self.std, "x_min": self.lo, "x_max": self.hi}

    @classmethod
    def from_dict(cls, d: dict) -> "InputNorm":
        return cls(list(d["inputs"]), list(d.get("log_inputs", [])), mean=np.asarray(d["x_mean"]),
                   std=np.asarray(d["x_std"]), lo=np.asarray(d["x_min"]), hi=np.asarray(d["x_max"]))


@dataclass
class OutputNorm:
    """Per-cell, per-field standardization; log10(|y| clipped) for log fields.

    `transform_kind="quantile"` replaces the standardization by the per-cell
    quantile map to a standard normal of SOLPS-NN (Dasbach & Wiesen 2023;
    scikit-learn QuantileTransformer with n_quantiles knots, output
    "normal"): every cell's marginal over the training cases becomes
    Gaussian whatever its range, sign structure or outliers, and the
    inverse map clips to the training range of the cell. Log fields are
    still taken in log10 first, so the interpolation between knots is in
    log space for multi-decade positive fields (the ranks are unchanged).

    `aux` adds auxiliary target heads (roadmap: the face-centred target
    load profiles learned next to the 2D fields): each entry maps a face
    set to the store's (case, target_iy) profile variable, e.g.
    {"q_target": {"inner_target": "q_inner_target",
                  "outer_target": "q_outer_target", "log": False}}.
    The profile is scattered onto the target-adjacent cells as one more
    output channel; `mask` (cell, field) is 0 where a channel is
    unsupervised, and the loss and metrics use it."""
    fields: dict[str, bool]          # name -> log10?
    aux: dict[str, dict] | None = None
    mean: np.ndarray | None = None   # (cell, field)
    std: np.ndarray | None = None
    mask: np.ndarray | None = None   # (cell, field) float32
    transform_kind: str = "standard"        # "standard" | "quantile"
    n_quantiles: int = 100
    quantiles: np.ndarray | None = None     # (cell, field, n_quantiles), quantile transform only
    names: list[str] = field(init=False)

    def __post_init__(self):
        self.aux = dict(self.aux or {})
        for name, spec in self.aux.items():
            self.fields = {**self.fields, name: bool(spec.get("log", False))}
        self.names = list(self.fields)

    def _aux_raw(self, ds: xr.Dataset, name: str) -> tuple[np.ndarray, np.ndarray]:
        y = np.zeros((ds.sizes["case"], ds.sizes["cell"]))
        m = np.zeros(ds.sizes["cell"], bool)
        for which, var in self.aux[name].items():
            if which == "log":
                continue
            cells = target_cells(ds, which)          # ordered along iy, like target_iy
            prof = ds[var].values
            if prof.shape[1] != len(cells):
                raise ValueError(f"{var}: {prof.shape[1]} profile points but {len(cells)} "
                                 f"{which} cells")
            y[:, cells] = prof
            m[cells] = True
        return y, m

    def _raw(self, ds: xr.Dataset, name: str) -> np.ndarray:
        y = self._aux_raw(ds, name)[0] if name in self.aux else ds[name].values.astype(np.float64)
        if self.fields[name]:
            y = np.log10(np.clip(np.abs(y), LOG_CLIP, None))
        return y

    @property
    def refs(self) -> np.ndarray:
        """Uniform knots in [0, 1] of the quantile transform."""
        return np.linspace(0.0, 1.0, int(self.n_quantiles))

    def fit(self, ds: xr.Dataset, idx: np.ndarray) -> "OutputNorm":
        means, stds, masks, quants = [], [], [], []
        for name in self.names:
            y = self._raw(ds, name)[idx]
            means.append(y.mean(0))
            stds.append(y.std(0) + 1e-12)
            masks.append(self._aux_raw(ds, name)[1] if name in self.aux
                         else np.ones(ds.sizes["cell"], bool))
            if self.transform_kind == "quantile":
                quants.append(np.quantile(y, self.refs, axis=0).T.astype(np.float32))   # (cell, nq)
        self.mean, self.std = np.stack(means, 1), np.stack(stds, 1)
        self.mask = np.stack(masks, 1).astype(np.float32)
        if self.transform_kind == "quantile":
            self.quantiles = np.stack(quants, 1)                                        # (cell, field, nq)
        elif self.transform_kind != "standard":
            raise ValueError(f"output transform must be standard or quantile, got {self.transform_kind!r}")
        return self

    # -- quantile map (per cell; mirrors sklearn's QuantileTransformer incl. the tie handling)
    def _to_normal(self, y: np.ndarray, j: int) -> np.ndarray:
        from scipy.special import ndtri
        q, r = self.quantiles[:, j, :].astype(np.float64), self.refs
        out = np.empty(y.shape, np.float64)
        for c in range(y.shape[-1]):
            u = 0.5 * (np.interp(y[..., c], q[c], r) - np.interp(-y[..., c], -q[c, ::-1], -r[::-1]))
            out[..., c] = u
        z = ndtri(np.clip(out, QUANTILE_EPS, 1 - QUANTILE_EPS))
        return z

    def _from_normal(self, z: np.ndarray, j: int) -> np.ndarray:
        from scipy.special import ndtr
        q, r = self.quantiles[:, j, :].astype(np.float64), self.refs
        u = ndtr(np.asarray(z, np.float64))
        out = np.empty(u.shape, np.float64)
        for c in range(u.shape[-1]):
            out[..., c] = np.interp(u[..., c], r, q[c])
        return out

    def transform(self, ds: xr.Dataset) -> np.ndarray:
        """(case, cell, field) normalized targets, float32 (filled per field
        to keep the float64 intermediate to one field at a time)."""
        Y = np.empty((ds.sizes["case"], ds.sizes["cell"], len(self.names)), np.float32)
        for j, n in enumerate(self.names):
            y = self._raw(ds, n)
            if self.transform_kind == "quantile":
                Y[:, :, j] = self._to_normal(y, j)
            else:
                Y[:, :, j] = (y - self.mean[:, j]) / self.std[:, j]
        return Y

    def forward(self, y: np.ndarray, j: int) -> np.ndarray:
        """Physical (..., cell) values of field j -> normalized (inverse of `inverse`)."""
        y = np.log10(y) if self.fields[self.names[j]] else np.asarray(y, np.float64)
        if self.transform_kind == "quantile":
            return self._to_normal(y, j).astype(np.float32)
        return ((y - self.mean[:, j]) / self.std[:, j]).astype(np.float32)

    def renormalize(self, Yn: np.ndarray, other: "OutputNorm") -> np.ndarray:
        """(..., cell, field) values normalized by `other` -> normalized by self. Each run
        fits its own OutputNorm on its own train split, so ensemble members must be
        brought to one normalization before their predictions are averaged."""
        assert other.names == self.names
        out = np.empty(Yn.shape, np.float32)
        for j in range(len(self.names)):
            out[..., j] = self.forward(other.inverse(Yn[..., j], j), j)
        return out

    def inverse_metric(self, Yn: np.ndarray, j: int) -> np.ndarray:
        """Normalized (..., cell) values of field j -> the field's metric: log10 of the
        physical value for log fields, the physical value otherwise (the space the loss,
        the error statistics and the ensemble mean / spread live in)."""
        if self.transform_kind == "quantile":
            return self._from_normal(Yn, j)
        return Yn * self.std[:, j] + self.mean[:, j]

    def inverse(self, Yn: np.ndarray, j: int) -> np.ndarray:
        """Normalized (..., cell) values of field j -> physical units."""
        y = self.inverse_metric(Yn, j)
        return 10.0 ** y if self.fields[self.names[j]] else y

    def to_dict(self) -> dict:
        d = {"fields": dict(self.fields), "y_mean": self.mean, "y_std": self.std,
             "aux_targets": dict(self.aux), "field_mask": self.mask,
             "output_transform": self.transform_kind}
        if self.transform_kind == "quantile":
            d["y_quantiles"] = self.quantiles
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OutputNorm":
        fields = {k: v for k, v in d["fields"].items() if k not in d.get("aux_targets", {})}
        kind = d.get("output_transform", "standard")
        q = np.asarray(d["y_quantiles"]) if kind == "quantile" else None
        return cls(fields, d.get("aux_targets"), mean=np.asarray(d["y_mean"]),
                   std=np.asarray(d["y_std"]), mask=np.asarray(d.get("field_mask")),
                   transform_kind=kind, n_quantiles=q.shape[-1] if q is not None else 100,
                   quantiles=q)


class MeshGraph:
    """Fixed graph of a stacked store: normalized node features, latent
    mesh, and per-batch replicated index tensors (cached per batch size)."""

    def __init__(self, ds: xr.Dataset, n_latent: int = 256, k_nn: int = 6, seed: int = 0,
                 target_weight: float = 1.0, node_features: str = "default",
                 k_enc: int = 1, k_dec: int | None = None, multimesh=()):
        import torch

        self.n_cells = ds.sizes["cell"]
        self.n_latent = n_latent
        self.node_feature_kind = node_features
        x = extended_node_features(ds) if node_features == "extended" else default_node_features(ds)
        self.node_features = ((x - x.mean(0)) / (x.std(0) + 1e-12)).astype(np.float32)
        lg = build_latent_graph(ds["cell_r"].values, ds["cell_z"].values,
                                n_latent=n_latent, k_nn=k_nn, seed=seed,
                                k_enc=k_enc, k_dec=k_dec, multimesh=multimesh)
        self.graph_extra = {"k_enc": int(k_enc), "k_dec": int(k_dec or k_enc),
                            "multimesh": [int(m) for m in (multimesh or ())]}
        self.split_decoder = int(k_enc) != 1 or int(k_dec or k_enc) != 1
        self.assign_index = lg["assign_index"].astype(np.int64)
        self.assign_attr = (lg["assign_attr"] / (np.abs(lg["assign_attr"]).max(0) + 1e-12)).astype(np.float32)
        self.latent_edges = lg["latent_edges"].astype(np.int64)
        self.latent_attr = (lg["latent_attr"] / (np.abs(lg["latent_attr"]).max(0) + 1e-12)).astype(np.float32)
        self.dec_index = lg["dec_index"].astype(np.int64)
        self.dec_attr = (lg["dec_attr"] / (np.abs(lg["dec_attr"]).max(0) + 1e-12)).astype(np.float32)
        self.cell_weight = target_cell_weights(ds, target_weight).astype(np.float32)
        self._cache: dict = {}
        self._torch = torch

    def batch(self, B: int, device) -> dict:
        key = (B, str(device))
        if key not in self._cache:
            t = self._torch
            nc, nl = self.n_cells, self.n_latent
            ai = np.concatenate([self.assign_index + np.array([[b * nc], [b * nl]]) for b in range(B)], 1)
            le = np.concatenate([self.latent_edges + b * nl for b in range(B)], 1)
            extra = {}
            if self.split_decoder:
                di = np.concatenate([self.dec_index + np.array([[b * nl], [b * nc]]) for b in range(B)], 1)
                extra = dict(dec_index=t.tensor(di, device=device),
                             dec_attr=t.tensor(np.tile(self.dec_attr, (B, 1)), device=device))
            self._cache[key] = dict(**extra,
                x=t.tensor(np.tile(self.node_features, (B, 1)), device=device),
                assign_index=t.tensor(ai, device=device),
                assign_attr=t.tensor(np.tile(self.assign_attr, (B, 1)), device=device),
                latent_edges=t.tensor(le, device=device),
                latent_attr=t.tensor(np.tile(self.latent_attr, (B, 1)), device=device),
                n_latent=B * nl,
                cell_weight=t.tensor(np.tile(self.cell_weight, B), device=device),
            )
        return self._cache[key]


def target_cells(ds: xr.Dataset, which: str) -> np.ndarray:
    """Cells adjacent to a target face set, ordered along iy."""
    fs = json.loads(ds.attrs["face_sets"])
    faces = ds["face_set"].values == fs[which]
    cells = ds["face_cells"].values[faces, 0]
    if "cell_iy" in ds:
        cells = cells[np.argsort(ds["cell_iy"].values[cells])]
    return cells


def target_cell_weights(ds: xr.Dataset, target_weight: float) -> np.ndarray:
    w = np.ones(ds.sizes["cell"])
    if target_weight != 1.0:
        for which in ("inner_target", "outer_target"):
            w[target_cells(ds, which)] = target_weight
        w /= w.mean()
    return w


def select_cases(ds: xr.Dataset, qc: str | None = "qc_pass",
                 exclude_regimes: list[str] | None = None,
                 max_te: float | None = None) -> np.ndarray:
    """Indices of usable cases (data-integrity flag, optional regime / Te cuts)."""
    keep = np.ones(ds.sizes["case"], dtype=bool)
    if qc and qc in ds:
        keep &= ds[qc].values.astype(bool)
    if exclude_regimes and "regime" in ds:
        keep &= ~np.isin(ds["regime"].values, list(exclude_regimes))
    if max_te is not None and "qc_max_te" in ds:
        keep &= ds["qc_max_te"].values <= max_te
    return np.where(keep)[0]


def train_val_split(idx: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    n_val = int(round(val_fraction * len(perm)))
    return perm[n_val:], perm[:n_val]
