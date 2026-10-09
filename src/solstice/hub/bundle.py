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
"""Create and load state-model checkpoint bundles (docs/specs/checkpoint_spec.md).

create_state_bundle() packages training checkpoints (solstice-train
best.pt files, one or an ensemble of seeds; or a notebook-saved per-field
mlp_v1 dict) into a self-contained bundle directory (format 0.2: one
member_<k>.safetensors + member_<k>.npz per member, mesh.nc, bundle.json,
optional uq_calibration.json + novelty.npz). load_state_bundle()
reconstructs a StatePredictor that maps raw physical control parameters
to per-cell fields — usable in a clean environment with just solstice
installed (plus torch_geometric for GNN bundles). 0.1 bundles still load.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from solstice.inference.checkpoint import BUNDLE_VERSION

# input feature engineering recorded in bundles (matches the notebooks):
# "sum" adds the parts (ptot = pe + pi); "first" takes one of identical
# channels (chi = hci = hce).
DEFAULT_TRANSFORM = {
    "log_inputs": ["core_fueling", "puff_D2", "puff_Ne"],
    "merged": {"ptot": {"op": "sum", "of": ["pe", "pi"]},
               "chi": {"op": "first", "of": ["hci", "hce"]}},
}


# readable names and units for the manifest ("variables.descriptions"); keys stay the
# SOLPS species convention of the stores (D0 neutral D, D1 = D+, N0..N7 = N, N+ .. N7+)
_SPECIES_LABEL = {"D0": "D0 (neutral D)", "D1": "D+"}
_FIELD_INFO = {
    "te": ("electron temperature", "eV"), "ti": ("ion temperature", "eV"),
    "ne": ("electron density", "m^-3"), "prad": ("radiated power density", "W/m^3"),
    "q_pol": ("poloidal heat flux density", "W/m^2"),
    "q_target": ("heat flux density deposited on the target plates (profile along the plate)", "W/m^2"),
    "gamma_target": ("D+ particle flux density to the target plates (profile along the plate)", "m^-2 s^-1"),
}
_INPUT_INFO = {
    "rmajor": ("major radius of the scaled grid", "m"), "btor": ("toroidal field", "T"),
    "ptot": ("power crossing the core boundary, pe + pi", "W"),
    "pe": ("electron power crossing the core boundary", "W"), "pi": ("ion power crossing the core boundary", "W"),
    "puff_D": ("D gas puff", "atoms/s"), "puff_D2": ("D2 gas puff", "atoms/s"), "puff_N": ("N gas puff", "atoms/s"),
    "puff_Ne": ("Ne gas puff", "atoms/s"), "core_fueling": ("core fueling rate", "atoms/s"),
    "dna": ("anomalous particle diffusivity", "m^2/s"), "chi": ("anomalous heat diffusivity, electrons = ions", "m^2/s"),
    "hci": ("anomalous ion heat diffusivity", "m^2/s"), "hce": ("anomalous electron heat diffusivity", "m^2/s"),
}


def _species_label(sp: str) -> str:
    if sp in _SPECIES_LABEL:
        return _SPECIES_LABEL[sp]
    el, z = sp.rstrip("0123456789"), sp[len(sp.rstrip("0123456789")):]
    return el if z in ("", "0") else (f"{el}+" if z == "1" else f"{el}{z}+")


def describe_variable(name: str) -> dict:
    """{"long_name", "units"} for a store / bundle variable name."""
    if name in _FIELD_INFO:
        ln, u = _FIELD_INFO[name]
    elif name in _INPUT_INFO:
        ln, u = _INPUT_INFO[name]
    elif name.startswith("na_"):
        ln, u = f"{_species_label(name[3:])} density", "m^-3"
    elif name.startswith("ua_"):
        ln, u = f"{_species_label(name[3:])} parallel velocity", "m/s"
    else:
        ln, u = name, ""
    return {"long_name": ln, "units": u}


def _engineer_inputs_batch(raw: dict, transform: dict, names: list[str],
                           x_mean: np.ndarray, x_std: np.ndarray,
                           ranges: dict | None = None, warn: bool = True) -> np.ndarray:
    """Raw physical inputs (scalars or (B,) arrays per name) -> standardized (B, n_in)."""
    import warnings
    feats = {k: np.atleast_1d(np.asarray(v, dtype=np.float64)) for k, v in raw.items()}
    for new, spec in transform.get("merged", {}).items():
        parts = spec["of"] if isinstance(spec, dict) else spec
        op = spec.get("op", "sum") if isinstance(spec, dict) else "sum"
        if new in feats:
            for p in parts:
                feats.pop(p, None)   # canonical DOF given directly; ignore parts
            continue
        if all(p in feats for p in parts):
            vals = np.stack([feats.pop(p) for p in parts])          # (parts, B)
            spread = vals.max(0) - vals.min(0)
            if warn and np.any(spread > 1e-9 * np.abs(vals).max(0)):
                if op == "sum":
                    warnings.warn(
                        f"{'/'.join(parts)} differ, but the model only sees their sum "
                        f"'{new}' (training data had them equal) — the asymmetry is "
                        "ignored. Pass '" + new + "' directly to be explicit.",
                        stacklevel=4)
                else:
                    warnings.warn(
                        f"{'/'.join(parts)} differ, but the model uses a single "
                        f"'{new}' = {parts[0]} (training data had them equal) — "
                        f"{'/'.join(parts[1:])} is ignored. Pass '" + new + "' directly.",
                        stacklevel=4)
            feats[new] = vals.sum(0) if op == "sum" else vals[0]
    # aliases: trained feature name -> canonical raw name (legacy checkpoints)
    for trained, canonical in transform.get("aliases", {}).items():
        if trained not in feats and canonical in feats:
            feats[trained] = feats[canonical]
    missing = [n for n in names if n not in feats]
    if missing:
        raise KeyError(f"missing inputs {missing}; the model needs {names}")
    x = np.stack(np.broadcast_arrays(*[feats[n] for n in names]), axis=1)   # (B, n_in)
    for j, n in enumerate(names):
        if n in transform.get("log_inputs", ()):
            x[:, j] = np.log10(np.clip(x[:, j], 1e-30, None))
    if ranges and warn:
        for j, n in enumerate(names):
            lo, hi = ranges.get(n, (-np.inf, np.inf))
            tol = 1e-6 * (hi - lo) if np.isfinite(hi - lo) else 0.0   # cases on the box edge are inside
            bad = (x[:, j] < lo - tol) | (x[:, j] > hi + tol)
            if bad.any():
                v = x[bad, j][0]
                warnings.warn(
                    f"input '{n}' = {v:.4g} is outside the training range "
                    f"[{lo:.4g}, {hi:.4g}] — the prediction is an extrapolation "
                    "and should not be trusted quantitatively", stacklevel=4)
    return (x - x_mean) / x_std


def _engineer_inputs(raw: dict, transform: dict, names: list[str],
                     x_mean: np.ndarray, x_std: np.ndarray,
                     ranges: dict | None = None) -> np.ndarray:
    """One case of raw physical inputs -> standardized (n_in,)."""
    return _engineer_inputs_batch(raw, transform, names, x_mean, x_std, ranges)[0]


def _transform_from_pt(pt: dict) -> dict:
    """Input feature engineering recorded in a training checkpoint (log inputs from the
    run, the pe+pi / hci=hce merges of solstice.training.data.engineer_inputs)."""
    return {"log_inputs": list(pt.get("log_inputs", DEFAULT_TRANSFORM["log_inputs"])),
            "merged": dict(DEFAULT_TRANSFORM["merged"])}


def mesh_from_store(store_path: str, out_path: str):
    """Write the canonical mesh of a stacked store (every variable without a `case`
    dimension, plus the attributes) as a standalone mesh.nc for a bundle."""
    import xarray as xr
    with xr.open_dataset(store_path) as ds:
        mesh = ds[[v for v in ds.data_vars if "case" not in ds[v].dims]]
        mesh.attrs = dict(ds.attrs)
        mesh.load().to_netcdf(out_path)
    return Path(out_path)


_GRAPH_KEYS = ("node_x", "assign_index", "assign_attr", "latent_edges", "latent_attr",
               "dec_index", "dec_attr")


def _member_arrays(pt: dict, mesh) -> dict:
    """Normalization + latent graph of one training checkpoint, as arrays for member_<k>.npz.
    The graph is rebuilt exactly as in training (solstice.training.data.MeshGraph: node
    features standardized over cells, k-means latent mesh seeded with the run's seed)."""
    from solstice.training.data import MeshGraph
    g = MeshGraph(mesh, n_latent=int(pt["n_latent"]), k_nn=int(pt.get("k_nn", 6)),
                  seed=int(pt.get("seed", 0)), node_features=str(pt.get("node_features", "default")),
                  k_enc=int(pt.get("k_enc", 1)), k_dec=pt.get("k_dec"),
                  multimesh=list(pt.get("multimesh") or []))
    arr = {"x_mean": pt["x_mean"], "x_std": pt["x_std"],
           "y_mean": pt["y_mean"], "y_std": pt["y_std"],
           "node_x": g.node_features, "assign_index": g.assign_index, "assign_attr": g.assign_attr,
           "latent_edges": g.latent_edges, "latent_attr": g.latent_attr,
           "dec_index": g.dec_index, "dec_attr": g.dec_attr}
    for k in ("x_min", "x_max", "field_mask"):
        if pt.get(k) is not None:
            arr[k] = pt[k]
    if pt.get("output_transform", "standard") == "quantile":
        arr["y_quantiles"] = np.asarray(pt["y_quantiles"], dtype=np.float32)
    return arr


def create_state_bundle(pt_path, out_dir: str, name: str, mesh_path: str,
                        provenance: dict | None = None,
                        transform: dict | None = None,
                        input_ranges: dict | None = None,
                        calibration: dict | None = None,
                        train_inputs: dict | None = None,
                        model_card: str | None = None) -> Path:
    """Package one or several training checkpoints (`pt_path`: a path or a list of paths
    of `best.pt` files = the members of a deep ensemble) into a bundle directory.

    Every member keeps its own input/output normalization (each run fits its quantile
    maps on its own training split) and its own latent graph (seeded by the run's seed);
    the predictor averages the members in the fields' metric (log10 for log fields).
    `calibration`: the dict written by solstice.uncertainty.StateEnsemble.calibrate
    (error bar / is-ok flag); `train_inputs`: raw training inputs (dict name -> (n,)) for
    the input-novelty guard that goes with it. `model_card`: markdown text."""
    import torch
    import xarray as xr
    from safetensors.torch import save_file

    paths = [pt_path] if isinstance(pt_path, (str, Path)) else list(pt_path)
    pts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(mesh_path, out / "mesh.nc")

    pt = pts[0]
    members, extra = [], {}
    if "model_class" in pt:  # joint model exported by solstice-train (gnn, mlp2d, ...)
        model_class = pt["model_class"]
        config = dict(pt["config"])
        fields = {k: bool(v) for k, v in pt["fields"].items()}
        aux = dict(pt.get("aux_targets") or {})
        out_tf = pt.get("output_transform", "standard")
        same = ("model_class", "config", "inputs", "fields", "output_transform", "n_latent", "k_nn",
                "node_features", "k_enc", "k_dec", "multimesh", "aux_targets")
        for q, path in zip(pts[1:], paths[1:]):
            for k in same:
                if json.dumps(q.get(k), sort_keys=True, default=str) != json.dumps(pt.get(k), sort_keys=True, default=str):
                    raise ValueError(f"{path}: {k} differs from {paths[0]}; members must share the architecture")
        mesh = xr.open_dataset(out / "mesh.nc")
        n_cells = mesh.sizes["cell"]
        if int(config.get("n_cells") or n_cells) != n_cells:
            raise ValueError(f"mesh has {n_cells} cells, the model {config.get('n_cells')}")
        for k, (q, path) in enumerate(zip(pts, paths)):
            save_file({kk: v.contiguous() for kk, v in q["state_dict"].items()},
                      out / f"member_{k}.safetensors")
            np.savez_compressed(out / f"member_{k}.npz", **_member_arrays(q, mesh))
            members.append({"source": str(path), "seed": int(q.get("seed", 0)),
                            "code_git": q.get("code_git"), "epoch": q.get("epoch"),
                            "val_loss": float(q["val_loss"]) if q.get("val_loss") is not None else None,
                            "train_store": Path(q["train_store"]).name if q.get("train_store") else None})
        mesh.close()
        extra = {"graph": {"n_latent": int(pt["n_latent"]), "k_nn": int(pt.get("k_nn", 6)),
                           "node_features": str(pt.get("node_features", "default")),
                           "k_enc": int(pt.get("k_enc", 1)), "k_dec": int(pt.get("k_dec") or pt.get("k_enc", 1)),
                           "multimesh": [int(m) for m in (pt.get("multimesh") or [])]},
                 # kept for 0.1-style readers
                 "n_latent": int(pt["n_latent"]), "k_nn": int(pt.get("k_nn", 6))}
        transform = transform or _transform_from_pt(pt)
    else:  # per-field MLP dict from the quickstart notebook (single member)
        if len(pts) > 1:
            raise ValueError("per-field mlp_v1 checkpoints bundle one at a time")
        w0 = pt["state_dict"]["0.weight"]
        w4 = pt["state_dict"]["4.weight"]
        model_class = "mlp_v1"
        config = {"in_dim": w0.shape[1], "hidden": w0.shape[0], "out_dim": w4.shape[0]}
        field = Path(paths[0]).stem.rsplit("-", 1)[-1]
        fields, aux, out_tf = {field: bool(pt["log10"])}, {}, "standard"
        save_file({k: v.contiguous() for k, v in pt["state_dict"].items()}, out / "member_0.safetensors")
        arr = {"x_mean": pt["x_mean"], "x_std": pt["x_std"],
               "y_mean": np.asarray(pt["cell_mean"])[:, None], "y_std": np.asarray(pt["cell_std"])[:, None]}
        for k in ("x_min", "x_max"):
            if pt.get(k) is not None:
                arr[k] = pt[k]
        np.savez_compressed(out / "member_0.npz", **{k: np.asarray(v, dtype=np.float64) for k, v in arr.items()})
        members.append({"source": str(paths[0]), "seed": None, "code_git": None, "epoch": None})
        transform = transform or DEFAULT_TRANSFORM

    inputs = pt["inputs"] if isinstance(pt["inputs"], list) else list(pt["inputs"])
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "name": name,
        "task": "state",
        "model": {"class": model_class, "config": config},
        "n_members": len(members),
        "members": members,
        "variables": {
            "inputs": inputs,
            "outputs": list(fields),
            "log10_outputs": fields,
            "aux_targets": aux,          # output channels defined on target-adjacent cells only
            "descriptions": {n: describe_variable(n) for n in [*inputs, *fields]},
        },
        "output_transform": out_tf,
        "input_transform": transform,
        "input_ranges": input_ranges or _ranges_from_pt(pt),
        "uncertainty": None,
        "provenance": {"parent": None, **(provenance or {})},
        "license": "CC-BY-4.0",
        **extra,
    }
    if calibration is not None:
        if train_inputs is None:
            raise ValueError("a calibration needs train_inputs for the novelty guard")
        from solstice.uncertainty import KNN, knn_dist
        (out / "uq_calibration.json").write_text(json.dumps(calibration, indent=1))
        x_train = _engineer_inputs_batch(train_inputs, transform, inputs, np.asarray(pt["x_mean"]),
                                         np.asarray(pt["x_std"]), warn=False)
        ref = np.sort(knn_dist(x_train, x_train, KNN, loo=True))
        np.savez_compressed(out / "novelty.npz", x_train=x_train, ref=ref)
        manifest["uncertainty"] = {"calibration": "uq_calibration.json", "novelty": "novelty.npz",
                                   "protocol": "solstice.uncertainty (calibrated member spread, is-ok flag, kNN novelty)"}
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "model_card.md").write_text(model_card or (
        f"# {name}\n\nSOLSTICE state model bundle ({len(members)} member(s)). See bundle.json for "
        "architecture, variables, provenance, and metrics.\n"))
    return out


class _Member:
    """One ensemble member: network, normalizations, latent graph (tensors on `device`)."""

    def __init__(self, core, in_norm, out_norm, graph: dict, n_latent: int, device):
        self.core, self.in_norm, self.out_norm, self.n_latent, self.device = core, in_norm, out_norm, n_latent, device
        self.g = graph                      # numpy arrays, see _GRAPH_KEYS; None for an MLP
        self.n_cells = out_norm.mean.shape[0]
        self.split_decoder = graph is not None and "dec_index" in graph and graph["dec_index"].shape[1] != self.n_cells
        self._cache: dict = {}

    def batched(self, B: int, torch):
        if B not in self._cache:
            g, nc, nl, dev = self.g, self.n_cells, self.n_latent, self.device
            ai = np.concatenate([g["assign_index"] + np.array([[b * nc], [b * nl]]) for b in range(B)], 1)
            le = np.concatenate([g["latent_edges"] + b * nl for b in range(B)], 1)
            kw = dict(x=torch.tensor(np.tile(g["node_x"], (B, 1)), device=dev),
                      assign_index=torch.tensor(ai, device=dev),
                      assign_attr=torch.tensor(np.tile(g["assign_attr"], (B, 1)), device=dev),
                      latent_edges=torch.tensor(le, device=dev),
                      latent_attr=torch.tensor(np.tile(g["latent_attr"], (B, 1)), device=dev),
                      n_latent=B * nl)
            if self.split_decoder:
                di = np.concatenate([g["dec_index"] + np.array([[b * nl], [b * nc]]) for b in range(B)], 1)
                kw.update(dec_index=torch.tensor(di, device=dev),
                          dec_attr=torch.tensor(np.tile(g["dec_attr"], (B, 1)), device=dev))
            self._cache[B] = kw
        return self._cache[B]


class StatePredictor:
    """Raw physical control parameters -> per-cell fields, from a bundle.

    predict(params)        one case (dict of scalars) -> {field: (cell,)}; auxiliary target
                           channels come back as {"inner_target": profile, "outer_target": profile}
                           ordered along the target (iy).
    predict_batch(raw)     many cases (dict of (B,) arrays) -> mean / std over the members
                           and, when the bundle carries a calibration, the error bar and
                           is-ok flags of solstice.uncertainty.
    members, mesh, manifest, fields (name -> log10?) are attributes."""

    def __init__(self, path: str, device: str = "cpu"):
        import torch
        import xarray as xr
        from safetensors.torch import load_file

        from solstice.inference.checkpoint import SUPPORTED_VERSIONS
        from solstice.models.registry import build_model
        from solstice.training.data import InputNorm, OutputNorm, target_cells

        path = Path(path)
        self.path, self.device = path, device
        self.manifest = m = json.loads((path / "bundle.json").read_text())
        if m["bundle_version"] not in SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported bundle_version {m['bundle_version']}")
        self.mesh = xr.open_dataset(path / "mesh.nc")
        self.fields = {k: bool(v) for k, v in m["variables"]["log10_outputs"].items()}
        self.aux = dict(m["variables"].get("aux_targets") or {})
        self.transform = m["input_transform"]
        self.inputs = list(m["variables"]["inputs"])
        self.is_graph = m["model"]["class"].startswith("gnn")
        self.cells = {"all": np.arange(self.mesh.sizes["cell"])}
        if "face_set" in self.mesh:
            for which in ("inner_target", "outer_target"):
                self.cells[which] = target_cells(self.mesh, which)

        self.members: list[_Member] = []
        legacy = m["bundle_version"] == "0.1"
        n_members = 1 if legacy else int(m.get("n_members", 1))
        for k in range(n_members):
            core = build_model(m["model"]["class"], m["model"]["config"])
            core.load_state_dict(load_file(path / ("weights.safetensors" if legacy else f"member_{k}.safetensors")))
            core.to(device).eval()
            arr = dict(np.load(path / ("normalization.npz" if legacy else f"member_{k}.npz")).items())
            ranges = m.get("input_ranges") or {}
            in_norm = InputNorm(self.inputs, list(self.transform.get("log_inputs", [])),
                                mean=np.asarray(arr["x_mean"], np.float64), std=np.asarray(arr["x_std"], np.float64),
                                lo=np.asarray(arr["x_min"]) if "x_min" in arr else np.array([ranges.get(n, (-np.inf,))[0] for n in self.inputs]),
                                hi=np.asarray(arr["x_max"]) if "x_max" in arr else np.array([ranges.get(n, (None, np.inf))[1] for n in self.inputs]))
            kind = m.get("output_transform", "standard")
            out_norm = OutputNorm({f: v for f, v in self.fields.items() if f not in self.aux}, self.aux or None,
                                  mean=np.asarray(arr["y_mean"], np.float64), std=np.asarray(arr["y_std"], np.float64),
                                  mask=np.asarray(arr["field_mask"], np.float32) if "field_mask" in arr else None,
                                  transform_kind=kind,
                                  n_quantiles=arr["y_quantiles"].shape[-1] if kind == "quantile" else 100,
                                  quantiles=arr["y_quantiles"] if kind == "quantile" else None)
            graph = None
            if self.is_graph:
                graph = ({k: arr[k] for k in _GRAPH_KEYS if k in arr} if not legacy
                         else self._legacy_graph(m))
            self.members.append(_Member(core, in_norm, out_norm, graph, int(m.get("n_latent") or 0), device))
        self.names = list(self.members[0].out_norm.names)
        self.mask = self.members[0].out_norm.mask      # (cell, field) or None

        self.calibration, self._x_train, self._novelty_ref = None, None, None
        unc = m.get("uncertainty")
        if unc:
            self.calibration = json.loads((path / unc["calibration"]).read_text())
            nov = np.load(path / unc["novelty"])
            self._x_train, self._novelty_ref = nov["x_train"], nov["ref"]

    def _legacy_graph(self, m) -> dict:
        """0.1 bundles: default node features, latent graph with seed 0."""
        from solstice.graphs import build_latent_graph, default_node_features
        x = default_node_features(self.mesh)
        x = ((x - x.mean(0)) / (x.std(0) + 1e-12)).astype(np.float32)
        lg = build_latent_graph(self.mesh.cell_r.values, self.mesh.cell_z.values,
                                n_latent=int(m["n_latent"]), k_nn=int(m.get("k_nn", 6)))
        return {"node_x": x, "assign_index": lg["assign_index"].astype(np.int64),
                "assign_attr": (lg["assign_attr"] / (np.abs(lg["assign_attr"]).max(0) + 1e-12)).astype(np.float32),
                "latent_edges": lg["latent_edges"].astype(np.int64),
                "latent_attr": (lg["latent_attr"] / (np.abs(lg["latent_attr"]).max(0) + 1e-12)).astype(np.float32)}

    # -- members --------------------------------------------------------------------------
    def _inputs(self, raw: dict, member: _Member, warn: bool) -> np.ndarray:
        return _engineer_inputs_batch(raw, self.transform, self.inputs, member.in_norm.mean, member.in_norm.std,
                                      ranges=self.manifest.get("input_ranges"), warn=warn).astype(np.float32)

    def _forward(self, member: _Member, Xn: np.ndarray, batch: int = 16) -> np.ndarray:
        """Standardized inputs (B, n_in) -> normalized outputs (B, cell, field) of one member."""
        import torch
        out = []
        with torch.no_grad():
            X = torch.tensor(Xn, device=self.device)
            for i in range(0, len(Xn), batch):
                xb = X[i:i + batch]; B = xb.shape[0]
                if member.g is None:                               # mlp baseline: (B, cell)
                    out.append(member.core(xb).cpu().numpy()[:, :, None])
                    continue
                kw = dict(member.batched(B, torch), params=xb,
                          params_latent=xb.repeat_interleave(member.n_latent, dim=0))
                out.append(member.core(**kw).float().cpu().numpy().reshape(B, member.n_cells, -1))
        return np.concatenate(out)

    def predict_members(self, raw: dict, batch: int = 16, warn: bool = True) -> np.ndarray:
        """(member, case, cell, field) in each field's metric (log10 for log fields)."""
        stack = []
        for k, mem in enumerate(self.members):
            Yn = self._forward(mem, self._inputs(raw, mem, warn and k == 0), batch)
            Y = np.empty(Yn.shape, np.float64)
            for j in range(Yn.shape[-1]):
                Y[..., j] = mem.out_norm.inverse_metric(Yn[..., j], j)
            stack.append(Y)
        return np.stack(stack)

    # -- public API -----------------------------------------------------------------------
    def _to_physical(self, metric: np.ndarray, name: str) -> np.ndarray:
        return 10.0 ** metric if self.fields[name] else metric

    def predict_batch(self, raw: dict, batch: int = 16, return_members: bool = False) -> dict:
        """raw: {input: scalar or (B,)} -> {"mean": {field: (B, cell) physical},
        "std": {field: (B, cell) member spread in the field's metric}, and with a calibrated
        bundle "errbar", "ok", "fuzzy" per field x region, "novel", "in_box", "confidence",
        "use" (solstice.uncertainty.apply_calibration)}. Aux channels are NaN off their cells."""
        S = self.predict_members(raw, batch)
        out = {"mean": {}, "std": {}, "mean_metric": {}}
        for j, n in enumerate(self.names):
            mm, sd = S[..., j].mean(0), S[..., j].std(0)
            if n in self.aux and self.mask is not None:
                off = self.mask[:, j] == 0
                mm[:, off], sd[:, off] = np.nan, np.nan
            out["mean_metric"][n], out["std"][n], out["mean"][n] = mm, sd, self._to_physical(mm, n)
        if return_members:
            out["members"] = {n: S[..., j] for j, n in enumerate(self.names)}
        if self.calibration is not None:
            from solstice.uncertainty import apply_calibration, novelty_check
            X0 = self._inputs(raw, self.members[0], warn=False).astype(np.float64)
            m0 = self.members[0].in_norm
            nov = novelty_check(X0, self._x_train, self._novelty_ref, m0.mean, m0.std, m0.lo, m0.hi)
            out.update(apply_calibration(self.calibration, out["std"], self.cells, nov))
        return out

    def predict(self, params: dict, batch: int = 16) -> dict:
        """One case of raw physical inputs (pe, pi or ptot, chi, puff..., scalars) ->
        {field: (cell,) physical}; the ensemble mean for multi-member bundles. Auxiliary
        target channels: {"inner_target": (n,), "outer_target": (n,)} profiles along iy."""
        r = self.predict_batch(params, batch)
        out = {}
        for n in self.names:
            y = r["mean"][n][0]
            if n in self.aux:
                out[n] = {which: y[self.cells[which]] for which in self.aux[n] if which in self.cells}
            else:
                out[n] = y
        return out

    @property
    def core(self):
        """The first member's network (training init / single-model use)."""
        return self.members[0].core


def load_state_bundle(path: str, device: str = "cpu") -> StatePredictor:
    return StatePredictor(path, device)


def _ranges_from_pt(pt) -> dict | None:
    """Training box per engineered input (x_min/x_max of the checkpoint)."""
    if pt.get("x_min") is None or pt.get("x_max") is None:
        return None
    names = pt["inputs"] if isinstance(pt["inputs"], list) else list(pt["inputs"])
    return {n: [float(lo), float(hi)]
            for n, lo, hi in zip(names, pt["x_min"], pt["x_max"])}


def create_source_bundle(pt_path: str, out_dir: str, name: str, mesh_path: str,
                         provenance: dict | None = None,
                         input_ranges: dict | None = None) -> Path:
    """Package a sources-notebook .pt (plasma state -> EIRENE sources)."""
    import torch

    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if pt.get("task") != "sources":
        raise ValueError(f"{pt_path} is not a sources checkpoint")
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in pt["state_dict"].items()},
              out / "weights.safetensors")
    plasma = {k: bool(v) for k, v in pt["plasma_features"].items()}
    np.savez(out / "normalization.npz",
             y_mean=pt["y_mean"], y_std=pt["y_std"],
             x_mean=pt["x_mean"], x_std=pt["x_std"],
             geom_mean=pt["geom_mean"], geom_std=pt["geom_std"],
             pf_mean=np.array([pt["pf_mean"][k] for k in plasma]),
             pf_std=np.array([pt["pf_std"][k] for k in plasma]))
    shutil.copy(mesh_path, out / "mesh.nc")

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "name": name,
        "task": "sources",
        "model": {"class": pt["model_class"], "config": dict(pt["config"])},
        "variables": {
            "plasma_features": plasma,      # name -> log10, node-feature order
            "outputs": list(pt["sources"]),
            "inputs": list(pt["inputs"]),   # FiLM conditioning params
        },
        "use_params": bool(pt.get("use_params", True)),
        "input_transform": DEFAULT_TRANSFORM,
        "input_ranges": input_ranges or _ranges_from_pt(pt),
        "provenance": {"parent": None, **(provenance or {})},
        "license": "CC-BY-4.0",
        "n_latent": pt.get("n_latent"),
        "k_nn": pt.get("k_nn", 6),
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "model_card.md").write_text(
        f"# {name}\n\nSOLSTICE sources model bundle (EIRENE replacement: local "
        "plasma state -> volumetric neutral sources). See bundle.json.\n")
    return out


class SourcePredictor:
    """Per-cell plasma state (+ control params) -> EIRENE source terms."""

    def __init__(self, path: str):
        import torch
        import xarray as xr
        from safetensors.torch import load_file

        from solstice.models.registry import build_model

        path = Path(path)
        self.manifest = json.loads((path / "bundle.json").read_text())
        if self.manifest["task"] != "sources":
            raise ValueError("not a sources bundle")
        self.core = build_model(self.manifest["model"]["class"],
                                self.manifest["model"]["config"])
        self.core.load_state_dict(load_file(path / "weights.safetensors"))
        self.core.eval()
        self.norm = dict(np.load(path / "normalization.npz").items())
        self.mesh = xr.open_dataset(path / "mesh.nc")
        self._graph = self._build_graph(torch)

    def _build_graph(self, torch):
        from solstice.graphs import build_latent_graph, default_node_features
        geom = default_node_features(self.mesh)
        geom = (geom - self.norm["geom_mean"]) / self.norm["geom_std"]
        g = {"geom": torch.tensor(geom, dtype=torch.float32)}
        n_latent = int(self.manifest["n_latent"])
        lg = build_latent_graph(self.mesh.cell_r.values, self.mesh.cell_z.values,
                                    n_latent=n_latent,
                                    k_nn=int(self.manifest.get("k_nn", 6)))
        g.update(
                assign_index=torch.tensor(lg["assign_index"]),
                assign_attr=torch.tensor(
                    lg["assign_attr"] / (np.abs(lg["assign_attr"]).max(0) + 1e-12),
                    dtype=torch.float32),
                latent_edges=torch.tensor(lg["latent_edges"]),
                latent_attr=torch.tensor(
                    lg["latent_attr"] / np.abs(lg["latent_attr"]).max(0),
                    dtype=torch.float32),
                n_latent=n_latent)
        return g

    def predict(self, plasma: dict, params: dict | None = None) -> dict:
        """plasma: per-cell arrays in physical units, keys = plasma_features.
        params: raw control parameters (required if the bundle uses FiLM)."""
        import torch

        pf = self.manifest["variables"]["plasma_features"]
        feats = [self._graph["geom"].numpy()[:, 0], self._graph["geom"].numpy()[:, 1]]
        for j, (name, is_log) in enumerate(pf.items()):
            a = np.asarray(plasma[name], dtype=np.float64)
            if is_log:
                a = np.log10(np.clip(np.abs(a), 1e-6, None))
            feats.append((a - self.norm["pf_mean"][j]) / self.norm["pf_std"][j])
        x = torch.tensor(np.stack(feats, axis=1), dtype=torch.float32)

        names = self.manifest["variables"]["inputs"]
        if self.manifest["use_params"]:
            if params is None:
                raise ValueError("this bundle conditions on control params")
            xn = _engineer_inputs(params, self.manifest["input_transform"], names,
                                  self.norm["x_mean"], self.norm["x_std"],
                                  ranges=self.manifest.get("input_ranges"))
        else:
            xn = np.zeros(len(names))
        pt = torch.tensor(xn, dtype=torch.float32)

        g = self._graph
        with torch.no_grad():
            pl = pt[None].expand(g["n_latent"], -1)
            yn = self.core(x, g["assign_index"], g["assign_attr"],
                           g["latent_edges"], g["latent_attr"], pl,
                           g["n_latent"]).numpy()
        y = yn * self.norm["y_std"] + self.norm["y_mean"]
        return {name: y[:, j] for j, name in
                enumerate(self.manifest["variables"]["outputs"])}


def load_source_bundle(path: str) -> SourcePredictor:
    return SourcePredictor(path)
