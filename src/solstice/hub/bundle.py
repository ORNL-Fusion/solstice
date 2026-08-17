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

create_state_bundle() packages a notebook-saved .pt (per-field mlp_v1
dict or joint gnn dict) into a self-contained bundle directory.
load_state_bundle() reconstructs a StatePredictor that maps raw physical
control parameters to per-cell fields — usable in a clean environment
with just solstice installed (plus torch_geometric for GNN bundles).
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


def _engineer_inputs(raw: dict, transform: dict, names: list[str],
                     x_mean: np.ndarray, x_std: np.ndarray,
                     ranges: dict | None = None) -> np.ndarray:
    feats = dict(raw)
    for new, spec in transform.get("merged", {}).items():
        parts = spec["of"] if isinstance(spec, dict) else spec
        op = spec.get("op", "sum") if isinstance(spec, dict) else "sum"
        if new in feats:
            for p in parts:
                feats.pop(p, None)   # canonical DOF given directly; ignore parts
            continue
        if all(p in feats for p in parts):
            vals = [float(feats.pop(p)) for p in parts]
            spread = max(vals) - min(vals)
            if spread > 1e-9 * max(abs(v) for v in vals):
                import warnings
                if op == "sum":
                    warnings.warn(
                        f"{'/'.join(parts)} differ, but the model only sees their sum "
                        f"'{new}' (training data had them equal) — the asymmetry is "
                        "ignored. Pass '" + new + "' directly to be explicit.",
                        stacklevel=3)
                else:
                    warnings.warn(
                        f"{'/'.join(parts)} differ, but the model uses a single "
                        f"'{new}' = {parts[0]} (training data had them equal) — "
                        f"{'/'.join(parts[1:])} is ignored. Pass '" + new + "' directly.",
                        stacklevel=3)
            feats[new] = sum(vals) if op == "sum" else vals[0]
    # aliases: trained feature name -> canonical raw name (legacy checkpoints)
    for trained, canonical in transform.get("aliases", {}).items():
        if trained not in feats and canonical in feats:
            feats[trained] = feats[canonical]
    x = np.array([float(feats[n]) for n in names], dtype=np.float64)
    for j, n in enumerate(names):
        if n in transform.get("log_inputs", ()):
            x[j] = np.log10(max(x[j], 1e-30))
    if ranges:
        import warnings
        for j, n in enumerate(names):
            lo, hi = ranges.get(n, (-np.inf, np.inf))
            if not (lo <= x[j] <= hi):
                warnings.warn(
                    f"input '{n}' = {x[j]:.4g} is outside the training range "
                    f"[{lo:.4g}, {hi:.4g}] — the prediction is an extrapolation "
                    "and should not be trusted quantitatively", stacklevel=3)
    return (x - x_mean) / x_std


def create_state_bundle(pt_path: str, out_dir: str, name: str, mesh_path: str,
                        provenance: dict | None = None,
                        transform: dict | None = None,
                        input_ranges: dict | None = None) -> Path:
    import torch

    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    if "model_class" in pt:  # joint GNN dict from the gnn notebook
        model_class = pt["model_class"]
        config = dict(pt["config"])
        fields = pt["fields"]
        norm = {"x_mean": pt["x_mean"], "x_std": pt["x_std"],
                "y_mean": pt["y_mean"], "y_std": pt["y_std"]}
        extra = {"n_latent": pt.get("n_latent"), "k_nn": pt.get("k_nn", 6)}
    else:  # per-field MLP dict from the quickstart notebook
        w0 = pt["state_dict"]["0.weight"]
        w4 = pt["state_dict"]["4.weight"]
        model_class = "mlp_v1"
        config = {"in_dim": w0.shape[1], "hidden": w0.shape[0], "out_dim": w4.shape[0]}
        field = Path(pt_path).stem.rsplit("-", 1)[-1]
        fields = {field: bool(pt["log10"])}
        norm = {"x_mean": pt["x_mean"], "x_std": pt["x_std"],
                "y_mean": pt["cell_mean"][:, None], "y_std": pt["cell_std"][:, None]}
        extra = {}

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in pt["state_dict"].items()},
              out / "weights.safetensors")
    np.savez(out / "normalization.npz",
             **{k: np.asarray(v, dtype=np.float64) for k, v in norm.items()})
    shutil.copy(mesh_path, out / "mesh.nc")

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "name": name,
        "task": "state",
        "model": {"class": model_class, "config": config},
        "variables": {
            "inputs": pt["inputs"] if isinstance(pt["inputs"], list) else list(pt["inputs"]),
            "outputs": list(fields),
            "log10_outputs": {k: bool(v) for k, v in fields.items()},
        },
        "input_transform": transform or DEFAULT_TRANSFORM,
        "input_ranges": input_ranges or _ranges_from_pt(pt),
        "provenance": {"parent": None, **(provenance or {})},
        "license": "CC-BY-4.0",
        **extra,
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "model_card.md").write_text(
        f"# {name}\n\nSOLSTICE state model bundle. See bundle.json for "
        "architecture, variables, provenance, and cost.\n")
    return out


class StatePredictor:
    """Raw physical control parameters -> per-cell fields, from a bundle."""

    def __init__(self, path: str):
        import torch
        import xarray as xr
        from safetensors.torch import load_file

        from solstice.models.registry import build_model

        path = Path(path)
        self.manifest = json.loads((path / "bundle.json").read_text())
        if self.manifest["bundle_version"] != BUNDLE_VERSION:
            raise ValueError(f"unsupported bundle_version {self.manifest['bundle_version']}")
        self.core = build_model(self.manifest["model"]["class"],
                                self.manifest["model"]["config"])
        self.core.load_state_dict(load_file(path / "weights.safetensors"))
        self.core.eval()
        self.norm = {k: v for k, v in np.load(path / "normalization.npz").items()}
        self.mesh = xr.open_dataset(path / "mesh.nc")
        self.fields = self.manifest["variables"]["log10_outputs"]

        self._graph = None
        if self.manifest["model"]["class"].startswith("gnn"):
            self._graph = self._build_graph(torch)

    def _build_graph(self, torch):
        from solstice.graphs import build_latent_graph, default_node_features
        x = torch.tensor(default_node_features(self.mesh))
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        g = {"x": x}
        n_latent = int(self.manifest["n_latent"])
        lg = build_latent_graph(self.mesh.cell_r.values, self.mesh.cell_z.values,
                                    n_latent=n_latent,
                                    k_nn=int(self.manifest.get("k_nn", 6)))
        g["assign_index"] = torch.tensor(lg["assign_index"])
        g["assign_attr"] = torch.tensor(
                lg["assign_attr"] / (np.abs(lg["assign_attr"]).max(0) + 1e-12),
                dtype=torch.float32)
        g["latent_edges"] = torch.tensor(lg["latent_edges"])
        g["latent_attr"] = torch.tensor(
                lg["latent_attr"] / np.abs(lg["latent_attr"]).max(0), dtype=torch.float32)
        g["n_latent"] = n_latent
        return g

    def predict(self, params: dict) -> dict:
        """params: raw physical inputs (pe, pi, core_fueling, puff_D2, dna, hci, hce...)."""
        import torch

        names = self.manifest["variables"]["inputs"]
        xn = _engineer_inputs(params, self.manifest["input_transform"], names,
                              self.norm["x_mean"], self.norm["x_std"],
                              ranges=self.manifest.get("input_ranges"))
        xt = torch.tensor(xn, dtype=torch.float32)
        with torch.no_grad():
            if self._graph is None:  # mlp baseline
                yn = self.core(xt[None])[0].numpy()[:, None]
            else:
                g = self._graph
                pl = xt[None].expand(g["n_latent"], -1)
                yn = self.core(g["x"], g["assign_index"], g["assign_attr"],
                               g["latent_edges"], g["latent_attr"], pl,
                               g["n_latent"]).numpy()
        y = yn * self.norm["y_std"] + self.norm["y_mean"]
        out = {}
        for j, (fname, is_log) in enumerate(self.fields.items()):
            out[fname] = 10 ** y[:, j] if is_log else y[:, j]
        return out


def load_state_bundle(path: str) -> StatePredictor:
    return StatePredictor(path)


def _ranges_from_pt(pt) -> dict | None:
    """Training box per engineered input (saved by the notebooks as x_min/x_max)."""
    if "x_min" not in pt or "x_max" not in pt:
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
