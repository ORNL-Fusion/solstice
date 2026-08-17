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
"""Checkpoint bundles. Format: docs/specs/checkpoint_spec.md.

A bundle directory must be loadable in a clean environment with just
solstice-fusion installed: bundle.json + weights.safetensors +
normalization.json + mesh.nc + model_card.md.
"""

from __future__ import annotations

import json
from pathlib import Path

BUNDLE_VERSION = "0.1"


def load_checkpoint(path: str):
    """Reconstruct a predictor from a bundle directory."""
    manifest = json.loads((Path(path) / "bundle.json").read_text())
    if manifest.get("task") == "state":
        from solstice.hub.bundle import load_state_bundle
        return load_state_bundle(path)
    if manifest.get("task") == "sources":
        from solstice.hub.bundle import load_source_bundle
        return load_source_bundle(path)
    return _load_generic(path)


def _load_generic(path: str):
    import torch
    from safetensors.torch import load_file

    from solstice.data.schema import load_case
    from solstice.models.base import Normalizer, SourceModel, StateModel
    from solstice.models.registry import build_model

    path = Path(path)
    manifest = json.loads((path / "bundle.json").read_text())
    if manifest["bundle_version"] != BUNDLE_VERSION:
        raise ValueError(
            f"bundle_version {manifest['bundle_version']} != supported {BUNDLE_VERSION}"
        )

    core = build_model(manifest["model"]["class"], manifest["model"]["config"])
    core.load_state_dict(load_file(path / "weights.safetensors"))
    core.eval()

    stats = json.loads((path / "normalization.json").read_text())
    mesh = load_case(str(path / "mesh.nc")) if (path / "mesh.nc").exists() else None

    cls = {"state": StateModel, "sources": SourceModel}[manifest["task"]]
    model = cls(core, Normalizer(stats["inputs"]), Normalizer(stats["outputs"]), mesh=mesh)
    model.manifest = manifest
    return model


def save_checkpoint(path: str, core, manifest: dict, stats: dict, mesh=None) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest, bundle_version=BUNDLE_VERSION)
    (path / "bundle.json").write_text(json.dumps(manifest, indent=2))
    (path / "normalization.json").write_text(json.dumps(stats, indent=2))
    save_file({k: v.contiguous() for k, v in core.state_dict().items()},
              path / "weights.safetensors")
    if mesh is not None:
        mesh.to_netcdf(path / "mesh.nc")
