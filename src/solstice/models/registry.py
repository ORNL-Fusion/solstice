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
"""Model registry: checkpoint bundles name architectures by registry key,
never by python path (docs/specs/checkpoint_spec.md)."""

from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    def wrap(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"model name already registered: {name}")
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return wrap


def get_model(name: str) -> type:
    if name not in _REGISTRY and name.startswith("gnn"):
        # GNNs register on import and need torch_geometric
        import solstice.models.gnn  # noqa: F401
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}") from None


def build_model(name: str, config: dict):
    return get_model(name)(**config)
