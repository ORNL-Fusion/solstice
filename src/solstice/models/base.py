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
"""Task interfaces. A ModelInterface owns normalization (pre/post) and a
core network; predict() is the stable user-facing signature per task.

Tasks:
  state:   control params (+ mesh)  -> plasma fields (te, ti, ne, ua, ...)
  sources: plasma state  (+ mesh)   -> neutral sources (sp, sne, qe, qi, sm)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

TASKS = ("state", "sources")


class Normalizer:
    """Per-variable affine normalization from bundle statistics."""

    def __init__(self, stats: dict):
        self.stats = stats

    def encode(self, name, values):
        s = self.stats[name]
        return (values - s["mean"]) / s["std"]

    def decode(self, name, values):
        s = self.stats[name]
        return values * s["std"] + s["mean"]


class ModelInterface(ABC):
    task: str

    def __init__(self, core, in_norm: Normalizer, out_norm: Normalizer, mesh=None):
        self.core = core
        self.in_norm = in_norm
        self.out_norm = out_norm
        self.mesh = mesh

    @abstractmethod
    def predict(self, inputs: dict, mesh=None) -> dict:
        """Named inputs -> named per-cell outputs, physical units."""


class StateModel(ModelInterface):
    task = "state"


class SourceModel(ModelInterface):
    task = "sources"
