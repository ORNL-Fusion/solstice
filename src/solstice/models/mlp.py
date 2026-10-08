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
"""mlp_v1: per-field MLP baseline (params -> full per-cell field).

Subclasses nn.Sequential with the exact layer layout of the quickstart
notebook so its saved state_dicts load unchanged ('0.weight', ...).
"""

import torch.nn as nn


from solstice.models.registry import register_model


@register_model("mlp_v1")
class MLPv1(nn.Sequential):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )


@register_model("mlp2d")
class MLP2D(nn.Module):
    """SOLPS-NN's NN2D (Dasbach et al.): params -> every cell of every field at
    once through a plain fully connected network (10 x 1000 SELU in the paper).
    Takes the same forward keywords as the GNN and ignores the graph ones."""

    def __init__(self, param_dim, out_features, n_cells, hidden=1000, n_layers=10,
                 activation="selu", dropout=0.0, **_ignored):
        super().__init__()
        act = {"selu": nn.SELU, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[activation]
        layers, d = [], param_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), act()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_cells * out_features)
        self.n_cells, self.out_features = n_cells, out_features

    def forward(self, params, **_ignored):
        """params: (B, param_dim) -> (B * n_cells, out_features)."""
        return self.head(self.body(params)).reshape(-1, self.out_features)
