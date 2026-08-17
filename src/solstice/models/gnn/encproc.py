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
"""gnn: encode-process-decode GNN with a latent mesh (the SOLSTICE GNN).

Encode-process-decode with a latent mesh:
cells are encoded onto a coarse latent
mesh (solstice.graphs.latent), a deep processor runs only there with
FiLM conditioning per layer, and a decoder maps back to cells with a
skip connection from the encoded cell features. Long-range information
travels across the latent mesh instead of one cell per layer.
"""

import torch
import torch.nn as nn

from solstice.models.gnn.layers import BipartiteConv, EdgeConv, NodeFiLM
from solstice.models.registry import register_model


@register_model("gnn")
class GNNEncProcDec(nn.Module):
    def __init__(self, node_features=2, param_dim=5, out_features=4,
                 hidden=128, n_process_layers=8, edge_dim=3, dropout=0.1,
                 film_hidden=128):
        super().__init__()
        self.cell_encoder = nn.Sequential(
            nn.Linear(node_features, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cell_to_latent = BipartiteConv(hidden, hidden, hidden, edge_dim)
        self.latent_init = nn.Parameter(torch.zeros(1, hidden))
        self.processor = nn.ModuleList(
            EdgeConv(hidden, hidden, edge_dim) for _ in range(n_process_layers))
        self.films = nn.ModuleList(
            NodeFiLM(param_dim, hidden, film_hidden) for _ in range(n_process_layers))
        self.latent_to_cell = BipartiteConv(hidden, hidden, hidden, edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, out_features))

    def forward(self, x, assign_index, assign_attr, latent_edges, latent_attr,
                params_latent, n_latent):
        """x: (N_cells, node_features); assign_index: (2, N_cells) cell->latent;
        params_latent: (n_latent, param_dim) params broadcast to latent nodes."""
        hc = self.cell_encoder(x)
        hl = self.latent_init.expand(n_latent, -1)
        hl = self.cell_to_latent(hc, hl, assign_index, assign_attr)
        for layer, film in zip(self.processor, self.films):
            h_new = film(layer(hl, latent_edges, latent_attr), params_latent)
            hl = hl + self.dropout(h_new)
        back_index = torch.flip(assign_index, dims=(0,))
        back_attr = torch.cat([-assign_attr[:, :2], assign_attr[:, 2:]], dim=-1)
        hc_out = self.latent_to_cell(hl, hc, back_index, back_attr)
        return self.decoder(torch.cat([hc, hc_out], dim=-1))
