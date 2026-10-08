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

from solstice.models.gnn.layers import BipartiteConv, EdgeConv, InteractionLayer, NodeFiLM
from solstice.models.registry import register_model


@register_model("gnn")
class GNNEncProcDec(nn.Module):
    def __init__(self, node_features=2, param_dim=5, out_features=4,
                 hidden=128, n_process_layers=8, edge_dim=3, dropout=0.1,
                 film_hidden=128, n_cells=None, cell_embed=0, params_to_cells=False,
                 decoder_hidden=None, processor="edgeconv", aggr="sum",
                 head_index=None, head_hidden=None):
        """cell_embed > 0: a learnable embedding of that width per mesh cell (needs
        n_cells) joins the node features, giving every cell its own parameters as
        the per-cell output weights of an MLP do. params_to_cells: the control
        parameters also enter the cell encoder and the decoder directly, not only
        through the FiLM layers of the latent processor. processor: "edgeconv"
        (original: fixed geometric edge features, mean aggregation) or
        "interaction" (GraphCast interaction network: learned edge latents with
        residual updates, LayerNorm MLPs, `aggr` aggregation). head_index: list of
        output-channel index lists, one decoder head (width head_hidden, default
        decoder_hidden) per group on the shared processor; None = one decoder."""
        super().__init__()
        self.cell_embed = int(cell_embed)
        self.params_to_cells = bool(params_to_cells)
        self.n_cells = n_cells
        if self.cell_embed:
            if not n_cells:
                raise ValueError("cell_embed needs n_cells")
            self.embed = nn.Parameter(torch.randn(n_cells, self.cell_embed) * 0.1)
        p_in = param_dim if self.params_to_cells else 0
        self.cell_encoder = nn.Sequential(
            nn.Linear(node_features + self.cell_embed + p_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cell_to_latent = BipartiteConv(hidden, hidden, hidden, edge_dim)
        self.latent_init = nn.Parameter(torch.zeros(1, hidden))
        self.processor_kind = str(processor)
        if self.processor_kind == "interaction":
            self.edge_embed = nn.Sequential(nn.Linear(edge_dim, hidden), nn.SiLU(),
                                            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
            self.processor = nn.ModuleList(
                InteractionLayer(hidden, aggr) for _ in range(n_process_layers))
        elif self.processor_kind == "edgeconv":
            self.processor = nn.ModuleList(
                EdgeConv(hidden, hidden, edge_dim) for _ in range(n_process_layers))
        else:
            raise ValueError(f"unknown processor {processor!r}")
        self.films = nn.ModuleList(
            NodeFiLM(param_dim, hidden, film_hidden) for _ in range(n_process_layers))
        self.latent_to_cell = BipartiteConv(hidden, hidden, hidden, edge_dim)
        self.dropout = nn.Dropout(dropout)
        dh = int(decoder_hidden or hidden)
        self.out_features = out_features
        self.head_index = [list(map(int, g)) for g in head_index] if head_index else None
        if self.head_index:
            covered = sorted(i for g in self.head_index for i in g)
            if covered != list(range(out_features)):
                raise ValueError("head_index must cover every output channel exactly once")
            hh = int(head_hidden or dh)
            self.decoders = nn.ModuleList(self._mlp(2 * hidden + p_in, hh, len(g)) for g in self.head_index)
            for k, g in enumerate(self.head_index):
                self.register_buffer(f"head_idx{k}", torch.tensor(g), persistent=False)
        else:
            self.decoder = self._mlp(2 * hidden + p_in, dh, out_features)

    @staticmethod
    def _mlp(n_in, width, n_out):
        return nn.Sequential(nn.Linear(n_in, width), nn.SiLU(),
                             nn.Linear(width, width // 2), nn.SiLU(),
                             nn.Linear(width // 2, n_out))

    def forward(self, x, assign_index, assign_attr, latent_edges, latent_attr,
                params_latent, n_latent, params=None, dec_index=None, dec_attr=None, **_ignored):
        """x: (N_cells, node_features); assign_index: (2, N_cells) cell->latent;
        params_latent: (n_latent, param_dim) params broadcast to latent nodes;
        params: (B, param_dim), needed when params_to_cells."""
        extra = []
        if self.cell_embed:
            B = x.shape[0] // self.n_cells
            extra.append(self.embed.repeat(B, 1))
        if self.params_to_cells:
            pc = params.repeat_interleave(x.shape[0] // params.shape[0], dim=0)
            extra.append(pc)
        hc = self.cell_encoder(torch.cat([x, *extra], dim=-1) if extra else x)
        hl = self.latent_init.expand(n_latent, -1)
        hl = self.cell_to_latent(hc, hl, assign_index, assign_attr)
        if self.processor_kind == "interaction":
            e = self.edge_embed(latent_attr)
            for layer, film in zip(self.processor, self.films):
                upd, e = layer(hl, latent_edges, e)
                hl = hl + self.dropout(film(upd, params_latent))
        else:
            for layer, film in zip(self.processor, self.films):
                h_new = film(layer(hl, latent_edges, latent_attr), params_latent)
                hl = hl + self.dropout(h_new)
        if dec_index is not None:
            back_index, back_attr = dec_index, dec_attr
        else:
            back_index = torch.flip(assign_index, dims=(0,))
            back_attr = torch.cat([-assign_attr[:, :2], assign_attr[:, 2:]], dim=-1)
        hc_out = self.latent_to_cell(hl, hc, back_index, back_attr)
        dec_in = [hc, hc_out] + ([pc] if self.params_to_cells else [])
        z = torch.cat(dec_in, dim=-1)
        if not self.head_index:
            return self.decoder(z)
        outs = [dec(z) for dec in self.decoders]
        out = outs[0].new_empty(z.shape[0], self.out_features)
        for k, o in enumerate(outs):
            out[:, getattr(self, f"head_idx{k}")] = o
        return out
