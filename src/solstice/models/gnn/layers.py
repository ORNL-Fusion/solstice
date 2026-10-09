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
"""Shared GNN building blocks (ported from the solpex-paper conditional GNN)."""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeConv(MessagePassing):
    """Message passing with edge attributes (dR, dZ, distance)."""

    def __init__(self, in_ch, out_ch, edge_dim=3):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_ch + edge_dim, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_ch + out_ch, out_ch),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(out)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class BipartiteConv(MessagePassing):
    """Message passing from a source node set to a target node set
    (cells -> latent mesh in the encoder, latent -> cells in the decoder)."""

    def __init__(self, src_ch, dst_ch, out_ch, edge_dim=3):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(src_ch + dst_ch + edge_dim, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(dst_ch + out_ch, out_ch),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x_src, x_dst, edge_index, edge_attr):
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr,
                             size=(x_src.shape[0], x_dst.shape[0]))
        out = self.update_mlp(torch.cat([x_dst, out], dim=-1))
        return self.norm(out)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class NodeFiLM(nn.Module):
    """FiLM conditioning: global params -> per-node scale + shift.
    Zero-initialized so conditioning starts as identity."""

    def __init__(self, param_dim, hidden_dim, film_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, hidden_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.hidden_dim = hidden_dim

    def forward(self, h, params):
        gamma, beta = self.net(params).split(self.hidden_dim, dim=-1)
        return h * (1.0 + gamma) + beta


def _mlp_ln(n_in, hidden):
    """GraphCast MLP: Linear-SiLU-Linear followed by LayerNorm."""
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))


class InteractionLayer(nn.Module):
    """GraphCast/WeatherNext-Graph processor block (Battaglia et al. interaction
    network): edges carry their own latent state, updated with a residual from
    (edge, sender, receiver); nodes aggregate the new edge states (sum by
    default) and return an update. The edge residual is applied here; the node
    residual is left to the caller so FiLM/dropout can act on the update."""

    def __init__(self, hidden, aggr="sum"):
        super().__init__()
        self.edge_mlp = _mlp_ln(3 * hidden, hidden)
        self.node_mlp = _mlp_ln(2 * hidden, hidden)
        self.aggr = aggr

    def forward(self, x, edge_index, e):
        src, dst = edge_index[0], edge_index[1]
        e = e + self.edge_mlp(torch.cat([e, x[src], x[dst]], dim=-1))
        agg = torch.zeros(x.shape[0], e.shape[-1], dtype=e.dtype, device=e.device).index_add_(0, dst, e)
        if self.aggr == "mean":
            deg = torch.zeros(x.shape[0], dtype=e.dtype, device=e.device).index_add_(
                0, dst, torch.ones_like(dst, dtype=e.dtype)).clamp_min(1)
            agg = agg / deg[:, None]
        return self.node_mlp(torch.cat([x, agg.to(x.dtype)], dim=-1)), e
