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
"""Latent (hidden) mesh for encode-process-decode GNNs.

GraphCast/anemoi pattern adapted to the tokamak edge: latent nodes are
k-means centroids of the cell centres in (R, Z), so only the thin
encoder/decoder ever see the machine mesh, and always through relative
geometry (dR, dZ, dist). Deterministic (seeded) so a checkpoint's
latent mesh is reproducible from its config.
"""

from __future__ import annotations

import numpy as np


def _kmeans(points: np.ndarray, k: int, seed: int = 0, n_iter: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), size=k, replace=False)]
    for _ in range(n_iter):
        d = ((points[:, None, :] - centroids[None]) ** 2).sum(-1)
        assign = d.argmin(1)
        for j in range(k):
            sel = assign == j
            if sel.any():
                centroids[j] = points[sel].mean(0)
    return centroids


def build_latent_graph(cell_r, cell_z, n_latent=256, k_nn=6, seed=0):
    """Latent mesh over the cell cloud.

    Returns dict with:
      latent_pos      (n_latent, 2)
      assign_index    (2, n_cells)  cell -> nearest latent node (encoder edges)
      assign_attr     (n_cells, 3)  dR, dZ, dist per encoder edge
      latent_edges    (2, E)        k-NN edges between latent nodes (undirected)
      latent_attr     (E, 3)
    Decoder edges are assign_index reversed with negated offsets.
    """
    pts = np.stack([np.asarray(cell_r), np.asarray(cell_z)], axis=1)
    latent = _kmeans(pts, n_latent, seed=seed)

    d2 = ((pts[:, None, :] - latent[None]) ** 2).sum(-1)
    nearest = d2.argmin(1)
    assign_index = np.stack([np.arange(len(pts)), nearest])
    dv = latent[nearest] - pts
    assign_attr = np.column_stack([dv, np.hypot(dv[:, 0], dv[:, 1])])

    k_nn = min(k_nn, n_latent - 1)
    ld2 = ((latent[:, None, :] - latent[None]) ** 2).sum(-1)
    np.fill_diagonal(ld2, np.inf)
    nbr = np.argsort(ld2, axis=1)[:, :k_nn]
    src = np.repeat(np.arange(n_latent), k_nn)
    dst = nbr.reshape(-1)
    pairs = np.unique(np.sort(np.stack([src, dst]), axis=0), axis=1)
    latent_edges = np.concatenate([pairs, pairs[::-1]], axis=1)
    ldv = latent[latent_edges[1]] - latent[latent_edges[0]]
    latent_attr = np.column_stack([ldv, np.hypot(ldv[:, 0], ldv[:, 1])])

    return {
        "latent_pos": latent,
        "assign_index": assign_index,
        "assign_attr": assign_attr,
        "latent_edges": latent_edges,
        "latent_attr": latent_attr,
    }
