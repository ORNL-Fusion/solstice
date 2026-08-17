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
"""Mesh -> graph construction. Edges are cell pairs sharing a face,
which works identically on structured and GOAT wide grids
(docs/specs/data_schema.md)."""

from __future__ import annotations

import numpy as np
import xarray as xr


def default_node_features(case: xr.Dataset) -> np.ndarray:
    """Per-cell geometry features [psi_n_approx, |B|] (solpex-paper convention).

    psi_n_approx is the normalised radius from the cell-cloud centroid —
    the same approximation the solpex-paper GNN used; replace with true
    psi_n when the mesh carries it."""
    r = np.asarray(case["cell_r"].values)
    z = np.asarray(case["cell_z"].values)
    rho = np.hypot(r - r.mean(), z - z.mean())
    psi_n = rho / rho.max() if rho.max() > 0 else np.zeros_like(rho)
    bmag = np.asarray(case["cell_b"].values)[:, 3]
    return np.stack([psi_n, bmag], axis=1).astype(np.float32)


def cell_adjacency_edges(case: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edge list (2, E) from face adjacency, plus per-edge
    geometry features (dR, dZ, dist) from cell centres."""
    face_cells = np.asarray(case["face_cells"].values, dtype=np.int64)
    interior = (face_cells >= 0).all(axis=1)
    a, b = face_cells[interior, 0], face_cells[interior, 1]
    edge_index = np.concatenate(
        [np.stack([a, b]), np.stack([b, a])], axis=1
    )
    r = np.asarray(case["cell_r"].values)
    z = np.asarray(case["cell_z"].values)
    dr = r[edge_index[1]] - r[edge_index[0]]
    dz = z[edge_index[1]] - z[edge_index[0]]
    edge_attr = np.stack([dr, dz, np.hypot(dr, dz)], axis=1)
    return edge_index, edge_attr
