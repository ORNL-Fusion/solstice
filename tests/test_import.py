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
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import xarray as xr

import solstice
from solstice.graphs import cell_adjacency_edges
from solstice.models import register_model, build_model


def test_registry_roundtrip():
    @register_model("dummy")
    class Dummy:
        def __init__(self, k=1):
            self.k = k

    assert build_model("dummy", {"k": 3}).k == 3
    with pytest.raises(KeyError):
        build_model("nope", {})


def test_cell_adjacency_edges():
    # two cells sharing one face; one boundary face
    case = xr.Dataset(
        {
            "face_cells": (("face", "two"), np.array([[0, 1], [1, -1]])),
            "cell_r": ("cell", np.array([0.0, 1.0])),
            "cell_z": ("cell", np.array([0.0, 0.0])),
        }
    )
    edge_index, edge_attr = cell_adjacency_edges(case)
    assert edge_index.shape == (2, 2)  # undirected -> both directions
    assert np.allclose(edge_attr[:, 2], 1.0)


def _tiny_geo(nx=3, ny=2):
    # regular grid, cell (ix, iy) spans [ix-.5, ix+.5] x [iy-.5, iy+.5]
    crx = np.zeros((nx + 2, ny + 2, 4))
    cry = np.zeros((nx + 2, ny + 2, 4))
    for ix in range(nx + 2):
        for iy in range(ny + 2):
            crx[ix, iy] = [ix - 0.5, ix + 0.5, ix - 0.5, ix + 0.5]
            cry[ix, iy] = [iy - 0.5, iy - 0.5, iy + 0.5, iy + 0.5]
    ixg, iyg = np.meshgrid(np.arange(nx + 2), np.arange(ny + 2), indexing="ij")
    region = np.zeros((nx + 2, ny + 2, 3), dtype=int)
    region[:, :, 0] = 2  # all SOL
    return {
        "nx": nx, "ny": ny, "crx": crx, "cry": cry,
        "vol": np.ones((nx + 2, ny + 2)),
        "bb": np.ones((nx + 2, ny + 2, 4)),
        "leftix_py": ixg - 1, "leftiy_py": iyg,
        "bottomix_py": ixg, "bottomiy_py": iyg - 1,
        "region": region,
        "region_ids": {"vol": {"is_core": 1, "sol": 2}},
    }


def test_structured_mesh_builder():
    from solstice.data.converters.from_solps import FACE_SETS, build_structured_mesh

    mesh = build_structured_mesh(_tiny_geo())
    assert mesh.sizes["cell"] == 6
    assert mesh.sizes["vertex"] == 12
    assert mesh.sizes["face"] == 17  # (nx+1)*ny + nx*(ny+1)
    fc = mesh["face_cells"].values
    assert (fc[:, 1] >= 0).sum() == 7 and (fc[:, 1] < 0).sum() == 10
    fs = mesh["face_set"].values
    assert (fs == FACE_SETS["inner_target"]).sum() == 2   # west, smaller R
    assert (fs == FACE_SETS["outer_target"]).sum() == 2
    assert (fs == FACE_SETS["pfr_boundary"]).sum() == 3   # south, non-core
    assert (fs == FACE_SETS["wall"]).sum() == 3
    # every interior face joins cells whose centres are one unit apart
    r, z = mesh["cell_r"].values, mesh["cell_z"].values
    a, b = fc[fc[:, 1] >= 0, 0], fc[fc[:, 1] >= 0, 1]
    d = np.hypot(r[a] - r[b], z[a] - z[b])
    assert np.allclose(d, 1.0)


def test_gnn_forward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from solstice.graphs import build_latent_graph
    from solstice.models import build_model

    n, k = 40, 8
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.normal(size=(n, 2)), dtype=torch.float32)

    r, z = rng.normal(size=n), rng.normal(size=n)
    lg = build_latent_graph(r, z, n_latent=k, k_nn=3, seed=0)
    m2 = build_model("gnn", {"node_features": 2, "param_dim": 5, "out_features": 3,
                        "hidden": 16, "n_process_layers": 2})
    out = m2(x, torch.tensor(lg["assign_index"]),
             torch.tensor(lg["assign_attr"], dtype=torch.float32),
             torch.tensor(lg["latent_edges"]),
             torch.tensor(lg["latent_attr"], dtype=torch.float32),
             torch.zeros(k, 5), k)
    assert out.shape == (n, 3) and torch.isfinite(out).all()
