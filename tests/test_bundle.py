# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from tests.test_import import _tiny_geo


@pytest.fixture()
def tiny_mesh(tmp_path):
    from solstice.data.converters.from_solps import build_structured_mesh
    mesh = build_structured_mesh(_tiny_geo())
    p = tmp_path / "mesh.nc"
    mesh.to_netcdf(p)
    return p, mesh.sizes["cell"]


def test_mlp_bundle_roundtrip(tmp_path, tiny_mesh):
    torch = pytest.importorskip("torch")
    from solstice.hub import create_state_bundle, load_state_bundle
    from solstice.models import build_model

    mesh_path, n_cells = tiny_mesh
    core = build_model("mlp_v1", {"in_dim": 5, "hidden": 8, "out_dim": n_cells})
    pt = {
        "state_dict": core.state_dict(),
        "cell_mean": np.zeros(n_cells), "cell_std": np.ones(n_cells),
        "log10": True,
        "inputs": ["chi", "core_fueling", "dna", "ptot", "puff_D2"],
        # realistic scalers so standardized features are O(1)
        "x_mean": np.array([0.7, 20.0, 0.5, 2e6, 21.0]),
        "x_std": np.array([0.1, 0.3, 0.3, 1e6, 0.5]),
    }
    ptp = tmp_path / "diiid-test-state-mlp-te.pt"
    torch.save(pt, ptp)

    out = create_state_bundle(ptp, tmp_path / "bundles", "diiid-test-state-mlp-te-v1",
                              mesh_path, provenance={"dataset": "test"})
    pred = load_state_bundle(out)
    fields = pred.predict({"pe": 1e6, "pi": 1e6, "core_fueling": 1e20,
                           "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.7})
    assert set(fields) == {"te"}
    assert fields["te"].shape == (n_cells,)
    assert np.isfinite(fields["te"]).all() and (fields["te"] >= 0).all()


def test_gnn_bundle_roundtrip(tmp_path, tiny_mesh):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from solstice.hub import create_state_bundle, load_state_bundle
    from solstice.models import build_model

    mesh_path, n_cells = tiny_mesh
    cfg = {"node_features": 2, "param_dim": 5, "out_features": 2,
           "hidden": 8, "n_process_layers": 2}
    core = build_model("gnn", cfg)
    pt = {
        "state_dict": core.state_dict(), "model_class": "gnn", "config": cfg,
        "fields": {"te": True, "ua_D1": False},
        "inputs": ["chi", "core_fueling", "dna", "ptot", "puff_D2"],
        "x_mean": np.zeros(5), "x_std": np.ones(5),
        "y_mean": np.zeros((n_cells, 2)), "y_std": np.ones((n_cells, 2)),
        "n_latent": 3,
    }
    ptp = tmp_path / "diiid-test-state-gnn.pt"
    torch.save(pt, ptp)

    out = create_state_bundle(ptp, tmp_path / "bundles", "diiid-test-state-gnn-v1",
                              mesh_path)
    pred = load_state_bundle(out)
    fields = pred.predict({"pe": 1e6, "pi": 1e6, "core_fueling": 1e20,
                           "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.7})
    assert set(fields) == {"te", "ua_D1"}
    assert all(np.isfinite(v).all() for v in fields.values())


def test_source_bundle_roundtrip(tmp_path, tiny_mesh):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from solstice.hub import create_source_bundle, load_source_bundle
    from solstice.models import build_model

    mesh_path, n_cells = tiny_mesh
    plasma = {"te": True, "ne": True, "ua_D1": False}
    cfg = {"node_features": 2 + len(plasma), "param_dim": 5, "out_features": 5,
           "hidden": 8, "n_process_layers": 2}
    core = build_model("gnn", cfg)
    pt = {
        "state_dict": core.state_dict(), "model_class": "gnn", "config": cfg,
        "task": "sources", "plasma_features": plasma,
        "sources": ["sp", "sne", "qe", "qi", "sm"],
        "use_params": True,
        "inputs": ["chi", "core_fueling", "dna", "ptot", "puff_D2"],
        "x_mean": np.array([0.7, 20.0, 0.5, 2e6, 21.0]),
        "x_std": np.array([0.1, 0.3, 0.3, 1e6, 0.5]),
        "geom_mean": np.zeros(2), "geom_std": np.ones(2),
        "pf_mean": {"te": 1.5, "ne": 19.5, "ua_D1": 0.0},
        "pf_std": {"te": 0.5, "ne": 0.5, "ua_D1": 5e3},
        "y_mean": np.zeros((n_cells, 5)), "y_std": np.ones((n_cells, 5)),
        "n_latent": 3,
    }
    ptp = tmp_path / "diiid-test-sources-gnn.pt"
    torch.save(pt, ptp)

    out = create_source_bundle(ptp, tmp_path / "bundles", "diiid-test-sources-gnn-v1",
                               mesh_path)
    pred = load_source_bundle(out)
    rng = np.random.default_rng(0)
    state = {"te": 10 ** rng.normal(1.5, 0.5, n_cells),
             "ne": 10 ** rng.normal(19.5, 0.5, n_cells),
             "ua_D1": rng.normal(0, 5e3, n_cells)}
    params = {"pe": 1e6, "pi": 1e6, "core_fueling": 1e20, "puff_D2": 1e21,
              "dna": 0.5, "hci": 0.7, "hce": 0.7}
    sources = pred.predict(state, params)
    assert set(sources) == {"sp", "sne", "qe", "qi", "sm"}
    assert all(v.shape == (n_cells,) and np.isfinite(v).all()
               for v in sources.values())
    with pytest.raises(ValueError):
        pred.predict(state)  # params required when use_params


def test_out_of_range_warning(tmp_path, tiny_mesh):
    torch = pytest.importorskip("torch")
    from solstice.hub import create_state_bundle, load_state_bundle
    from solstice.models import build_model

    mesh_path, n_cells = tiny_mesh
    core = build_model("mlp_v1", {"in_dim": 5, "hidden": 8, "out_dim": n_cells})
    pt = {
        "state_dict": core.state_dict(),
        "cell_mean": np.zeros(n_cells), "cell_std": np.ones(n_cells),
        "log10": False,
        "inputs": ["chi", "core_fueling", "dna", "ptot", "puff_D2"],
        "x_mean": np.array([0.7, 20.0, 0.5, 2e6, 21.0]),
        "x_std": np.array([0.1, 0.3, 0.3, 1e6, 0.5]),
        "x_min": np.array([0.1, 19.9, 0.1, 2e6, 20.1]),
        "x_max": np.array([2.0, 20.9, 2.0, 16e6, 21.7]),
    }
    ptp = tmp_path / "pepc-test-state-mlp-te.pt"
    torch.save(pt, ptp)
    pred = load_state_bundle(create_state_bundle(
        ptp, tmp_path / "b", "pepc-test-state-v1", mesh_path))
    ok = {"pe": 4e6, "pi": 4e6, "core_fueling": 3e20, "puff_D2": 1e21,
          "dna": 0.5, "hci": 0.7, "hce": 0.7}
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        pred.predict(ok)                          # inside the box: no warning
    with pytest.warns(UserWarning, match="ptot.*outside the training range"):
        pred.predict({**ok, "pe": 20e6, "pi": 20e6})   # ptot = 40 MW: outside


def test_asymmetric_pair_warnings(tmp_path, tiny_mesh):
    torch = pytest.importorskip("torch")
    from solstice.hub import create_state_bundle, load_state_bundle
    from solstice.models import build_model

    mesh_path, n_cells = tiny_mesh
    core = build_model("mlp_v1", {"in_dim": 5, "hidden": 8, "out_dim": n_cells})
    pt = {"state_dict": core.state_dict(),
          "cell_mean": np.zeros(n_cells), "cell_std": np.ones(n_cells),
          "log10": False,
          "inputs": ["chi", "core_fueling", "dna", "ptot", "puff_D2"],
          "x_mean": np.array([0.7, 20.0, 0.5, 2e6, 21.0]),
          "x_std": np.array([0.1, 0.3, 0.3, 1e6, 0.5])}
    ptp = tmp_path / "t.pt"
    torch.save(pt, ptp)
    pred = load_state_bundle(create_state_bundle(ptp, tmp_path / "b", "t-v1", mesh_path))

    canonical = {"ptot": 6e6, "chi": 0.7, "core_fueling": 3e20,
                 "puff_D2": 1e21, "dna": 0.5}
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        pred.predict(canonical)                                    # honest form: silent
        pred.predict({"pe": 3e6, "pi": 3e6, "core_fueling": 3e20,  # equal pairs: silent
                      "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.7})
    with pytest.warns(UserWarning, match="only sees their sum"):
        pred.predict({"pe": 2e6, "pi": 4e6, "core_fueling": 3e20,
                      "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.7})
    with pytest.warns(UserWarning, match="hce is ignored"):
        pred.predict({"pe": 3e6, "pi": 3e6, "core_fueling": 3e20,
                      "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.9})
