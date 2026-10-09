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


# ---------------------------------------------------------------------------------------
# bundle format 0.2: seed ensembles of solstice-train checkpoints (per-member quantile
# normalization and latent graph, extended node features, grouped heads, aux target
# channels, calibrated uncertainty), and 0.1 bundles still loading
# ---------------------------------------------------------------------------------------

def _tiny_store(tmp_path, n_cases=12, seed=0):
    """A stacked store on the tiny structured mesh: random inputs, log/linear fields and
    one target profile per plate, enough to fit the training-side normalizations."""
    import xarray as xr
    from solstice.data.converters.from_solps import build_structured_mesh
    from solstice.training.data import target_cells

    mesh = build_structured_mesh(_tiny_geo())
    rng = np.random.default_rng(seed)
    nc = mesh.sizes["cell"]
    ds = mesh.copy()
    for name, lo, hi in [("ptot", 2e6, 8e6), ("chi", 0.2, 1.5), ("puff_D2", 1e20, 1e22)]:
        ds[f"input_{name}"] = ("case", rng.uniform(lo, hi, n_cases))
    ds["te"] = (("case", "cell"), 10 ** rng.normal(1.0, 0.5, (n_cases, nc)))
    ds["ua_D1"] = (("case", "cell"), rng.normal(0.0, 1e3, (n_cases, nc)))
    n_t = len(target_cells(mesh, "outer_target"))
    ds["q_inner_target"] = (("case", "target_iy"), rng.uniform(1e5, 1e6, (n_cases, n_t)))
    ds["q_outer_target"] = (("case", "target_iy"), rng.uniform(1e5, 1e6, (n_cases, n_t)))
    ds["qc_pass"] = ("case", np.ones(n_cases, bool))
    p = tmp_path / "store.nc"
    ds.to_netcdf(p)
    return p


def _train_like_checkpoint(store_path, seed, out_dir):
    """What solstice-train writes to best.pt for one seed, with random weights."""
    import torch
    import xarray as xr
    from solstice.models import build_model
    from solstice.training.data import InputNorm, MeshGraph, OutputNorm, engineer_inputs, select_cases

    ds = xr.open_dataset(store_path)
    idx = select_cases(ds)
    raw = engineer_inputs(ds)
    in_norm = InputNorm(names=sorted(raw), log_inputs=["puff_D2"]).fit(raw, idx)
    aux = {"q_target": {"inner_target": "q_inner_target", "outer_target": "q_outer_target", "log": False}}
    out_norm = OutputNorm({"te": True, "ua_D1": False}, aux, transform_kind="quantile", n_quantiles=5).fit(ds, idx)
    graph = MeshGraph(ds, n_latent=3, k_nn=2, seed=seed, node_features="extended")
    cfg = {"node_features": graph.node_features.shape[1], "param_dim": len(in_norm.names),
           "out_features": len(out_norm.names), "n_cells": graph.n_cells, "hidden": 8,
           "n_process_layers": 2, "cell_embed": 2, "params_to_cells": True, "decoder_hidden": 8,
           "head_index": [[0, 1], [2]], "head_hidden": 8}
    torch.manual_seed(seed)
    model = build_model("gnn", cfg)
    pt = {"state_dict": model.state_dict(), "model_class": "gnn", "config": cfg,
          **in_norm.to_dict(), **out_norm.to_dict(),
          "n_latent": 3, "k_nn": 2, "node_features": "extended", **graph.graph_extra,
          "train_store": str(store_path), "code_git": "test", "seed": seed, "epoch": 1,
          "val_loss": 0.5, "epochs": 1, "train_config": {"data": {"split_seed": 0}}}
    out_dir.mkdir(parents=True)
    torch.save(pt, out_dir / "best.pt")
    ds.close()
    return out_dir


def test_ensemble_bundle_matches_training_side(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import xarray as xr
    from solstice.hub.bundle import create_state_bundle, load_state_bundle, mesh_from_store
    from solstice.training.data import engineer_inputs, select_cases
    from solstice.uncertainty import StateEnsemble

    store = _tiny_store(tmp_path)
    runs = [_train_like_checkpoint(store, k, tmp_path / f"seed{k}") for k in range(2)]
    ds = xr.open_dataset(store)
    idx = select_cases(ds)
    raw = {k: v[idx] for k, v in engineer_inputs(ds).items()}

    # training-side reference: calibrated ensemble over the run directories
    ens = StateEnsemble(runs, ds, train_store=str(store))
    cal = ens.calibrate(ds, idx[:6], fussiness=2.0, use_fields=["te"], use_regions=["all", "outer_target"])
    ref = ens.predict(raw)

    mesh = mesh_from_store(store, tmp_path / "mesh.nc")
    b = create_state_bundle([r / "best.pt" for r in runs], tmp_path / "bundles", "pepc-test-state-v2", mesh,
                            calibration=cal, train_inputs=raw, provenance={"dataset": "tiny"})
    for f in ("bundle.json", "mesh.nc", "model_card.md", "member_0.safetensors", "member_1.npz",
              "uq_calibration.json", "novelty.npz"):
        assert (b / f).exists(), f
    man = __import__("json").loads((b / "bundle.json").read_text())
    assert man["bundle_version"] == "0.2" and man["n_members"] == 2
    assert man["output_transform"] == "quantile" and man["graph"]["node_features"] == "extended"
    assert man["members"][1]["seed"] == 1

    pred = load_state_bundle(b)
    got = pred.predict_batch(raw)
    for n in ("te", "ua_D1"):
        np.testing.assert_allclose(got["mean"][n], ref["mean"][n], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(got["std"][n], ref["std"][n], rtol=1e-5, atol=1e-8)
        for reg in ("all", "outer_target"):
            np.testing.assert_allclose(got["errbar"][n][reg], ref["errbar"][n][reg], rtol=1e-5)
            assert (got["ok"][n][reg] == ref["ok"][n][reg]).all()
    assert (got["use"] == ref["use"]).all() and (got["novel"] == ref["novel"]).all()
    np.testing.assert_allclose(got["confidence"], ref["confidence"], rtol=1e-5)
    # aux channel: defined on the target cells only, returned as per-plate profiles
    assert np.isnan(got["mean"]["q_target"]).any() and not np.isnan(got["mean"]["te"]).any()
    one = pred.predict({k: float(v[0]) for k, v in raw.items()})
    assert set(one) == {"te", "ua_D1", "q_target"}
    assert set(one["q_target"]) == {"inner_target", "outer_target"}
    assert one["q_target"]["outer_target"].shape == (len(pred.cells["outer_target"]),)
    assert np.isfinite(one["q_target"]["outer_target"]).all()
    np.testing.assert_allclose(one["te"], ref["mean"]["te"][0], rtol=1e-5)
    # a request far outside the training box is flagged, not silently predicted
    far = {k: (v[:1] * (50.0 if k == "ptot" else 1.0)) for k, v in raw.items()}
    with pytest.warns(UserWarning, match="outside the training range"):
        r = pred.predict_batch(far)
    assert not r["in_box"][0] and not r["use"][0] and r["confidence"][0] == 0.0
    ds.close()


def test_legacy_v01_bundle_still_loads(tmp_path, tiny_mesh):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import json
    from safetensors.torch import save_file
    from solstice.hub import load_state_bundle
    from solstice.models import build_model

    mesh_path, n_cells = tiny_mesh
    cfg = {"node_features": 2, "param_dim": 5, "out_features": 2, "hidden": 8, "n_process_layers": 2}
    core = build_model("gnn", cfg)
    b = tmp_path / "pepc-legacy-v1"; b.mkdir()
    save_file({k: v.contiguous() for k, v in core.state_dict().items()}, b / "weights.safetensors")
    np.savez(b / "normalization.npz", x_mean=np.zeros(5), x_std=np.ones(5),
             y_mean=np.zeros((n_cells, 2)), y_std=np.ones((n_cells, 2)))
    import shutil; shutil.copy(mesh_path, b / "mesh.nc")
    (b / "bundle.json").write_text(json.dumps({
        "bundle_version": "0.1", "name": "pepc-legacy-v1", "task": "state",
        "model": {"class": "gnn", "config": cfg},
        "variables": {"inputs": ["chi", "core_fueling", "dna", "ptot", "puff_D2"],
                      "outputs": ["te", "ua_D1"], "log10_outputs": {"te": True, "ua_D1": False}},
        "input_transform": {"log_inputs": ["core_fueling", "puff_D2"],
                            "merged": {"ptot": {"op": "sum", "of": ["pe", "pi"]}, "chi": {"op": "first", "of": ["hci", "hce"]}}},
        "provenance": {"parent": None}, "license": "CC-BY-4.0", "n_latent": 3, "k_nn": 2}))
    pred = load_state_bundle(b)
    out = pred.predict({"pe": 1e6, "pi": 1e6, "core_fueling": 1e20, "puff_D2": 1e21, "dna": 0.5, "hci": 0.7, "hce": 0.7})
    assert set(out) == {"te", "ua_D1"} and out["te"].shape == (n_cells,) and (out["te"] > 0).all()
    assert len(pred.members) == 1 and pred.calibration is None
