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
"""Feature engineering and normalization of solstice.training.data on a toy store."""
import json

import numpy as np
import pytest
import xarray as xr

from solstice.training import data as td


def _toy_store(n=20, n_cells=6):
    rng = np.random.default_rng(1)
    ds = xr.Dataset()
    ds["input_pe"] = ("case", rng.uniform(1e6, 2e6, n))
    ds["input_pi"] = ("case", ds["input_pe"].values.copy())
    ds["input_hci"] = ("case", rng.uniform(0.1, 2, n))
    ds["input_hce"] = ("case", ds["input_hci"].values.copy())
    ds["input_puff_D"] = ("case", 10 ** rng.uniform(18, 24, n))
    ds["input_const"] = ("case", np.full(n, 7.0))
    ds["te"] = (("case", "cell"), 10 ** rng.uniform(-1, 3, (n, n_cells)))
    ds["ua_D1"] = (("case", "cell"), rng.normal(0, 1e4, (n, n_cells)))
    ds["qc_pass"] = ("case", np.r_[np.ones(n - 2, bool), False, False])
    ds["regime"] = ("case", np.array(["attached"] * (n - 4) + ["cold core"] * 4))
    return ds


def test_engineer_inputs_merges_and_drops():
    raw = td.engineer_inputs(_toy_store())
    assert set(raw) == {"ptot", "chi", "puff_D"}
    assert np.allclose(raw["ptot"], 2 * _toy_store()["input_pe"].values)


def test_input_norm_log_and_standardize():
    ds = _toy_store()
    raw = td.engineer_inputs(ds)
    idx = np.arange(ds.sizes["case"])
    norm = td.InputNorm(names=["chi", "ptot", "puff_D"], log_inputs=["puff_D"]).fit(raw, idx)
    Xn = norm.transform(raw)
    assert Xn.shape == (20, 3) and Xn.dtype == np.float32
    np.testing.assert_allclose(Xn.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(Xn.std(0), 1, atol=1e-4)
    assert 18 <= norm.lo[2] and norm.hi[2] <= 24          # log10 units for puff_D
    with pytest.raises(KeyError):
        norm.transform({"chi": raw["chi"]})


def test_output_norm_round_trip():
    ds = _toy_store()
    idx = np.arange(ds.sizes["case"])
    norm = td.OutputNorm({"te": True, "ua_D1": False}).fit(ds, idx)
    Yn = norm.transform(ds)
    assert Yn.shape == (20, 6, 2)
    np.testing.assert_allclose(norm.inverse(Yn[:, :, 0], 0), ds["te"].values, rtol=1e-5)
    np.testing.assert_allclose(norm.inverse(Yn[:, :, 1], 1), ds["ua_D1"].values, rtol=1e-5)


def test_select_and_split():
    ds = _toy_store()
    idx = td.select_cases(ds, qc="qc_pass", exclude_regimes=["cold core"])
    assert len(idx) == 16 and idx.max() < 16
    tr, va = td.train_val_split(idx, 0.25, seed=0)
    assert len(va) == 4 and len(tr) == 12
    assert not set(tr) & set(va)


def _toy_mesh_store(n=8, n_cells=6):
    """Toy store with face sets: cells 4, 5 at the outer target, cells 0, 2 at the inner."""
    ds = _toy_store(n, n_cells)
    ds["cell_iy"] = ("cell", np.array([1, 1, 2, 2, 1, 2]))
    ds["face_cells"] = (("face", "two"), np.array([[4, -1], [5, -1], [0, -1], [2, -1], [1, 3]]))
    ds["face_set"] = ("face", np.array([2, 2, 1, 1, 0], np.int8))
    ds.attrs["face_sets"] = json.dumps({"interior": 0, "inner_target": 1, "outer_target": 2})
    ds["q_outer_target"] = (("case", "target_iy"), np.tile([10.0, 20.0], (n, 1)))
    ds["q_inner_target"] = (("case", "target_iy"), np.full((n, 2), -3.0))
    return ds


def test_output_norm_aux_head_mask_and_placement():
    ds = _toy_mesh_store()
    aux = {"q_target": {"inner_target": "q_inner_target", "outer_target": "q_outer_target"}}
    norm = td.OutputNorm({"te": True}, aux).fit(ds, np.arange(8))
    assert norm.names == ["te", "q_target"] and norm.fields["q_target"] is False
    np.testing.assert_array_equal(norm.mask[:, 0], 1)
    np.testing.assert_array_equal(norm.mask[:, 1], [1, 0, 1, 0, 1, 1])
    Yn = norm.transform(ds)
    y = norm.inverse(Yn[:, :, 1], 1)
    np.testing.assert_allclose(y[:, 4], 10.0, rtol=1e-5)     # ordered along iy: iy=1 -> 10
    np.testing.assert_allclose(y[:, 5], 20.0, rtol=1e-5)
    np.testing.assert_allclose(y[:, [0, 2]], -3.0, rtol=1e-5)
    with pytest.raises(ValueError, match="profile points"):
        td.OutputNorm({"te": True}, aux).fit(ds.isel(target_iy=slice(0, 1)), np.arange(8))
    back = td.OutputNorm.from_dict(norm.to_dict())
    assert back.names == norm.names and np.array_equal(back.mask, norm.mask)


def test_output_norm_quantile_round_trip_and_gaussian():
    ds = _toy_store()
    idx = np.arange(ds.sizes["case"])
    norm = td.OutputNorm({"te": True, "ua_D1": False}, transform_kind="quantile",
                         n_quantiles=10).fit(ds, idx)
    Yn = norm.transform(ds)
    assert Yn.shape == (20, 6, 2) and norm.quantiles.shape == (6, 2, 10)
    # training values map back to themselves (knots interpolate the training set)
    np.testing.assert_allclose(norm.inverse(Yn[:, :, 0], 0), ds["te"].values, rtol=1e-4)
    np.testing.assert_allclose(norm.inverse(Yn[:, :, 1], 1), ds["ua_D1"].values, rtol=1e-4)
    # per-cell marginals are (discretely) standard normal: symmetric, bounded, centred
    assert np.all(np.abs(Yn) < 5.3) and abs(float(Yn.mean())) < 0.2
    # ranks are preserved per cell
    for c in range(6):
        assert np.all(np.diff(Yn[np.argsort(ds["te"].values[:, c]), c, 0]) >= 0)
    # export round trip
    back = td.OutputNorm.from_dict(norm.to_dict())
    assert back.transform_kind == "quantile" and back.n_quantiles == 10
    np.testing.assert_allclose(back.transform(ds), Yn)


def test_output_norm_quantile_constant_cells_map_to_zero():
    ds = _toy_store()
    ds["flat"] = (("case", "cell"), np.full((20, 6), 3.0))
    norm = td.OutputNorm({"flat": False}, transform_kind="quantile", n_quantiles=8).fit(ds, np.arange(20))
    Yn = norm.transform(ds)
    np.testing.assert_allclose(Yn, 0.0, atol=1e-12)
    np.testing.assert_allclose(norm.inverse(Yn[:, :, 0], 0), 3.0)
