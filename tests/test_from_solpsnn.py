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
"""Array-level conventions of the SOLPS-NN release converter (no release data needed)."""
import numpy as np

from solstice.data.converters import from_solpsnn as fs


def test_canonical_inputs_split_power_and_chi():
    X = np.array([[3.0, 2.5, 1e7, 1e20, 1e19, 1e21, 0.5, 1.5]])
    inp = fs.canonical_inputs(X)
    assert inp["pe"][0] == inp["pi"][0] == 5e6
    assert inp["hci"][0] == inp["hce"][0] == 1.5
    assert inp["rmajor"][0] == 3.0 and inp["btor"][0] == 2.5
    assert inp["puff_D"][0] == 1e20 and inp["puff_N"][0] == 1e19
    assert set(inp) == set(fs.INPUT_UNITS)


def test_regime_precedence():
    te_omp = np.array([5.0, 100.0, 100.0, 100.0])
    te_ot = np.array([1.0, 90.0, 20.0, 1.0])
    assert list(fs.classify_regime(te_omp, te_ot)) == \
        ["cold core", "sheath-limited", "attached", "detached"]


def test_interior_drops_guard_cells_iy_fastest():
    a = np.arange(2 * 4 * 3).reshape(2, 4, 3)          # (n, nx+2, ny+2) with nx=2, ny=1
    cells = fs.interior(a)
    assert cells.shape == (2, 2)
    np.testing.assert_array_equal(cells[0], a[0, 1:-1, 1:-1].reshape(-1))


def test_poloidal_heat_flux_targets_and_scaling():
    n, nx, ny = 2, 3, 2
    fht = np.zeros((n, nx + 2, ny + 2, 2))
    fht[:, :, :, 0] = 2.0                               # uniform west-face flux [W]
    fht[:, :, :, 1] = 99.0                              # radial component must be ignored
    gs_x = np.full((nx + 2, ny + 2), 0.5)               # unscaled x-face areas
    scale = np.array([1.0, 2.0])
    q_cc, q_in, q_out = fs.poloidal_heat_flux(fht, gs_x, scale)
    assert q_cc.shape == (n, nx * ny) and q_in.shape == (n, ny) and q_out.shape == (n, ny)
    np.testing.assert_allclose(q_out[0], 4.0)           # 2 W / 0.5 m^2
    np.testing.assert_allclose(q_out[1], 1.0)           # areas scale with R^2
    np.testing.assert_allclose(q_in, q_out)
    np.testing.assert_allclose(q_cc[0], 4.0)            # mean of west and east faces
