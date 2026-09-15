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

from solstice.data.converters import to_solps as ts

NX, NY, NS = 3, 2, 2
NCELL = (NX + 2) * (NY + 2)


def _block(name, arr, dtype="real"):
    flat = np.asarray(arr, dtype=float).flatten(order="F")
    return (f"*cf:    {dtype:4s} {flat.size:16d}    {name}\n"
            + "".join(ts._render_block(flat)))


def _write_b2fstate(path, rng):
    """Synthetic structured b2fstate: 3x2 interior, D0 + D+ species."""
    ne = rng.uniform(1e18, 1e20, (NX + 2, NY + 2))
    na = np.stack([np.full((NX + 2, NY + 2), 1e12), ne], axis=-1)
    te = rng.uniform(1, 100, (NX + 2, NY + 2)) * ts.EV_TO_J
    ti = rng.uniform(1, 100, (NX + 2, NY + 2)) * ts.EV_TO_J
    ua = rng.normal(0, 1e4, (NX + 2, NY + 2, NS))
    fna = rng.normal(size=(NX + 2, NY + 2, 2, NS))
    txt = (
        "VERSION03.000.008 test\n"
        f"*cf:    int                3    nx,ny,ns\n {NX:10d} {NY:10d} {NS:10d}\n"
        "*cf:    char             120    label\n test label\n"
        + _block("zamin", [0.0, 1.0]) + _block("zamax", [0.0, 1.0])
        + _block("zn", [1.0, 1.0]) + _block("am", [2.0, 2.0])
        + _block("na", na) + _block("ne", ne) + _block("ua", ua)
        + _block("te", te) + _block("ti", ti) + _block("po", np.zeros_like(ne))
        + _block("fna", fna) + _block("time", [1.5])
    )
    path.write_text(txt)
    return {"ne": ne, "na": na, "te": te, "ti": ti, "ua": ua, "fna": fna}


@pytest.fixture()
def ref_run(tmp_path):
    run = tmp_path / "ref"
    run.mkdir()
    truth = _write_b2fstate(run / "b2fstate", np.random.default_rng(0))
    (run / "b2fstati").write_text((run / "b2fstate").read_text())
    (run / "b2mn.prt").write_text("log")
    (run / "run.log").write_text("log")
    (run / "output.0001").write_text("x")
    (run / "b2mn.exe.dir").mkdir()
    (run / "run.csh").write_text("#!/bin/tcsh\ncd /home/x/runs/ref\nb2run b2mn > run.log &\n")
    return run, truth


def test_b2f_roundtrip(ref_run):
    run, truth = ref_run
    assert ts.b2f_read_dims(run / "b2fstate") == (NX, NY, NS)
    assert np.allclose(ts.b2f_extract("ne", run / "b2fstate"), truth["ne"])
    assert np.allclose(ts.b2f_extract("na", run / "b2fstate"), truth["na"])
    assert ts.b2f_extract("fna", run / "b2fstate", reshape=False).size == 2 * NS * NCELL
    assert set(ts.list_real_blocks(run / "b2fstate")) == {
        "zamin", "zamax", "zn", "am", "na", "ne", "ua", "te", "ti", "po", "fna", "time"}


def test_check_flatness_detects_structure(ref_run):
    run, _ = ref_run
    assert ts.check_flatness(run, verbose=False) is False


def test_splice_only_touches_named_blocks(ref_run, tmp_path):
    run, truth = ref_run
    out = tmp_path / "spliced"
    new_te = np.full((NX + 2, NY + 2), 7.0 * ts.EV_TO_J)
    ts.write_spliced_b2fstate(run / "b2fstate", out, {"te": new_te})
    assert np.allclose(ts.b2f_extract("te", out), new_te)
    ts.verify_nn_state(run / "b2fstate", out, {"te"})
    with pytest.raises(AssertionError):
        ts.verify_nn_state(run / "b2fstate", out, {"ne"})


def test_flat_state(ref_run, tmp_path):
    run, _ = ref_run
    ref_state = run / "b2fstate"
    n0 = ts.neutral_density_default(ref_state, 5e19)
    assert n0 == 1e12  # reuses the reference's uniform fluid-neutral floor
    fields = ts.build_flat_state(ref_state, 5e19, 40.0, 30.0, n0)
    out = tmp_path / "flat"
    ts.write_spliced_b2fstate(ref_state, out, fields)
    ts.verify_flat_state(ref_state, out)
    assert ts.check_flatness(tmp_path, "flat", verbose=False)
    na = ts.b2f_extract("na", out)
    assert np.allclose(na[:, :, 1], 5e19) and np.allclose(na[:, :, 0], 1e12)
    assert np.allclose(ts.b2f_extract("ti", out) / ts.EV_TO_J, 30.0)
    assert np.all(ts.b2f_extract("fna", out, reshape=False) == 0)
    assert ts.b2f_extract("time", out, reshape=False)[0] == 0


def test_stage_run_dir(ref_run, tmp_path):
    run, _ = ref_run
    out = tmp_path / "ref_flat_init"
    ts.stage_run_dir(run, out)
    for gone in ("b2fstate", "b2fstati", "b2mn.prt", "run.log", "output.0001", "b2mn.exe.dir"):
        assert not (out / gone).exists()
    assert "cd /home/x/runs/ref_flat_init\n" in (out / "run.csh").read_text()
    with pytest.raises(FileExistsError):
        ts.stage_run_dir(run, out)


def test_cli_flat(ref_run, tmp_path):
    run, _ = ref_run
    out = tmp_path / "cli_out"
    ts.main(["--init", "flat", "--reference-run", str(run), "--output-dir", str(out),
             "--ne", "1e19", "--te", "20", "--ti", "20"])
    assert ts.check_flatness(out, verbose=False)
    assert np.allclose(ts.b2f_extract("ne", out / "b2fstati"), 1e19)


def test_namelist_parsing():
    text = "  enepar(1,1) = 2*3.0e6,\n  conpar(0,1,1) = 1.0, 7.5e20,\n"
    assert ts._find_namelist_value(text, "enepar(1,1)") == [3.0e6, 3.0e6]
    assert ts._find_namelist_value(text, "conpar(0,1,1)")[1] == 7.5e20
    assert ts._find_namelist_value(text, "missing") is None
