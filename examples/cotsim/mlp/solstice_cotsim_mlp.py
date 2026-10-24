# =========================================================================================
# (C) (or copyright) 2026. UT-Battelle, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =========================================================================================
"""Numpy-only reader of the SOLSTICE COTSIM MLP h5 file (needs numpy and h5py; no torch).
Mirrors matlab/read_NN_solstice.m: predict -> 295 physical outputs, result -> the content of
examples/cotsim/cotsim_boundary_demo.json without the 2D fields.

    from solstice_cotsim_mlp import CotsimMLP
    nn = CotsimMLP("solstice_cotsim_mlp.h5")
    y = nn.predict({"chi": 0.3, "core_fueling": 1e20, ...})     # (1, 295) or (N, 295)
    r = nn.result(y[0], rho_c=0.95)

    python solstice_cotsim_mlp.py solstice_cotsim_mlp.h5 [--reference reference_demo.json]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

DEMO_PARAMS = {"ptot": 1.299e7, "puff_D2": 1.623e21, "core_fueling": 1.0e20, "dna": 0.3, "chi": 0.3,
               "gamma_b_in": 9.312e20, "gamma_b_out": 2.691e21, "p_b_in": 68.47, "p_b_out": 1.304e7}


class CotsimMLP:
    def __init__(self, path: str):
        import h5py
        with h5py.File(path, "r") as f:
            layers = sorted(f["model_weights"])
            self.W = [f[f"model_weights/{n}/{n}/vars/kernel"][()].T.astype(np.float64) for n in layers]
            self.b = [f[f"model_weights/{n}/{n}/vars/bias"][()].astype(np.float64) for n in layers]
            nz = f["normalization"]
            self.x_mean, self.x_std = nz["x_mean"][()], nz["x_std"][()]
            self.x_log, self.y_log = nz["x_log"][()] == 1, nz["y_log"][()] == 1
            self.y_mean, self.y_std = nz["y_mean"][()], nz["y_std"][()]
            dec = lambda a: [s.decode() if isinstance(s, bytes) else s for s in a]
            self.input_names = dec(nz["input_names"][()])
            self.output_names = dec(nz["output_names"][()])
            self.output_units = dec(nz["output_units"][()])
            self.mesh = {k: f["solstice_mesh"][k][()] for k in f["solstice_mesh"]}
        self.mesh["closed"] = np.isin(self.mesh["rows"], self.mesh["closed_rows"])
        on = np.array(self.output_names)
        self.idx = {}
        for k in ("te", "ti", "ne", "ni"):
            for kind in ("fsa", "omp"):
                self.idx[f"{k}_{kind}"] = np.flatnonzero(np.char.startswith(on, f"{k}_{kind}_"))
        for side in ("outer", "inner"):
            self.idx[f"q_{side}"] = np.flatnonzero(np.char.startswith(on, f"q_{side}_"))
            self.idx[f"iy_{side}"] = np.array([int(n.rsplit("_", 1)[1]) for n in on[self.idx[f"q_{side}"]]])
        for n in ("q_peak_outer", "q_peak_inner", "Te_strike_outer", "ne_strike_outer",
                  "Te_strike_inner", "ne_strike_inner", "P_rad"):
            self.idx[n] = np.flatnonzero(on == n)

    def predict(self, inputs) -> np.ndarray:
        if isinstance(inputs, dict):
            X = np.stack(np.broadcast_arrays(*[np.atleast_1d(np.asarray(inputs[n], float))
                                               for n in self.input_names]), 1)
        else:
            X = np.atleast_2d(np.asarray(inputs, float))
        v = X.T.copy()
        v[self.x_log] = np.log10(v[self.x_log])
        v = (v - self.x_mean[:, None]) / self.x_std[:, None]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            v = W @ v + b[:, None]
            if i < len(self.W) - 1:
                v = np.maximum(v, 0.0)
        v = v * self.y_std[:, None] + self.y_mean[:, None]
        v[self.y_log] = 10.0 ** v[self.y_log]
        return v.T

    def psi_n(self, rho_c: float, psi_axis_to_sep: float | None = None) -> np.ndarray:
        dpsi = self.mesh["dpsi"]
        span = dpsi[0] / (rho_c - 1.0) if psi_axis_to_sep is None else abs(psi_axis_to_sep)
        return 1.0 + dpsi / span

    def result(self, y: np.ndarray, rho_c: float = 0.95, psi_axis_to_sep: float | None = None,
               params: dict | None = None) -> dict:
        y = np.asarray(y).ravel(); x = self.psi_n(rho_c, psi_axis_to_sep); closed = self.mesh["closed"]
        xc = x[closed]
        if not (xc.min() - 1e-9 <= rho_c <= xc.max() + 1e-9):
            raise ValueError(f"rho_c={rho_c} outside the closed rows ({xc.min():.4f} .. {xc.max():.4f})")
        g = lambda k: y[self.idx[k]]
        bc, prof = {}, {"iy": self.mesh["rows"].tolist(), "x": x.tolist(), "R_omp": self.mesh["R_omp"].tolist(),
                        "dpsi": self.mesh["dpsi"].tolist(), "closed": closed.tolist()}
        for k in ("te", "ti", "ne", "ni"):
            fsa, omp = g(f"{k}_fsa"), g(f"{k}_omp")
            bc[k] = {"fsa": float(np.interp(rho_c, xc, fsa)), "omp": float(np.interp(rho_c, xc, omp[closed]))}
            full = np.full(len(x), np.nan); full[closed] = fsa
            prof[f"{k}_fsa"] = [None if np.isnan(v) else float(v) for v in full]; prof[f"{k}_omp"] = omp.tolist()
        tg = {"P_rad_W": float(g("P_rad")[0])}
        for side in ("inner", "outer"):
            q, iy = g(f"q_{side}"), self.idx[f"iy_{side}"]
            tg[f"{side}_target"] = {"q_peak_W_m2": float(g(f"q_peak_{side}")[0]), "iy_peak": int(iy[np.argmax(q)]),
                                    "Te_strike_eV": float(g(f"Te_strike_{side}")[0]),
                                    "ne_strike_m3": float(g(f"ne_strike_{side}")[0]),
                                    "q_profile_W_m2": q.tolist(), "iy": iy.tolist()}
        r = {"model": "SOLSTICE COTSIM MLP (pepc-diiid-cotsim-677-mlp)", "rho_c": rho_c}
        if params:
            r["cotsim_boundary_inputs"] = params
        r.update(boundary_conditions=bc, targets=tg, profiles=prof, ni_definition="ni = na_D1")
        return r


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5")
    ap.add_argument("--reference", help="write the demo outputs here (checked by matlab/cotsim_mlp_demo.m)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter(); nn = CotsimMLP(a.h5); t_load = time.perf_counter() - t0
    y = nn.predict(DEMO_PARAMS)[0]; r = nn.result(y, 0.95, params=DEMO_PARAMS)
    for k, u in (("te", "eV"), ("ti", "eV"), ("ne", "m^-3"), ("ni", "m^-3")):
        print(f"  {k:3s} FSA={r['boundary_conditions'][k]['fsa']:.6e} {u}, OMP={r['boundary_conditions'][k]['omp']:.6e} {u}")
    print(f"P_rad = {r['targets']['P_rad_W']:.6e} W")
    for t in ("inner_target", "outer_target"):
        print(f"{t}: q_peak = {r['targets'][t]['q_peak_W_m2']:.6e} W/m^2")
    n = 2000; t0 = time.perf_counter()
    for _ in range(n):
        nn.predict(DEMO_PARAMS)
    t1 = (time.perf_counter() - t0) / n
    X = np.tile([DEMO_PARAMS[k] for k in nn.input_names], (1000, 1)); t0 = time.perf_counter(); nn.predict(X)
    tb = (time.perf_counter() - t0) / 1000
    print(f"load {t_load:.3f} s | predict one case {t1*1e6:.0f} us | batch of 1000 {tb*1e6:.1f} us per case")
    if a.reference:
        json.dump({"params": DEMO_PARAMS, "input_names": nn.input_names, "output_names": nn.output_names,
                   "outputs": y.tolist(), "result": r}, open(a.reference, "w"), indent=1)
        print("wrote", a.reference)


if __name__ == "__main__":
    sys.exit(main())
