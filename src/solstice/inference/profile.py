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
"""Model cost profiling: parameters, memory, inference latency.

Every released bundle reports this block (checkpoint_spec.md `cost`);
real-time consumers (control loops target ~1 ms per step) read it to
decide where a model can run.
"""

from __future__ import annotations

import time


def profile(core, example_input, n_warmup: int = 10, n_reps: int = 100) -> dict:
    """Cost report for a torch module.

    example_input: a representative input tensor (batch dim included);
    latency is per forward call of that batch, so pass batch=1 for the
    real-time number and a large batch for throughput.
    """
    import numpy as np
    import torch

    device = next(core.parameters()).device
    x = example_input.to(device)
    n_params = sum(p.numel() for p in core.parameters())
    weights_mb = sum(p.numel() * p.element_size() for p in core.parameters()) / 1e6

    core.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            core(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            core(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1e3  # ms
    batch = int(x.shape[0]) if x.ndim > 1 else 1
    return {
        "n_params": int(n_params),
        "weights_mb": round(weights_mb, 3),
        "device": str(device),
        "batch": batch,
        "latency_ms_median": round(float(np.median(times)), 4),
        "latency_ms_p95": round(float(np.percentile(times, 95)), 4),
        "throughput_per_s": round(batch / (np.median(times) / 1e3), 1),
    }
