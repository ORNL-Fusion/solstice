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
"""Call a released SOLSTICE state model: control parameters -> plasma background.

    python examples/predict_state.py /path/to/pepc-diiid-state-v1

The bundle is self-contained (weights, normalization, mesh); inputs are raw
physical control parameters. Out-of-training-box requests raise a warning —
treat those predictions as extrapolations.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from solstice.hub import load_state_bundle

model = load_state_bundle(sys.argv[1])
print(f"loaded {model.manifest['name']}: "
      f"{model.manifest['model']['class']} -> {list(model.fields)}")

# a mid-range DIII-D operating point (SI units)
# the model's true degrees of freedom (training data has pe = pi and
# hci = hce, so the model sees ptot = pe + pi and a single chi)
params = {
    "ptot": 6.0e6,           # total core-boundary power pe + pi [W]
    "core_fueling": 3.0e20,  # core fueling parameter
    "puff_D2": 1.0e21,       # D2 gas puff [atom/s]
    "dna": 0.5, "chi": 0.7,  # transport coefficients [m^2/s]
}
fields = model.predict(params)
for name, values in fields.items():
    print(f"  {name:8s} min {values.min():10.3e}  max {values.max():10.3e}")

# plot Te and ne on the bundled mesh
verts = np.stack([model.mesh.cell_corners_r.values,
                  model.mesh.cell_corners_z.values], axis=-1)
fig, axs = plt.subplots(1, 2, figsize=(9, 6))
for ax, (name, log) in zip(axs, [("te", False), ("ne", True)]):
    v = np.log10(np.clip(fields[name], 1e-30, None)) if log else fields[name]
    pc = PolyCollection(verts, array=v, cmap="viridis", edgecolor="none")
    ax.add_collection(pc)
    ax.autoscale(); ax.set_aspect("equal")
    ax.set_title(("log10 " if log else "") + name)
    plt.colorbar(pc, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig("state_prediction.png", dpi=150)
print("wrote state_prediction.png")
