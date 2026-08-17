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
"""Call a released SOLSTICE sources model: plasma state -> EIRENE source terms.

    python examples/predict_sources.py /path/to/pepc-diiid-sources-v1 \
        [/path/to/solstice_store.nc [case_index]]

With a store file, the plasma state of one SOLPS case is used as input and
the prediction can be compared against that case's true sources. Without
one, a synthetic mid-range plasma state exercises the model.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from solstice.hub import load_source_bundle

model = load_source_bundle(sys.argv[1])
pf = model.manifest["variables"]["plasma_features"]
print(f"loaded {model.manifest['name']}: plasma {list(pf)} -> "
      f"{model.manifest['variables']['outputs']}")

n_cells = model.mesh.sizes["cell"]
# the model's true degrees of freedom (pe = pi -> ptot, hci = hce -> chi)
params = {"ptot": 6.0e6, "chi": 0.7, "core_fueling": 3.0e20,
          "puff_D2": 1.0e21, "dna": 0.5}

if len(sys.argv) > 2:  # real plasma state from a training store
    import xarray as xr
    ds = xr.open_dataset(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    plasma = {name: ds[name].values[k] for name in pf}
    params = {v[6:]: float(ds[v].values[k]) for v in ds.data_vars
              if v.startswith("input_")}
    print(f"plasma state from case {ds.case.values[k]}")
else:  # synthetic mid-range state
    rho = np.hypot(model.mesh.cell_r.values - model.mesh.cell_r.values.mean(),
                   model.mesh.cell_z.values - model.mesh.cell_z.values.mean())
    shape = np.exp(-3 * rho / rho.max())
    plasma = {}
    for name in pf:
        if name.startswith("t"):
            plasma[name] = 5.0 + 500.0 * shape          # eV
        elif name.startswith(("n", "na")):
            plasma[name] = 1e18 + 5e19 * shape          # m^-3
        else:
            plasma[name] = np.zeros(n_cells)            # velocities

sources = model.predict(plasma, params)
for name, values in sources.items():
    print(f"  {name:5s} min {values.min():10.3e}  max {values.max():10.3e}")

verts = np.stack([model.mesh.cell_corners_r.values,
                  model.mesh.cell_corners_z.values], axis=-1)
fig, axs = plt.subplots(1, 2, figsize=(9, 6))
for ax, name in zip(axs, ["sp", "qe"]):
    v = sources[name]
    lim = np.max(np.abs(v))
    pc = PolyCollection(verts, array=v, cmap="RdBu_r", edgecolor="none",
                        clim=(-lim, lim))
    ax.add_collection(pc)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(name)
    plt.colorbar(pc, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig("sources_prediction.png", dpi=150)
print("wrote sources_prediction.png")
