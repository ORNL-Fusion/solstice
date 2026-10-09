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
"""solpex-paper coupling_dataset*.npz -> canonical cases.

The legacy fixed channel-index conventions live only here. Conversion is
lossless: (ix, iy) are retained as cell labels so image-based loaders
can reconstruct the padded (ny, nx) view.

Status: skeleton — pending validation of legacy geometry against
SOLPS-routines readers before conversion (known risk: legacy psi_n and
mask conventions may not match the expert readers).
"""

from __future__ import annotations

# Legacy channel orders (from solpex-paper build_coupling_dataset.py):
LEGACY_PLASMA = ("Te", "Ti", "ne", "ni", "ua", "vol", "hx", "hy",
                 "bb0", "bb1", "bb2", "bb3", "R", "Z")
LEGACY_SOURCES = ("Sp", "Sne", "Qe", "Qi", "Sm", "dab2", "dmb2", "tab2", "tmb2")


def convert_dataset(npz_path: str, out_dir: str):
    raise NotImplementedError("pending: legacy npz -> canonical case store")
