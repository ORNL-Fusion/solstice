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
"""Released models. Names follow pepc-{machine}-{task}-v{N}; weights are
downloaded from GitHub Releases on first use and cached locally."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

# release name -> (git tag, asset filename)
RELEASES: dict[str, tuple[str, str]] = {
    "pepc-diiid-state-v1": ("v0.1.0", "pepc-diiid-state-v1.zip"),
    "pepc-diiid-sources-v1": ("v0.1.0", "pepc-diiid-sources-v1.zip"),
}
_RELEASE_URL = "https://github.com/ORNL-Fusion/solstice/releases/download/{tag}/{asset}"
_CACHE = Path(os.environ.get("SOLSTICE_CACHE", Path.home() / ".cache" / "solstice"))


def load(name: str):
    """Download (once) and load a released model by name."""
    from solstice.inference import load_checkpoint

    if name not in RELEASES:
        raise KeyError(f"unknown release {name!r}; available: {sorted(RELEASES)}")
    bundle_dir = _CACHE / name
    if not (bundle_dir / "bundle.json").exists():
        tag, asset = RELEASES[name]
        url = _RELEASE_URL.format(tag=tag, asset=asset)
        _CACHE.mkdir(parents=True, exist_ok=True)
        zpath = _CACHE / asset
        import urllib.request
        print(f"downloading {name} from {url}")
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(_CACHE)
        zpath.unlink()
    return load_checkpoint(bundle_dir)


from solstice.hub.bundle import (create_source_bundle, create_state_bundle,  # noqa: E402,F401
                                 load_source_bundle, load_state_bundle)
