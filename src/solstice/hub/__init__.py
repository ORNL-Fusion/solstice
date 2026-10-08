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
downloaded from GitHub Releases on first use and cached locally
($SOLSTICE_CACHE, default ~/.cache/solstice). For a private repository set
$GITHUB_TOKEN (or $SOLSTICE_GITHUB_TOKEN); the download then goes through the API."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

# release name -> (git tag, asset filename[, github repo]); default repo = _RELEASE_REPO
_RELEASE_REPO = "ORNL-Fusion/solstice"
RELEASES: dict[str, tuple] = {
    "pepc-diiid-state-v1": ("v0.1.0", "pepc-diiid-state-v1.zip", "ORNL-Fusion/solstice"),
    "pepc-diiid-sources-v1": ("v0.1.0", "pepc-diiid-sources-v1.zip", "ORNL-Fusion/solstice"),
    "pepc-jet-state-v1": ("v0.2.1", "pepc-jet-state-v1.zip"),
    "pepc-diiid-state-v2": ("v0.2.1", "pepc-diiid-state-v2.zip"),
    "pepc-diiid-cotsim-284-state": ("v0.2.1", "pepc-diiid-cotsim-284-state.zip"),
}
_RELEASE_URL = "https://github.com/{repo}/releases/download/{tag}/{asset}"
_CACHE = Path(os.environ.get("SOLSTICE_CACHE", Path.home() / ".cache" / "solstice"))


def _download(repo: str, tag: str, asset: str, dest: Path) -> None:
    """Fetch a release asset. Anonymous from the public download URL; with a token in
    $SOLSTICE_GITHUB_TOKEN or $GITHUB_TOKEN through the API (needed for private repos)."""
    import json
    import urllib.request

    token = os.environ.get("SOLSTICE_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        url = _RELEASE_URL.format(repo=repo, tag=tag, asset=asset)
        print(f"downloading {asset} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(f"{url} not found: the release is missing or the repository is "
                                   "private (set GITHUB_TOKEN to download from a private repo)") from e
            raise
        return
    hdr = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=hdr)
    with urllib.request.urlopen(req) as r:
        assets = {a["name"]: a["url"] for a in json.load(r)["assets"]}
    if asset not in assets:
        raise RuntimeError(f"release {tag} of {repo} has no asset {asset}; has {sorted(assets)}")
    print(f"downloading {asset} from {repo} release {tag} (authenticated)")
    req = urllib.request.Request(assets[asset], headers={**hdr, "Accept": "application/octet-stream"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def load(name: str):
    """Download (once) and load a released model by name."""
    from solstice.inference import load_checkpoint

    if name not in RELEASES:
        raise KeyError(f"unknown release {name!r}; available: {sorted(RELEASES)}")
    bundle_dir = _CACHE / name
    if not (bundle_dir / "bundle.json").exists():
        tag, asset, *repo = RELEASES[name]
        _CACHE.mkdir(parents=True, exist_ok=True)
        zpath = _CACHE / asset
        _download(repo[0] if repo else _RELEASE_REPO, tag, asset, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(_CACHE)
        zpath.unlink()
    return load_checkpoint(bundle_dir)


from solstice.hub.bundle import (create_source_bundle, create_state_bundle,  # noqa: E402,F401
                                 load_source_bundle, load_state_bundle)
