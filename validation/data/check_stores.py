"""Sanity check of the SOLPS-NN stores used for training and validation.

    python validation/data/check_stores.py [--md5]      # stores from $SOLSTICE_DATA or ./stores

Prints case counts, qc, regime split (paper: attached 38 / sheath-limited 27 / detached 20 /
cold core 14 %), the 8 scan inputs after the pe+pi -> ptot and hci=hce -> chi merge, the field
list, and asserts the train and test case ids do not overlap. --md5 compares with the
checksums recorded when the stores were built (2.5 GB, ~1 min).
"""
import argparse, hashlib, os, sys
from pathlib import Path
import numpy as np, xarray as xr

EXPECTED = {   # bytes, md5 recorded at build time (scripts/nersc/HANDOFF.md section 1)
    "solstice_store_solpsnn_v1_train.nc": (2_059_983_848, "f906a1979f2897b61db83c8adb28ce0d", 5756),
    "solstice_store_solpsnn_v1_test.nc": (524_168_162, "aa0a4c8c2fbb96fb893848a149f645d6", 1465),
}
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=os.environ.get("SOLSTICE_DATA", str(Path(__file__).parent / "stores")))
ap.add_argument("--md5", action="store_true")
a = ap.parse_args()
ids, ok = {}, True
for name, (nbytes, md5, ncase) in EXPECTED.items():
    f = Path(a.data) / name
    print(f"== {f}")
    size = f.stat().st_size
    print(f"   size {size:,} bytes ({'ok' if size == nbytes else 'MISMATCH, expected %d' % nbytes})")
    ok &= size == nbytes
    if a.md5:
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
        print(f"   md5 {h.hexdigest()} ({'ok' if h.hexdigest() == md5 else 'MISMATCH, expected ' + md5})")
        ok &= h.hexdigest() == md5
    d = xr.open_dataset(f)
    n = d.sizes["case"]; print(f"   cases {n} (expected {ncase}), qc_pass {int(d.qc_pass.sum())}, cells {d.sizes['cell']}")
    ok &= n == ncase
    reg, cnt = np.unique(d.regime.values, return_counts=True)
    print("   regimes " + ", ".join(f"{r} {100*c/n:.0f}%" for r, c in zip(reg, cnt)))
    inputs = sorted(v[6:] for v in d.data_vars if v.startswith("input_"))
    print(f"   inputs ({len(inputs)}): {' '.join(inputs)}  -> 8 after pe+pi->ptot, hci=hce->chi")
    fields = sorted(v for v in d.data_vars if ("case", "cell") == d[v].dims)
    print(f"   per-cell fields ({len(fields)}): {' '.join(fields)}")
    ids[name] = set(d.case.values); d.close()
tr, te = ids.values()
print(f"== train/test overlap: {len(tr & te)} cases" + (" (ok)" if not (tr & te) else " (BAD)"))
ok &= not (tr & te)
print("ALL OK" if ok else "PROBLEMS FOUND"); sys.exit(0 if ok else 1)
