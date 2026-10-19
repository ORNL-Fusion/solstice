#!/bin/bash
# Build the two SOLSTICE stores from the SOLPS-NN Training Data v1 release (Dasbach et al.,
# Zenodo 10.5281/zenodo.19237127, 17.5 GB unzipped: baserun/, train/, test/).
# Converter: src/solstice/data/converters/from_solpsnn.py (entry point `solstice-store-solpsnn`).
# Needs SOLPS-routines on PYTHONPATH for the baserun mesh; runs on a laptop in ~1 h, then
# copy the two .nc files to $SOLSTICE_DATA on Perlmutter (Globus or scp) and run check_stores.py.
#
#   bash validation/data/build_stores.sh <release_dir> <out_dir>
set -euo pipefail
REL=${1:?unzipped Zenodo release dir}; OUT=${2:-$(dirname "$0")/stores}
export PYTHONPATH=${SOLPS_ROUTINES:-/path/to/SOLPS-routines/src}:${PYTHONPATH:-}
solstice-store-solpsnn "$REL" "$OUT" --splits test train --prefix solstice_store_solpsnn_v1
ls -l "$OUT"/solstice_store_solpsnn_v1_*.nc
