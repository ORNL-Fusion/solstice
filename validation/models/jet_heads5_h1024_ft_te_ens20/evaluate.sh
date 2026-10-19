#!/bin/bash
# Re-evaluate the 20-member deep ensemble on the test store (one GPU, ~10 min) and rewrite
# ens20_mean.json. Run from anywhere; needs the stores in $SOLSTICE_DATA.
#   bash validation/models/jet_heads5_h1024_ft_te_ens20/evaluate.sh [--device cuda:0]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../../.."        # repo root
module load ${PYTORCH_MODULE:-pytorch/2.13.0}
source "${SOLSTICE_VENV:-$SCRATCH/venvs/solstice}/bin/activate"
export PYTHONPATH=$PWD/src SOLSTICE_DATA=${SOLSTICE_DATA:-/pscratch/sd/d/diawa/nn} OMP_NUM_THREADS=8
python scripts/nersc/ensemble_eval.py "$HERE"/seed{0..19} --out "$HERE/ens20_mean.json" "$@" | tee "$HERE/evaluate.log"
