#!/bin/bash
# Seed ensembles: N independently seeded trainings of one configuration, one
# A100 80 GB each in the shared queue. Gives the deep-ensemble spread (the
# uncertainty estimate the roadmap asks for) and the seed-to-seed variance
# of every metric, which the hyperparameter comparisons need anyway.
#
# Default model is the current reference, heads5_h1024 (+ plasma-head Te
# fine-tune): d2048 base (quantile, MAE, cell_embed 32, params_to_cells,
# extended nodes, decoder 2048, 300 ep) with 5 grouped decoder heads of width
# 1024, then 60 epochs on decoders.0 only (te x5, lr 3e-4) from best.pt.
# One seed is ~1.2 h (heads5 ~55 min + ft ~5 min + data loading).
#
#   sbatch scripts/nersc/ensemble.sbatch                     # 5 seeds, heads5 + Te ft
#   KENDALL=1 sbatch scripts/nersc/ensemble.sbatch           # heads5 + Kendall weighting
#   FT_TE=0 sbatch --array=0-19 scripts/nersc/ensemble.sbatch
#   python scripts/nersc/ensemble_eval.py runs/ensemble/heads5_h1024/seed*_ft_te
#
#SBATCH -A m5186
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -t 2:30:00
#SBATCH --array=0-4
#SBATCH -J solstice-ens
#SBATCH -o logs/ens/slurm-%x-%A_%a.out
set -euo pipefail

KENDALL=${KENDALL:-0}; FT_TE=${FT_TE:-1}; SEED=$SLURM_ARRAY_TASK_ID
HEADS='model.heads={plasma: [te, ti, ne], deut: ["na_D*"], nitro: ["na_N*"], flow: ["ua_*", q_pol], target: [q_target, gamma_target, prad]}'
TAG=heads5_h1024; W=()
[ "$KENDALL" = 1 ] && { TAG=${TAG}_kendall; W=(--set train.loss_weighting=kendall); }

cd /pscratch/sd/d/diawa/nn/solstice
module load ${PYTORCH_MODULE:-pytorch/2.13.0}
source "${SOLSTICE_VENV:-$SCRATCH/venvs/solstice}/bin/activate"
export PYTHONPATH=$PWD/src SOLSTICE_DATA=${SOLSTICE_DATA:-/pscratch/sd/d/diawa/nn} OMP_NUM_THREADS=8
MODEL=(--config configs/training/solpsnn_state.yaml --set data.output_transform=quantile --set train.loss=mae
  --set model.config.cell_embed=32 --set model.config.params_to_cells=true --set graph.node_features=extended
  --set model.config.decoder_hidden=2048 --set "$HEADS" --set model.config.head_hidden=1024 --set seed=$SEED)
OUT=runs/ensemble/$TAG/seed$SEED

echo "seed $SEED: $TAG (ft_te=$FT_TE) -> $OUT"
[ -f $OUT/metrics.json ] || python -m solstice.training.cli "${MODEL[@]}" "${W[@]}" \
  --set train.epochs=300 --set train.max_hours=1.8 --set out_dir=$OUT
[ -f $OUT/metrics.json ] || { echo "$OUT stopped before finishing; resubmit to resume"; exit 0; }
[ "$FT_TE" = 1 ] && [ ! -f ${OUT}_ft_te/metrics.json ] && python -m solstice.training.cli "${MODEL[@]}" \
  --set train.epochs=60 --set train.init_from=$OUT/best.pt --set 'train.trainable=["decoders.0.*"]' \
  --set 'train.field_weights={te: 5}' --set train.lr=3e-4 --set train.max_hours=0.4 --set out_dir=${OUT}_ft_te
exit 0
