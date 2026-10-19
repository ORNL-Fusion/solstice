#!/bin/bash
# Execute the SOLPS-NN validation notebook on the 20-seed ensemble (one shared A100, ~30 min).
#   sbatch validation/notebooks/run_notebook.sbatch                       # from the repo root
#   SOLSTICE_RUN='runs/foo/seed*' NB_OUT=jet_eval_foo.ipynb sbatch validation/notebooks/run_notebook.sbatch
#SBATCH -A m5186
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -t 1:30:00
#SBATCH -J solstice-nb
#SBATCH -o logs/slurm/slurm-%x-%j.out
set -euo pipefail
cd /pscratch/sd/d/diawa/nn/solstice/validation/notebooks
module load ${PYTORCH_MODULE:-pytorch/2.13.0}
source "${SOLSTICE_VENV:-$SCRATCH/venvs/solstice}/bin/activate"
export PYTHONPATH=/pscratch/sd/d/diawa/nn/solstice/src SOLSTICE_DATA=${SOLSTICE_DATA:-/pscratch/sd/d/diawa/nn} OMP_NUM_THREADS=8
export SOLSTICE_RUN=${SOLSTICE_RUN:-'validation/models/jet_heads5_h1024_ft_te_ens20/seed*'}
python make_jet_eval_nb.py      # regenerate the template from the generator
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
  jet_eval.ipynb --output ${NB_OUT:-jet_eval_heads5_ens20.ipynb}
