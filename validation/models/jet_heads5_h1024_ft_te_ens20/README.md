# JET state model ensemble

This 20-member SOLSTICE GNN ensemble is released as `pepc-jet-state-v1`. Members were trained independently on the SOLPS-NN training split, then fine-tuned on `te`. The held-out test metrics are recorded in `ens20_mean.json`.

## Files

- `ens20_mean.json`: metrics for all 20 members.
- `ens5_mean.json`: metrics for members 0–4.
- `parents/ens5_mean_noft.json`: pre-fine-tuning metrics for members 0–4.
- `recipe/ensemble.sbatch`: Slurm training job.
- `recipe/solpsnn_state.yaml`: base training configuration.
- `evaluate.sh`: command to recompute ensemble metrics.

Checkpoint weights are distributed in the release bundle rather than Git. The evaluation notebook is in `validation/notebooks/`.
