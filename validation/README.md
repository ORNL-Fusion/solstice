# JET validation

This directory contains the SOLSTICE JET state model (`pepc-jet-state-v1`), its data preparation, and test-set evaluation.

## Dataset

The model uses the SOLPS-NN Training Data v1 release: a SOLPS-ITER scan on a JET-shaped 102 × 48 grid, scaled from 1 to 10 m major radius. The supplied split contains 5,756 training cases and 1,465 test cases. The data and paper are cited below and described in [`data/README.md`](data/README.md).

## Model and results

The model predicts 27 per-cell outputs and uses a 20-member ensemble. On the held-out test set, median relative errors are 4.2% for `te`, 4.9% for `ti`, and 7.6% for `ne` across all cells. Cold-core cases, parallel velocities, and ensemble uncertainty remain the main limitations.

Full metrics are in [`ens20_mean.json`](models/jet_heads5_h1024_ft_te_ens20/ens20_mean.json); evaluation plots are in [`jet_eval_heads5_ens20.ipynb`](notebooks/jet_eval_heads5_ens20.ipynb). Model details and the training recipe are in [`models/jet_heads5_h1024_ft_te_ens20/README.md`](models/jet_heads5_h1024_ft_te_ens20/README.md).

## Reproduce

With the SOLSTICE environment active and `SOLSTICE_DATA` pointing to the two stores:

```bash
python validation/data/check_stores.py --md5
sbatch --array=0-19 validation/models/jet_heads5_h1024_ft_te_ens20/recipe/ensemble.sbatch
bash validation/models/jet_heads5_h1024_ft_te_ens20/evaluate.sh
sbatch validation/notebooks/run_notebook.sbatch
```

Model weights are distributed in the `pepc-jet-state-v1` release bundle.

## References

- S. Dasbach, S. Brezinsek, Y. Liang, D. Reiser, and S. Wiesen, “Deep-Learning based surrogate models for plasma exhaust simulations — SOLPS-NN,” arXiv:2604.19223 (2026), [arXiv](https://arxiv.org/abs/2604.19223).
- S. Dasbach, S. Brezinsek, Y. Liang, D. Reiser, and S. Wiesen, *SOLPS-NN Training Data v1* (2026), [Zenodo](https://doi.org/10.5281/zenodo.19237127), CC BY 4.0.
