# validation/ — SOLSTICE JET state model (pepc-jet-state-v1)

Everything behind the JET state model: the dataset of Dasbach et al. (JET-shaped grid
scaled with the major radius from 1 to 10 m) and how it was turned into training stores,
the trained model, and the notebooks that test it.

```
validation/
  data/        data release -> solstice stores (build + check scripts, data notebook)
  models/      the trained model: 20 ensemble members, recipe, test metrics
  notebooks/   test notebooks (generator, template, executed copy, batch job)
```

## Data

SOLPS-NN Training Data v1, Dasbach et al., Zenodo 10.5281/zenodo.19237127 (CC BY 4.0).
SOLPS-ITER scan on one 102 x 48 mesh scaled with the major radius, 8 scalar inputs.
Train 5756 cases, test 1465 cases, no overlap. Details in `data/README.md`.

## Model

GNN state model for the JET-shaped grid, 8 inputs -> 27 per-cell outputs (te, ti, ne, 10 densities, 10 parallel
velocities, q_pol, prad, and the target heat and particle flux profiles). 20 seeds trained
with the same recipe and used as a deep ensemble (mean = prediction, spread = error
estimate). Details and the exact training command in `models/.../README.md`.

## Results

Test set, 1465 cases, 20-member ensemble mean. Median relative error per cell.

| field | all cells | outer target | OMP separatrix |
|---|---|---|---|
| te | 4.2 % | 8.9 % | 2.5 % |
| ti | 4.9 % | 9.5 % | 3.2 % |
| ne | 7.6 % | 15.5 % | 4.3 % |
| na_D1 | 8.4 % | 16.7 % | 5.0 % |
| na_N0 | 9.9 % | 23.8 % | 4.4 % |
| ua_D1 | 21.6 % | 9.5 % | 37.6 % |
| q_pol | 20.7 % | 14.6 % | 19.0 % |
| prad | 16.5 % | 18.8 % | 10.0 % |
| q_target | 10.4 % | 10.2 % | |
| gamma_target | 12.3 % | 13.1 % | |

Te by regime: attached 2.3 %, detached 3.7 %, sheath-limited 4.4 %, cold core 24.4 %.
Single seeds: Te all cells 5.2 +- 0.2 %.

Weak spots: cold-core cases (target heads unusable there), parallel velocities,
and the ensemble spread, which under-estimates the error (about 60 % of cells
within one standard deviation).

The released bundle (`pepc-jet-state-v1`) averages the members in log space
and gives Te 4.3 % / 9.2 % / 2.5 % on the same cases (recorded in its bundle.json).

Full numbers: `models/jet_heads5_h1024_ft_te_ens20/ens20_mean.json`.
Figures: `notebooks/jet_eval_heads5_ens20.ipynb`.

## Reproduce

From the repo root with the solstice environment active and `SOLSTICE_DATA` set:

1. `python validation/data/check_stores.py --md5`
2. `sbatch --array=0-19 validation/models/jet_heads5_h1024_ft_te_ens20/recipe/ensemble.sbatch`
3. `bash validation/models/jet_heads5_h1024_ft_te_ens20/evaluate.sh`
4. `sbatch validation/notebooks/run_notebook.sbatch`

The model weights (20 x 94 MB) are not in git; they ship as the `pepc-jet-state-v1`
release bundle.
