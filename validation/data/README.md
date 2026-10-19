# Data: SOLPS-NN Training Data v1 (Dasbach et al.) -> SOLSTICE stores

Source: S. Dasbach et al., "SOLPS-NN", arXiv:2604.19223; data release
"SOLPS-NN Training Data v1", Zenodo 10.5281/zenodo.19237127 (CC BY 4.0, 17.5 GB unzipped,
`baserun/`, `train/`, `test/`). A SOLPS-ITER scan (B2.5, fluid neutrals, no drifts) on one
JET-shaped 102 x 48 grid (4896 cells) scaled with the major radius R from 1 to 10 m at fixed
aspect ratio; 8 scalar inputs sampled by LHS; species D0, D+, N0..N7+. The release's own
train/test split is kept one to one (train 5756 cases, test 1465, no overlap).

## Processing (what the converter does)

Code: `src/solstice/data/converters/from_solpsnn.py` (CLI `solstice-store-solpsnn`,
unit tests `tests/test_from_solpsnn.py`). Wrapper here: `build_stores.sh`. Needs
SOLPS-routines for the baserun mesh; was run on a Mac, then the stores were copied to
Perlmutter (sizes and md5 in `check_stores.py`).

- Mesh: the unscaled baserun mesh is stored once (R_JET = 3.0007 m); the physical mesh of
  case k is `cell_r, cell_z, corners, vx` x `mesh_scale[k]` (= R / R_JET), volumes x scale^3.
  `cell_b` is the baserun field (B enters as the `btor` input).
- Inputs (`input_*`, 10 columns): rmajor, btor, pe, pi, puff_D, puff_N, core_fueling, dna,
  hci, hce. The paper's 8 parameters come back at training time
  (`solstice.training.data.engineer_inputs`): pe + pi -> ptot, hci = hce -> chi.
  puff_D, puff_N, core_fueling are log-uniform in the scan and are log-transformed.
- Per-cell fields (25): te, ti, ne, na_{D0,D1,N0..N7}, ua_{D0,D1,N0..N7}, q_pol, prad.
- Target profiles (`target_iy` dimension, 48 rows): q_{inner,outer}_target and
  gamma_{inner,outer}_target, face-centred heat and D+ flux on the two plates; they
  reproduce the release's scalar QoIs `pwmxap` / `fnixap` to machine precision. The
  training config turns them into the two auxiliary output heads q_target / gamma_target
  supervised only on target-adjacent cells.
- `regime` per case (sheath-limited / attached / detached / cold core, from the release's
  Te criteria); split matches the paper's 27 / 38 / 20 / 14 %.
- `qc_pass`: NaN fraction <= 2 % of cells; every case in both splits passes.

## Training-side transforms (in `solstice.training.data`)

Inputs: standardised after the log transform above. Outputs: fields log10 (te, ti, ne,
na_*, prad) or linear (ua_*, q_pol, q_target, gamma_target), then a per-cell quantile map
(100 bins, `data.output_transform: quantile`, after the paper's per-cell QuantileTransformer)
fitted on each run's own train split and stored in `best.pt` (`y_quantiles`, shape
4896 x 27 x 100). Validation holdout 10 % of train. Note for ensembles: every member has its
own quantile map, so predictions are averaged in one member's normalisation
(`OutputNorm.renormalize`), as `scripts/nersc/ensemble_eval.py` and the notebook do.

## Files

- `dasbach_dataset.ipynb` (+ generator `make_dasbach_dataset_nb.py`) — exploration of the raw
  release on the Mac: reproduces the paper's Section 2 figures and the Section 5 QoIs from the
  raw files (needs the 17.5 GB release, not the stores).

- `build_stores.sh` — exact converter command.
- `check_stores.py` — sizes, md5 (`--md5`), counts, regimes, inputs, fields, overlap check.
- `stores/` — symlinks to the two `.nc` stores in `$SOLSTICE_DATA` (gitignored).
