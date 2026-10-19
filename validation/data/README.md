# Dataset and SOLSTICE stores

## Source

S. Dasbach, S. Brezinsek, Y. Liang, D. Reiser, and S. Wiesen, “Deep-Learning based surrogate models for plasma exhaust simulations — SOLPS-NN,” arXiv:2604.19223 (2026), [paper](https://arxiv.org/abs/2604.19223).

S. Dasbach, S. Brezinsek, Y. Liang, D. Reiser, and S. Wiesen, *SOLPS-NN Training Data v1* (2026), [Zenodo](https://doi.org/10.5281/zenodo.19237127), CC BY 4.0.

The release contains a SOLPS-ITER scan on a JET-shaped 102 × 48 mesh scaled from 1 to 10 m major radius, with eight scan parameters and separate train (5,756 cases) and test (1,465 cases) splits. This validation uses the supplied split without overlap.

## Conversion

The converter is [`from_solpsnn.py`](../../src/solstice/data/converters/from_solpsnn.py); [`build_stores.sh`](build_stores.sh) records the command used. It requires the raw release and SOLPS-routines for the baserun mesh. The resulting NetCDF stores live outside Git under `$SOLSTICE_DATA`; [`check_stores.py`](check_stores.py) checks their sizes, checksums, contents, and split separation.

The baserun mesh is stored at `R_JET = 3.0007 m`. For each case, coordinates and lengths scale by `R/R_JET`, and volumes scale by its cube. The stored ten input columns are reduced during training to the paper’s eight parameters: `pe + pi` becomes `ptot`, and `hci = hce` becomes `chi`. The stores contain 25 cell fields and target heat and particle flux profiles.

## Files

- `dasbach_dataset.ipynb` and `make_dasbach_dataset_nb.py`: analysis of the original release; the raw 17.5 GB dataset is required.
- `build_stores.sh`: store conversion command.
- `check_stores.py`: store integrity and split checks.
- `stores/`: local links to the NetCDF files; not tracked by Git.
