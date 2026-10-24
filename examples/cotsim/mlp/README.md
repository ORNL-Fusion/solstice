# SOLSTICE COTSIM MLP

A small ReLU MLP distilled from the SOLSTICE GNN `pepc-diiid-cotsim-677-state` for the REACT
COTSIM coupling. It maps the 9 operating-point inputs directly to the quantities COTSIM uses
and runs in MATLAB from one HDF5 file, with no Python.

## Files

| File | Purpose |
|---|---|
| `matlab/read_NN_solstice.m` | MATLAB reader adapted from the COTSIM `read_NN` class (`predict`, `result`, `at_rho`) |
| `matlab/plot_NN_solstice.m` | MATLAB plots as in the notebook: profiles vs psi_N, target heat flux |
| `matlab/cotsim_mlp_demo.m` | MATLAB demo: notebook operating point, print, plot, JSON, check vs Python, timing |
| `solstice_cotsim_mlp.py` | numpy + h5py reader with the same API |
| `MODEL_CARD.md` | becomes README.md of the release |
| `package_release.py` | export + verify + assemble `release/pepc-diiid-cotsim-677-mlp/` and its zip |
| `reduce.py` | 2D fields -> COTSIM quantities (vectorized `examples/cotsim/boundary.py`, identical to machine precision) |
| `make_targets.py` | teacher set: GNN on 30k sampled inputs + SOLPS truth of the 677 store cases |
| `train_mlp.py`, `common.py` | training of the student |
| `export_h5.py`, `verify_h5.py` | h5 export, and a numpy replay of the MATLAB arithmetic against torch |

`data/` (73 MB teacher set), `runs/` and `release/` are build outputs and are git-ignored.
The release asset is `release/pepc-diiid-cotsim-677-mlp.zip`.

## Inputs (order of `normalization/input_names`)

chi, dna [m^2/s]; core_fueling, gamma_b_in, gamma_b_out, puff_D2 [1/s]; p_b_in, p_b_out, ptot [W].
log10 is applied to core_fueling, gamma_b_in, gamma_b_out, puff_D2 (`x_log`), then
`(x - x_mean)/x_std`. Training box: `input_ranges` of the GNN bundle.

## Outputs (295, physical units, `10^(y*y_std + y_mean)`)

- `te_fsa, ti_fsa, ne_fsa, ni_fsa`: flux-surface averages on the 18 closed rows (core -> separatrix)
- `te_omp, ti_omp, ne_omp, ni_omp`: outer-midplane values on all 36 rows (core -> wall)
- `q_outer, q_inner`: |q_pol| on the 36 plate cells of each target (sorted by iy)
- `q_peak_outer, q_peak_inner, Te_strike_outer, ne_strike_outer, Te_strike_inner, ne_strike_inner, P_rad`

ni is n_D+ (the model's only charged species). rho_c is not baked in: `solstice_mesh/dpsi`
(psi - psi_sep per row) lets the reader interpolate to any rho_c, with the pinned
normalisation of `boundary.py` or `psi_sep - psi_axis` from the equilibrium.

## MATLAB

```matlab
nn = read_NN_solstice('solstice_cotsim_mlp.h5');
y  = nn.predict([0.3 1e20 0.3 9.312e20 2.691e21 68.47 1.304e7 1.299e7 1.623e21]);   % input_names order, or a struct
r  = nn.result(y, 0.95);    % same content as examples/cotsim/cotsim_boundary_demo.json, without 2D fields
plot_NN_solstice(r);
```

The h5 walk is the same as the original COTSIM `read_NN` (layer count from the
`layer_names` attribute, `model_weights/<l>/<l>/vars/{kernel,bias}`), so that class also
loads the weights; only its normalization handling is model-specific.

## Accuracy (median relative error, 68 SOLPS validation cases held out of training)

| Quantity | MLP vs SOLPS | GNN vs SOLPS |
|---|---|---|
| Te, Ti, ne, ni flux-surface averages | 0.8 to 1.2 % | 0.7 to 1.1 % |
| OMP profiles | 0.9 % | 0.9 to 1.0 % |
| q_peak outer / inner | 3.0 / 3.9 % | 3.4 / 5.0 % |
| Te strike outer / inner | 5.5 / 4.9 % | 5.1 / 5.5 % |
| ne strike outer / inner | 5.4 / 4.4 % | 5.7 / 4.7 % |
| P_rad | 1.6 % | 1.7 % |

Model: 9 -> 512 -> 512 -> 512 -> 295, ReLU, 0.68 M parameters, 2.8 MB h5. Forward pass
~0.1 ms per case on CPU (numpy). h5 replay vs torch: max relative difference 4e-6 (float32).

## Reproduce

```bash
module load pytorch/2.13.0; source $SCRATCH/venvs/solstice/bin/activate
python coupler/cotsim/mlp/make_targets.py --out coupler/cotsim/mlp/data/targets.npz      # ~3 min on 1 A100
python coupler/cotsim/mlp/train_mlp.py --data coupler/cotsim/mlp/data/targets.npz --out coupler/cotsim/mlp/runs/h512_l3   # ~1 min
python coupler/cotsim/mlp/package_release.py coupler/cotsim/mlp/runs/h512_l3/best.pt          # h5, check, folder + zip
```

Size sweep (same data, seed 0): 256x3 (0.21 M), 512x3 (0.68 M, shipped), 512x4, 1024x3 all
land within a few tenths of a percent of the GNN; 512x3 was best on the target quantities.
