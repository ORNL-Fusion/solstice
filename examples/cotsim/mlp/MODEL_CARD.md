# pepc-diiid-cotsim-677-mlp

SOLSTICE surrogate for the REACT COTSIM coupling: a small fully connected ReLU network that
maps the DIII-D L-mode operating point and the SOLPS core-boundary fluxes directly to the
quantities COTSIM uses. It is distilled from the SOLSTICE graph network
`pepc-diiid-cotsim-677-state` and the 677 SOLPS-ITER cases it was trained on, and runs in
MATLAB (or numpy) from one HDF5 file with no Python or GPU dependency.

## Contents

| File | Purpose |
|---|---|
| `solstice_cotsim_mlp.h5` | weights, normalization, input/output names and units, mesh constants |
| `read_NN_solstice.m` | MATLAB reader: `predict`, `result` (boundary values at rho_c, targets, profiles), `at_rho`, `unpack` |
| `plot_NN_solstice.m` | MATLAB plots: Te, Ti, ne, ni profiles vs psi_N with rho_c; target heat-flux profiles |
| `cotsim_mlp_demo.m` | MATLAB demo of the notebook operating point: print, plot, JSON, check, timing |
| `solstice_cotsim_mlp.py` | numpy + h5py reader with the same API (`CotsimMLP`) |
| `reference_demo.json` | Python outputs at the demo point; the MATLAB demo compares against it |
| `metrics.json` | validation metrics |

## Quick start (MATLAB)

```matlab
nn = read_NN_solstice('solstice_cotsim_mlp.h5');
p  = struct('ptot',1.299e7,'puff_D2',1.623e21,'core_fueling',1e20,'dna',0.3,'chi',0.3, ...
            'gamma_b_in',9.312e20,'gamma_b_out',2.691e21,'p_b_in',68.47,'p_b_out',1.304e7);
y  = nn.predict(p);              % 1 x 295; an N x 9 matrix gives N x 295
r  = nn.result(y, 0.95);         % r.boundary_conditions.te.fsa, r.targets.outer_target.q_peak_W_m2, ...
plot_NN_solstice(r);
```

Or run `cotsim_mlp_demo` in this folder.

## Inputs (9)

| Name | Meaning | Unit |
|---|---|---|
| `chi` | anomalous heat diffusivity (electrons = ions) | m^2/s |
| `core_fueling` | core fueling rate | atoms/s |
| `dna` | anomalous particle diffusivity | m^2/s |
| `gamma_b_in` | net neutral influx through the core boundary | 1/s |
| `gamma_b_out` | D+ plasma outflux through the core boundary | 1/s |
| `p_b_in` | incoming neutral energy through the core boundary | W |
| `p_b_out` | plasma power outflux through the core boundary (fheyreg + fhiyreg) | W |
| `ptot` | power crossing the core boundary, pe + pi | W |
| `puff_D2` | D2 gas puff | atoms/s |

Definitions and sign conventions: SOLSTICE `examples/cotsim/README.md`. Training range: see
`input_range` in `metrics.json`; predictions outside it are extrapolations.

## Outputs (295)

- Te, Ti [eV], ne, ni [m^-3] flux-surface averages on the 18 closed mesh rows, and outer-midplane values on all 36 rows
- |q_pol| [W/m^2] on the 36 plate cells of the inner and outer targets
- peak heat flux, strike-point Te and ne on both targets, total radiated power [W]

ni is the D+ density. `result(y, rho_c)` interpolates to rho_c in normalised poloidal flux.
Without `psi_sep - psi_axis` the innermost mesh row is pinned to rho_c, as in the SOLSTICE
notebook; pass it as the third argument when the equilibrium is known. The 2D fields of the
GNN are not predicted.

## Accuracy

Median relative error on the 68 SOLPS validation cases held out of training (same split as the GNN):

| Quantity | MLP | GNN |
|---|---|---|
| Te / Ti / ne flux-surface profiles | 1.2 / 0.9 / 0.8 % | 1.1 / 1.0 / 0.7 % |
| Outer-midplane profiles | 0.9 % | 0.9 to 1.0 % |
| Peak heat flux, outer / inner | 3.0 / 3.9 % | 3.4 / 5.0 % |
| Strike-point Te, outer / inner | 5.5 / 4.9 % | 5.1 / 5.5 % |
| Strike-point ne, outer / inner | 5.4 / 4.4 % | 5.7 / 4.7 % |
| Radiated power | 1.6 % | 1.7 % |

At the notebook operating point the MLP agrees with the GNN within 0.6 % on Te, Ti, ne, ni at
rho_c, 1 % on peak heat flux, 1.2 % on P_rad, and 5 % on inner strike-point Te.

## Model

9 -> 512 -> 512 -> 512 -> 295, ReLU, 0.68 M parameters (float32). Inputs: log10 of
core_fueling, puff_D2, gamma_b_in, gamma_b_out, then standardized. Outputs: standardized
log10. Trained on 30k GNN predictions (Latin hypercube over the training box plus jitter around
the training cases) and the 609 SOLPS training cases (weight 5), smooth-L1 loss, AdamW, 1500
epochs, best epoch on a teacher holdout.

## License and citation

Weights CC BY 4.0, code Apache-2.0. Cite A. Diaw et al., arXiv:2607.21407 (2026), and
J. D. Lore et al., Nucl. Fusion 63, 046015 (2023) for the DIII-D SOLPS-ITER configuration.
This work is part of the REACT project within the GENESIS project led by Lehigh University.
