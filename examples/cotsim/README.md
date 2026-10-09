# COTSIM coupling: SOLPS and SOLSTICE quantities

This example documents the information passed between SOLPS-ITER, the
SOLSTICE state surrogate, and COTSIM at the SOLPS core boundary `rho_c`.
The reference boundary is the SOLPS/EIRENE core-side surface (`input.dat`,
standard surface 1).

This work is part of the REACT project within the GENESIS project led by
Lehigh University.

## Quantities COTSIM needs

| Quantity | Meaning | Source |
|---|---|---|
| `Te` | Electron temperature at `rho_c` [eV] | SOLSTICE state prediction |
| `Ti` | Ion temperature at `rho_c` [eV] | SOLSTICE state prediction |
| `ne` | Electron density at `rho_c` [m^-3] | SOLSTICE state prediction |
| `ni` | Ion density at `rho_c` [m^-3] | Sum of charged-ion state fields, or `ne` fallback |
| `Gamma_b_out` | Plasma particle outflux through the core boundary [1/s] | `b2tallies.nc`, `fnayreg` for D+ |
| `P_b_out` | Plasma power outflux through the core boundary [W] | `b2tallies.nc`, `fheyreg + fhiyreg` (B2's `fhtyreg` adds neutral energy terms and is stored separately as `p_b_out_total`) |
| `Gamma_b_in` | Net neutral influx through the core boundary [1/s] | EIRENE `fort.44`, WLD surface 1 |
| `P_b_in` | Incoming neutral energy through the core boundary [W] | EIRENE `ewlda + ewldm`, WLD surface 1 |

The neutral particle convention includes all simulated neutral species:

```text
Gamma_b_in = (wldna - wldra - wldpa)
            + (wldnm - wldrm - wldpm)
```

For EIRENE WLD arrays, stratum `0` is already the total. Later strata are
components and must not be summed again.

`wldpeb` is retained as a separate diagnostic for reflected/emitted neutral
energy. It is not subtracted from `P_b_in`.

## Sign convention

The stored values retain the native SOLPS/EIRENE orientation:

- positive `Gamma_b_in`: neutral particles entering the SOLPS domain from the core-side boundary;
- positive `Gamma_b_out`: plasma particles leaving the core toward SOLPS;
- positive `P_b_in`: incoming neutral energy;
- positive `P_b_out`: plasma power leaving the core.

If COTSIM uses the opposite normal direction, reverse the sign at the
coupling interface rather than changing the database values.

## What SOLSTICE predicts

The state network predicts the 2D plasma background from operating-point
inputs. The COTSIM example then computes flux-surface averages at `rho_c`,
outer-midplane profiles, target heat-load diagnostics, and total radiated
power. The boundary flux and power quantities are SOLPS/EIRENE diagnostics
provided as scalar inputs; they are not reconstructed from the volumetric
source model (`sp`, `sne`, `qe`, `qi`, or `sm`).

## Running the example

```bash
  python examples/cotsim/boundary.py <state-bundle> \
  --rho-c 0.95 --ptot 6e6 --puff-D2 1e21 --core-fueling 3e20 \
  --dna 0.5 --chi 0.7 --gamma-b-in 8.18e20 --gamma-b-out 2.28e21 \
  --p-b-in 28.9 --p-b-out 1.01e7 --out cotsim_boundary.json
```

The JSON output contains boundary conditions, profiles, target diagnostics,
model-quality flags, and a `cotsim_boundary_inputs` block.

## Dataset status

The COTSIM store contains 677 converged DIII-D cases with finite direct
`b2tallies.nc` and EIRENE WLD diagnostics: 286 archived runs plus 391 runs
whose tallies were regenerated with a one-step SOLPS-ITER restart. The
original 761-case state dataset remains unchanged. The released model is
`pepc-diiid-cotsim-677-state` (recipe `configs/training/diiid_cotsim_677_state.yaml`,
68 validation cases held out of the same store; Te median relative error 1.2 % all cells,
5.1 % at the outer target, peak outer-target heat flux 3.4 %). It replaces the 284-case
proof-of-concept model, which did not cover the operating range of the regenerated runs.
