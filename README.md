# SOLSTICE

**S**crape-**O**ff **L**ayer **S**urrogate **T**raining, **I**nference & **C**oupling **E**cosystem — neural-network surrogates for SOLPS-ITER edge plasmas.

Two model tasks, one framework:

- **State models** (`params -> plasma`): plasma backgrounds
  (Te, Ti, ne, ua, heat flux, radiated power) from control parameters.
- **Source models** (`plasma -> sources`): neutral source terms
  (Sp, Sne, Qe, Qi, Sm) from the local plasma state.

Released weights are self-describing bundles: download and predict
anywhere — no training code, no training data.

## Install

```bash
pip install "solstice-fusion[models] @ git+https://github.com/ORNL-Fusion/solstice.git"
```

## Quick start

```python
from solstice import hub

state = hub.load("pepc-diiid-state-v1")
fields = state.predict({"ptot": 6e6, "chi": 0.7, "core_fueling": 3e20,
                        "puff_D2": 1e21, "dna": 0.5})

sources = hub.load("pepc-diiid-sources-v1")
plasma = {k: fields[k] for k in sources.manifest["variables"]["plasma_features"]}
terms = sources.predict(plasma, params={"ptot": 6e6, "chi": 0.7,
                                        "core_fueling": 3e20,
                                        "puff_D2": 1e21, "dna": 0.5})
```

Weights download from GitHub Releases on first use and are cached in
`~/.cache/solstice`. Runnable examples: `examples/predict_state.py`,
`examples/predict_sources.py`.

Inputs are the models' true degrees of freedom (the training data has
pe = pi and hci = hce, so the models see `ptot = pe + pi` and a single
`chi`). Requests outside the training parameter box raise a warning —
treat those predictions as extrapolations.

## Released models

Names follow `pepc-{machine}-{task}-v{N}` (PEPC: ORNL Power Exhaust and
Particle Control group). Architecture: `gnn` — latent-mesh
encode-process-decode with FiLM conditioning. Metrics, provenance, and
caveats ship inside each bundle (`bundle.json`, `model_card.md`).

| model | task | outputs |
|---|---|---|
| `pepc-diiid-state-v1` | params -> plasma | te, ti, ne, na_D1, ua_D1, q_pol, prad |
| `pepc-diiid-sources-v1` | plasma -> sources | sp, sne, qe, qi, sm |

## Repository layout

```
src/solstice/
  data/        dataset schema + converters (SOLPS output -> canonical)
  graphs/      mesh -> graph construction
  models/      registry + architectures
  inference/   checkpoint loading, profiling
  hub/         released-model bundles: create / load / predict
  physics/     scalar QoIs from predicted fields
docs/specs/    data schema and checkpoint bundle — the contracts
examples/      predict scripts for released models
```

## Design principles

1. **Unstructured-first mesh.** The canonical mesh is cells/faces/vertices,
   matching GOAT wide-grid SOLPS output. Structured grids are the special
   case where cells carry optional `(ix, iy)` labels. See
   `docs/specs/data_schema.md`.
2. **SOLPS file reading follows
   [SOLPS-routines](https://github.com/ORNL-Fusion/SOLPS-routines)** (Jeremy Lore, ORNL).
3. **Self-describing checkpoints.** Weights + config + normalization stats
   + mesh + provenance in one bundle. See `docs/specs/checkpoint_spec.md`.
4. **Graphs are derived.** Datasets store the mesh; loaders build edges
   (face adjacency) at load time, identically for structured and wide grids.

## Data

Training data comes from SOLPS-ITER simulations and is **not** distributed
with this repository. Converters in `solstice.data.converters` build
canonical cases from SOLPS run directories. The DIII-D configuration
follows Lore et al., and the released DIII-D models derive from it —
please cite:

```bibtex
@article{Lore2023,
  author  = {J. D. Lore and S. {De Pascuale} and P. Laiu and B. Russo and
             J.-S. Park and J. M. Park and S. L. Brunton and J. N. Kutz and
             A. A. Kaptanoglu},
  title   = {Time-dependent SOLPS-ITER simulations of the tokamak plasma
             boundary for model predictive control using SINDy},
  journal = {Nuclear Fusion},
  volume  = {63},
  number  = {046015},
  pages   = {1--12},
  year    = {2023},
  doi     = {10.1088/1741-4326/acbe0e}
}
```

## Citation

See `CITATION.cff`.

## License

Code: Apache-2.0. Released model weights: CC BY 4.0 (see model cards).
