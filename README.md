# SOLSTICE

**S**crape-**O**ff **L**ayer **S**urrogate **T**raining, **I**nference &
**C**oupling **E**cosystem for neural-network surrogates of SOLPS-ITER edge
plasmas.

SOLSTICE provides trained models and tools for predicting 2D plasma states
from operating-point parameters and coupling those predictions to reduced
edge-physics workflows such as COTSIM.

## Install

    pip install "solstice-fusion[models] @ git+https://github.com/abdoudiaw/solstice.git"

## Quick start

    from solstice import hub

    model = hub.load("pepc-diiid-state-v2")
    params = {
        "ptot": 6e6,
        "chi": 0.7,
        "core_fueling": 3e20,
        "puff_D2": 1e21,
        "dna": 0.5,
    }
    result = model.predict_batch(params)
    fields = {name: values[0] for name, values in result["mean"].items()}

## Examples

See [examples/README.md](examples/README.md) for runnable state, source, and
REACT COTSIM workflows.

## Released models

| Model | Description |
|---|---|
| pepc-diiid-state-v2 | DIII-D state model |
| pepc-diiid-cotsim-284-state | COTSIM-focused DIII-D state model |
| pepc-diiid-state-v1 | Previous DIII-D state model |
| pepc-jet-state-v1 | JET-shaped-grid state model |
| pepc-diiid-sources-v1 | Optional plasma-to-source model |

Each bundle contains its model definition, normalization, mesh, provenance,
and model card.

## Development

Training data is not distributed with this repository. SOLPS converters and
training tools are under src/solstice/. SOLPS file parsing follows
[SOLPS-routines](https://github.com/ORNL-Fusion/SOLPS-routines).

## Acknowledgments

This work was supported in part by the U.S. Department of Energy (DOE)
Fusion Innovation Research Engine (FIRE) Collaborative Advanced Profile
Prediction for Fusion Pilot Plant Design (APP-FPP) via an MIT subcontract
under Award No. DE-SC0025853, and by the DOE Office of Science, Office of
Fusion Energy Sciences, through the project *Enabling Tokamak Pulse
Simulation by Machine Learning of Core-Pedestal-Boundary Physics*.

## License

Code: Apache-2.0. Released model weights: CC BY 4.0; see the model cards.
