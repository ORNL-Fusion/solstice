# Examples

- `quickstart.ipynb` — load the released models and predict
  ([open in Colab](https://colab.research.google.com/github/ORNL-Fusion/solstice/blob/main/examples/quickstart.ipynb))

Using released models (self-contained bundles, raw physical inputs):

- `predict_state.py` — control parameters -> plasma background
  (`python examples/predict_state.py /path/to/pepc-diiid-state-v1`)
- `predict_sources.py` — plasma state -> EIRENE source terms
  (`python examples/predict_sources.py /path/to/pepc-diiid-sources-v1`)

Warm-starting SOLPS-ITER (console script, see `solstice.data.converters.to_solps`):

- `solstice-b2fstati --init nn --reference-run RUN` — prediction -> `b2fstati` in a staged copy of RUN
- `solstice-b2fstati --init flat --reference-run RUN` — uniform cold start, same staging
- `solstice-b2fstati --check RUN` — is RUN's `b2fstati` flat or a pre-converged state?
