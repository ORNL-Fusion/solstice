# Examples

- `quickstart.ipynb` — load the released plasma-state model and predict
  ([open in Colab](https://colab.research.google.com/github/ORNL-Fusion/solstice/blob/main/examples/quickstart.ipynb))

Using released models (self-contained bundles, raw physical inputs):

- `predict_state.py` — control parameters -> plasma background
  (`python examples/predict_state.py ~/.cache/solstice/pepc-diiid-state-v2`, the bundle
  directory `hub.load('pepc-diiid-state-v2')` downloads)
- `predict_sources.py` — plasma state -> EIRENE source terms (interface only: no
  sources model is released yet)
- `cotsim/boundary.py` — COTSIM coupling: loads a state bundle, predicts the 2D plasma
  state, and reports Te, Ti, ne, ni at `rho_c`, profiles versus `psi_N`, target heat loads,
  `P_rad`, and the SOLPS/EIRENE boundary inputs `Gamma_b_in`, `Gamma_b_out`, `P_b_in`,
  and `P_b_out`.
- `cotsim/boundary_demo.ipynb` — presentation-ready notebook showing model loading,
  inference, physical 2D state plots, flux-surface/OMP profiles, and COTSIM outputs.
- `cotsim/README.md` — detailed definitions, SOLPS/EIRENE sources, sign conventions,
  and the SOLSTICE-to-COTSIM workflow.

The four boundary flux and power values are scalar inputs at the core boundary. The
state network predicts the 2D plasma fields; it does not reconstruct these integrated
SOLPS/EIRENE diagnostics from volumetric source fields.

Warm-starting SOLPS-ITER (console script, see `solstice.data.converters.to_solps`):

- `solstice-b2fstati --init nn --reference-run RUN` — prediction -> `b2fstati` in a staged copy of RUN
- `solstice-b2fstati --init flat --reference-run RUN` — uniform cold start, same staging
- `solstice-b2fstati --check RUN` — is RUN's `b2fstati` flat or a pre-converged state?
