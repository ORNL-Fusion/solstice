# Checkpoint bundle spec (bundle_version 0.2)

A released model is a directory (or zip) that is sufficient to
reconstruct the network and predict — no training code, no training data.

```
<name>/                        e.g. pepc-diiid-state-v2/
  bundle.json                  manifest, see below
  member_<k>.safetensors       state dict of ensemble member k (k = 0 .. n_members-1)
  member_<k>.npz               member k: input scalers (x_mean, x_std, x_min, x_max), output
                               stats (y_mean, y_std per cell and field, y_quantiles (cell,
                               field, n_quantiles) for the quantile transform, field_mask),
                               and its latent graph (node_x, assign_index/attr,
                               latent_edges/attr, dec_index/attr) exactly as trained
  mesh.nc                      canonical mesh (fixed-geometry models ship their grid)
  model_card.md                human-readable card (intended use, metrics, caveats)
  uq_calibration.json          optional: calibration of the member spread (is-ok flag)
  novelty.npz                  optional: standardized training inputs + kNN reference
                               distances for the input-novelty guard
```

Sources bundles keep the 0.1 layout (`weights.safetensors`, `normalization.npz`).

## bundle.json

```json
{
  "bundle_version": "0.2",
  "name": "pepc-diiid-state-v2",
  "task": "state",                    // "state" | "sources"
  "model": {
    "class": "gnn",                   // registry name, not a python path
    "config": { ... }                 // kwargs to reconstruct the network (same for all members)
  },
  "n_members": 5,                     // deep ensemble: prediction = member mean
  "members": [{"seed": 0, "code_git": "...", "epoch": 310, "source": "runs/..."}, ...],
  "graph": {"n_latent": 512, "k_nn": 6, "node_features": "extended",   // state GNNs
            "k_enc": 1, "k_dec": 1, "multimesh": []},
  "variables": {
    "inputs":  ["chi", "core_fueling", ...],      // engineered model inputs
    "outputs": ["te", "ti", ...],
    "log10_outputs": {"te": true, ...},
    "aux_targets": {"q_target": {"inner_target": "q_inner_target",    // channels defined on
                                 "outer_target": "q_outer_target"}}   // target cells only
  },
  "output_transform": "quantile",     // "standard" | "quantile" (per-cell, SOLPS-NN style)
  "input_transform": {"log_inputs": [...], "merged": {...}},
  "input_ranges": {"ptot": [lo, hi], ...},         // training box, transformed units
  "uncertainty": {"calibration": "uq_calibration.json", "novelty": "novelty.npz"},  // or null
  "provenance": {
    "machine": "diiid", "regime": "lmode",
    "dataset": "diiid-lmode-d1", "schema_version": "0",
    "code_git": "<hash>", "metrics": { ... },
    "parent": "diiid-lmode-sources-gnn-v1",  // warm-start lineage; null if from scratch
    "cost": {                                 // solstice.inference.profile output
      "n_params": 0, "weights_mb": 0.0,
      "device": "cuda", "batch": 1,
      "latency_ms_median": 0.0, "latency_ms_p95": 0.0,
      "throughput_per_s": 0.0
    }
  },
  "license": "CC-BY-4.0"
}
```

## Rules

- `model.class` resolves through `solstice.models.registry` only.
- Loading = `solstice.inference.load_checkpoint(path)`; it must succeed
  in a clean environment with just `solstice-fusion` installed.
- Names: `{machine}-{regime}-{task}-{arch}[-mini]-v{N}`; the training
  dataset version goes in provenance and, once several generations
  exist, into the name (e.g. `-d2`).
- Bundles are immutable; a change of weights is a new `v{N}`.
- **Bundles are valid training inits, not just inference artifacts.**
  `load_checkpoint(path).core` is an ordinary torch module: fine-tune it
  on new data (transfer learning, e.g. DIII-D -> KSTAR warm start),
  continue training on an extended dataset, or use the loaded model as
  a frozen teacher for distillation into a smaller/`-mini` student.
  Every model trained from a bundle records it in `provenance.parent`,
  so lineage chains (pretrain -> fine-tune -> distill) stay auditable.
  Optimizer/scheduler state is deliberately NOT in bundles — mid-run
  resume uses ordinary training checkpoints, which are never released.
- Format changes bump `bundle_version` with a migration note here
  (anemoi-style checkpoint migrations).

## Migration notes

- **0.1 -> 0.2 (2026-10-02).** State bundles may hold several members, each
  with its own normalization and latent graph (every training run fits its
  quantile maps on its own split and seeds its k-means latent mesh with its
  own seed, so neither can be shared). `weights.safetensors` /
  `normalization.npz` became `member_<k>.*`; the latent graph is shipped
  instead of rebuilt, so a bundle no longer depends on the k-means
  implementation staying bit-identical. New manifest keys: `n_members`,
  `members`, `graph`, `output_transform`, `variables.aux_targets`,
  `uncertainty`. `StatePredictor.predict()` keeps its signature (the mean for
  ensembles; aux channels come back as per-plate profiles);
  `predict_batch()` adds the member spread and, with a calibration, the
  error bar / is-ok flags of `solstice.uncertainty`. 0.1 state bundles still
  load (single member, graph rebuilt with seed 0 as before).
