# Checkpoint bundle spec (v0)

A released model is a directory (or zip) that is sufficient to
reconstruct the network and predict — no training code, no training data.

```
<name>/                        e.g. diiid-lmode-sources-gnn-v1/
  bundle.json                  manifest, see below
  weights.safetensors          model state dict
  normalization.npz            per-variable stats (arrays: per-cell mean/std, input scalers)
  mesh.nc                      canonical mesh (fixed-geometry models ship their grid)
  model_card.md                human-readable card (intended use, metrics, caveats)
```

## bundle.json

```json
{
  "bundle_version": "0.1",
  "name": "diiid-lmode-sources-gnn-v1",
  "task": "sources",                  // "state" | "sources"
  "model": {
    "class": "gnn_encproc",           // registry name, not a python path
    "config": { ... }                 // kwargs to reconstruct the network
  },
  "variables": {
    "inputs":  [{"name": "te", "units": "eV"}, ...],
    "outputs": [{"name": "sp", "units": "..."}, ...]
  },
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
