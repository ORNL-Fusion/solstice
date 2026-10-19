# jet_heads5_h1024_ft_te_ens20 — the trained model (released as `pepc-jet-state-v1`)

20 independently seeded SOLSTICE GNN state models, each trained in two stages, used as a
deep ensemble (mean = prediction, seed spread = uncertainty). Copied 2026-10-02 from
`runs/ensemble/heads5_h1024/seed{k}_ft_te` (originals untouched; stage-1 parent
weights are only there, `seed{k}/best.pt`; the run directories came from the retired
solstice-exp worktree).

```
seed{0..19}/                 member k: best.pt (94 MB), metrics.json, log.jsonl (local only, not in git)
parents/seed{k}/             stage-1 metrics + curve of member k (local only)
ens20_mean.json              deep-ensemble metrics on the test store (scripts/nersc/ensemble_eval.py)
ens5_mean.json               same for seeds 0-4 only; parents/ens5_mean_noft.json = seeds 0-4 before fine-tune
recipe/ensemble.sbatch       the exact Slurm script that trained them (array 0-19)
recipe/solpsnn_state.yaml    base config it extends
evaluate.sh                  recompute ens20_mean.json (GPU)
```

## Recipe

Base config `configs/training/solpsnn_state.yaml`, overridden as in `recipe/ensemble.sbatch`:

- Stage 1 (300 epochs, max 1.8 h, one A100): quantile output transform, MAE loss,
  cell_embed 32, params_to_cells, extended node features, decoder hidden 2048, five
  grouped decoder heads of width 1024 (plasma: te ti ne; deut: na_D*; nitro: na_N*;
  flow: ua_* q_pol; target: q_target gamma_target prad), lr 1e-3, batch 32, target_weight 5.
- Stage 2 (60 epochs, lr 3e-4): Te fine-tune of the plasma head only
  (`train.trainable=["decoders.0.*"]`, `field_weights={te: 5}`) from the stage-1 `best.pt`.

Architecture from `best.pt`: gnn encode-process-decode, hidden 128, 8 processor layers,
dropout 0.1, 256 latent nodes, k_nn 6, 6 node features, 8 parameters, 27 output channels
(22 state variables + ne, prad, q_pol + aux heads q_target, gamma_target), 5,276,827
parameters. Inputs: btor, chi, core_fueling, dna, ptot, puff_D, puff_N, rmajor (log:
puff_D, puff_N, core_fueling). Training code git `f09baf1` / `ec57fd2` (exp/te-gap, merged
into main). Fine-tune best epochs 39-60 of 60.

## Metrics (test store, 1465 cases, mean of 20; from `ens20_mean.json`)

| field | R2 (norm) | median abs / rel error, all cells | outer target | OMP sep |
|---|---|---|---|---|
| te | 0.914 | 2.38 eV / 4.2 % | 0.76 eV / 8.9 % | 2.5 % |
| ti | 0.898 | 4.9 eV / 4.9 % | 0.96 eV / 9.5 % | 3.2 % |
| ne | 0.916 | 8.5e17 / 7.6 % | 2.1e18 / 15.5 % | 4.3 % |
| na_D1 | 0.904 | 6.3e17 / 8.4 % | 1.4e18 / 16.7 % | 5.0 % |
| na_N0 | 0.898 | 3.2e12 / 9.9 % | 5.7e15 / 23.8 % | 4.4 % |
| ua_D1 | 0.805 | 1.2e3 / 21.6 % | 1.3e3 / 9.5 % | 37.6 % |
| q_pol | 0.769 | 1.0e5 / 20.7 % | 9.3e4 / 14.6 % | 19.0 % |
| prad | 0.803 | 319 / 16.5 % | 8.6e4 / 18.8 % | 10.0 % |
| q_target (aux) | 0.865 | 2.6e4 / 10.4 % | 5.6e4 / 10.2 % | - |
| gamma_target (aux) | 0.844 | 7.1e20 / 12.3 % | 1.2e21 / 13.1 % | - |

Te by regime: attached 2.3 %, detached 3.7 %, sheath-limited 4.4 %, cold core 24.4 %.
Single seeds: Te all cells 5.22 +- 0.16 %. Peak outer-target heat flux from the q_target
head: 8.2 % median error; integrated outer-target D+ flux: 7.6 %.

Caveats: cold-core target quantities unusable (q_peak 86 %, Gamma 100 %);
ua_* 20-45 %; ensemble spread over-confident (59 % of cells within 1 sigma vs 68 % for Gaussian).
Full per-field numbers (ot_sep, by_regime, rmse) are in the json.
