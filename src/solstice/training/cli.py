# =========================================================================================
# (C) (or copyright) 2026. UT-Battelle, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract DE-AC05-00OR22725 with
# UT-Battelle, LLC, which manages Oak Ridge National Laboratory (ORNL) for the U.S.
# Department of Energy (DOE). The U.S. Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
# to reproduce, prepare derivative works, distribute copies to the public, perform
# publicly and display publicly, and to permit others to do so. The DOE will provide
# public access to these results in accordance with the DOE Public Access Plan
# (http://energy.gov/downloads/doe-public-access-plan).
# =========================================================================================
# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: Apache-2.0
"""`solstice-train`: train / evaluate a state model from a YAML config.

    solstice-train --config configs/training/solpsnn_state.yaml \
        [--set train.epochs=2 --set out_dir=runs/x ...] [--eval-only]

Multi-GPU (one process per GPU, data parallel):
    torchrun --nproc_per_node=4 -m solstice.training.cli --config ...

Device: CUDA when available, else CPU (`--device mps` to try Apple GPUs).
Re-running with the same out_dir resumes from last.pt. `--eval-only`
loads out_dir/best.pt and only evaluates.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

from solstice.models import build_model
from solstice.training.data import (InputNorm, MeshGraph, OutputNorm, engineer_inputs,
                                    select_cases, train_val_split)
from solstice.training.evaluate import evaluate, format_metrics, predict
from solstice.training.trainer import Trainer


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return _expand_env(cfg)


def _expand_env(node):
    """Expand $VAR / ${VAR} in every string of the config (store paths)."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return os.path.expandvars(node)
    return node


def _fields_spec(spec: dict) -> dict[str, bool]:
    out = {}
    for name, how in spec.items():
        if isinstance(how, bool):
            out[name] = how
        elif str(how).lower() in ("log", "log10"):
            out[name] = True
        elif str(how).lower() in ("linear", "lin"):
            out[name] = False
        else:
            raise ValueError(f"field transform for {name} must be log or linear, got {how!r}")
    return out


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def head_groups(names: list[str], heads: dict) -> list[list[int]]:
    """model.heads {group: [field patterns]} -> channel index lists, one per decoder head;
    a field goes to the first group that matches it, unmatched fields to a last "rest" head."""
    groups, taken = [], set()
    for grp, pats in heads.items():
        pats = [pats] if isinstance(pats, str) else list(pats)
        g = [j for j, n in enumerate(names) if j not in taken and any(fnmatch.fnmatch(n, q) for q in pats)]
        if not g:
            raise ValueError(f"model.heads group {grp!r} matches no field of {names}")
        taken.update(g)
        groups.append(g)
    rest = [j for j in range(len(names)) if j not in taken]
    return groups + ([rest] if rest else [])


def _setup_distributed(device_arg: str | None):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device_arg:
        device = device_arg
    elif torch.cuda.is_available():
        device = f"cuda:{local}"
    else:
        device = "cpu"
    if world > 1:
        backend = "nccl" if device.startswith("cuda") else "gloo"
        torch.distributed.init_process_group(backend=backend)
        if device.startswith("cuda"):
            torch.cuda.set_device(local)
    return device, rank, world


def run(cfg: dict, eval_only: bool = False, device_arg: str | None = None) -> dict:
    device, rank, world = _setup_distributed(device_arg)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed + rank)
    out_dir = Path(cfg["out_dir"])
    dcfg, mcfg, gcfg, tcfg = cfg["data"], cfg["model"], cfg.get("graph", {}), cfg.get("train", {})
    fields = _fields_spec(dcfg["fields"])
    t0 = time.time()

    ds = xr.open_dataset(dcfg["train_store"])
    idx = select_cases(ds, dcfg.get("qc", "qc_pass"), dcfg.get("exclude_regimes"), dcfg.get("max_te"))
    if dcfg.get("limit"):
        idx = idx[: int(dcfg["limit"])]
    # data.split_seed: fix the validation holdout across seeds (default: follows `seed`), so a
    # seed ensemble shares one set of cases no member trained on (its calibration set)
    itr, ival = train_val_split(idx, float(dcfg.get("val_fraction", 0.1)), int(dcfg.get("split_seed", seed)))
    raw = engineer_inputs(ds, verbose=rank == 0)
    in_norm = InputNorm(names=list(dcfg.get("inputs") or sorted(raw)),
                        log_inputs=list(dcfg.get("log_inputs", []))).fit(raw, itr)
    out_norm = OutputNorm(fields, dcfg.get("aux_targets"),
                          transform_kind=str(dcfg.get("output_transform", "standard")),
                          n_quantiles=int(dcfg.get("n_quantiles", 100))).fit(ds, itr)
    Xn, Y = in_norm.transform(raw), out_norm.transform(ds)
    graph = MeshGraph(ds, n_latent=int(gcfg.get("n_latent", 256)), k_nn=int(gcfg.get("k_nn", 6)),
                      seed=seed, target_weight=float(tcfg.get("target_weight", 1.0)),
                      node_features=str(gcfg.get("node_features", "default")),
                      k_enc=int(gcfg.get("k_enc", 1)), k_dec=gcfg.get("k_dec"),
                      multimesh=list(gcfg.get("multimesh") or []))
    if rank == 0:
        print(f"store {dcfg['train_store']}: {ds.sizes['case']} cases, {len(idx)} selected, "
              f"{len(itr)} train / {len(ival)} val; {ds.sizes['cell']} cells; "
              f"inputs {in_norm.names}; outputs {out_norm.names}; device {device}; "
              f"world {world} ({time.time() - t0:.0f} s)", flush=True)

    model_cfg = {"node_features": graph.node_features.shape[1], "param_dim": len(in_norm.names),
                 "out_features": len(out_norm.names), "n_cells": graph.n_cells, **mcfg.get("config", {})}
    if mcfg.get("heads"):
        model_cfg["head_index"] = head_groups(out_norm.names, mcfg["heads"])
    model = build_model(mcfg.get("class", "gnn"), model_cfg)
    export_meta = {"model_class": mcfg.get("class", "gnn"), "config": model_cfg,
                   **in_norm.to_dict(), **out_norm.to_dict(),
                   "n_latent": graph.n_latent, "k_nn": int(gcfg.get("k_nn", 6)),
                   "node_features": graph.node_feature_kind, **graph.graph_extra,
                   "train_store": str(dcfg["train_store"]), "code_git": _git_hash(),
                   "train_config": cfg}
    if rank == 0:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"{mcfg.get('class', 'gnn')}: {n_par:,} parameters", flush=True)

    result = {}
    if eval_only:
        ck = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
    else:
        trainer = Trainer(model, graph, Xn[itr], Y[itr], Xn[ival], Y[ival], tcfg, out_dir,
                          device, export_meta, rank=rank, world_size=world, seed=seed,
                          field_mask=out_norm.mask, field_names=out_norm.names)
        result["train"] = trainer.train()
        model = trainer.model
    if rank != 0:
        return result
    if not (out_dir / "best.pt").exists():
        return result
    if "train" in result and result["train"]["epochs_done"] < trainer.epochs:
        # stopped at train.max_hours: no metrics.json, so a resubmission resumes from last.pt
        return result

    batch = int(tcfg.get("batch", 32))
    metrics = {"val": evaluate(ds, ival, out_norm, predict(model, graph, Xn[ival], device, batch),
                               Y[ival])}
    print(f"\n== validation ({len(ival)} cases of the training store)")
    print(format_metrics(metrics["val"]))
    if dcfg.get("test_store"):
        dt = xr.open_dataset(dcfg["test_store"])
        it = select_cases(dt, dcfg.get("qc", "qc_pass"), dcfg.get("exclude_regimes"), dcfg.get("max_te"))
        Xt, Yt = in_norm.transform(engineer_inputs(dt)), out_norm.transform(dt)
        metrics["test"] = evaluate(dt, it, out_norm, predict(model, graph, Xt[it], device, batch), Yt[it])
        print(f"\n== test store {dcfg['test_store']} ({len(it)} cases)")
        print(format_metrics(metrics["test"]))
    metrics.update(result)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config entry, dotted keys (e.g. train.epochs=5)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: cuda if available)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    run(cfg, eval_only=args.eval_only, device_arg=args.device)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
