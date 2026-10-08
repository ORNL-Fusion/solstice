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
"""State-model trainer: replicated fixed graph, target-weighted MSE in
normalized units, cosine schedule with warmup, bf16 autocast on CUDA,
periodic checkpoints with resume (Slurm time limits), and optional
data-parallel training across GPUs (launch with torchrun; each rank
trains on its shard of every epoch's permutation and gradients are
averaged by DistributedDataParallel).

Writes to out_dir:
  last.pt      resumable training state (model, optimizer, schedule, epoch)
  best.pt      export dict of the best-validation model, the format
               solstice.hub.create_state_bundle consumes
  log.jsonl    one record per epoch
"""

from __future__ import annotations

import fnmatch
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from solstice.training.data import MeshGraph


class Trainer:
    def __init__(self, model: nn.Module, graph: MeshGraph,
                 Xn_train: np.ndarray, Y_train: np.ndarray,
                 Xn_val: np.ndarray, Y_val: np.ndarray,
                 cfg: dict, out_dir: str, device, export_meta: dict,
                 rank: int = 0, world_size: int = 1, seed: int = 0,
                 field_mask: np.ndarray | None = None, field_names: list[str] | None = None):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.device = torch.device(device)
        self.rank, self.world = rank, world_size
        self.seed = seed
        self.graph = graph
        self.export_meta = export_meta
        self.n_cells = graph.n_cells
        self.model = model.to(self.device)
        self._init_and_freeze(cfg)
        self.ddp = world_size > 1
        if self.ddp:
            ids = [self.device.index] if self.device.type == "cuda" else None
            self.net = nn.parallel.DistributedDataParallel(self.model, device_ids=ids)
        else:
            self.net = self.model
        self.X_tr = torch.tensor(Xn_train, device=self.device)
        self.Y_tr = torch.tensor(Y_train, device=self.device)
        self.X_val = torch.tensor(Xn_val, device=self.device)
        self.Y_val = torch.tensor(Y_val, device=self.device)
        self.n_train, self.n_val = len(Xn_train), len(Xn_val)
        n_out = Y_train.shape[-1]
        mask = np.ones((self.n_cells, n_out), np.float32) if field_mask is None else field_mask
        mask = mask / mask.mean(0, keepdims=True)
        self.field_names = list(field_names or [f"f{j}" for j in range(n_out)])
        fw = field_weights(self.field_names, cfg.get("field_weights"))
        self.field_mask = torch.tensor(mask * fw[None, :], device=self.device)
        # kendall: learned per-field log scale s_f (Kendall, Gal & Cipolla 2018), training loss
        # mean_f exp(-s_f) L_f + s_f (mae: Laplace) or 0.5 exp(-2 s_f) L_f + s_f (mse); the
        # validation loss (best.pt selection) stays the fixed-weight one.
        self.weighting = str(cfg.get("loss_weighting", "fixed")).lower()
        if self.weighting not in ("fixed", "kendall"):
            raise ValueError(f"unknown loss_weighting {self.weighting!r}")
        self.log_scale = nn.Parameter(torch.zeros(n_out, device=self.device)) \
            if self.weighting == "kendall" else None

        self.epochs = int(cfg.get("epochs", 300))
        self.batch = int(cfg.get("batch", 32))
        warm = int(cfg.get("warmup", 5))
        groups = [{"params": [p for p in self.model.parameters() if p.requires_grad]}]
        if self.log_scale is not None:
            groups.append({"params": [self.log_scale], "weight_decay": 0.0})
        self.opt = torch.optim.AdamW(groups, lr=float(cfg.get("lr", 1e-3)),
                                     weight_decay=float(cfg.get("weight_decay", 1e-4)))

        def lr_lambda(ep):
            if ep < warm:
                return (ep + 1) / warm
            t = (ep - warm) / max(1, self.epochs - warm)
            return float(cfg.get("lr_min_frac", 0.01)) + (1 - float(cfg.get("lr_min_frac", 0.01))) \
                * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.amp = bool(cfg.get("amp", True)) and self.device.type == "cuda"
        self.clip = float(cfg.get("grad_clip", 1.0))
        self.loss_kind = str(cfg.get("loss", "mse")).lower()        # mse | mae | huber
        self.huber_delta = float(cfg.get("huber_delta", 1.0))
        self.epoch = 0
        self.best = float("inf")
        self.history: list[dict] = []

    def _init_and_freeze(self, cfg: dict):
        """train.init_from: start from a best.pt (same architecture); train.trainable:
        parameter-name patterns (fnmatch) to train, everything else frozen."""
        if cfg.get("init_from"):
            ck = torch.load(cfg["init_from"], map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["state_dict"])
            if self.rank == 0:
                print(f"initialised from {cfg['init_from']} (epoch {ck.get('epoch')})", flush=True)
        pats = cfg.get("trainable")
        if pats:
            pats = [pats] if isinstance(pats, str) else list(pats)
            n_on = 0
            for name, p in self.model.named_parameters():
                p.requires_grad_(any(fnmatch.fnmatch(name, q) for q in pats))
                n_on += p.numel() if p.requires_grad else 0
            if n_on == 0:
                raise ValueError(f"train.trainable {pats} matches no parameter")
            if self.rank == 0:
                print(f"trainable: {n_on:,} parameters matching {pats}", flush=True)

    # ---- batching -------------------------------------------------------------------
    def _forward(self, X, Y, idx: np.ndarray, train: bool):
        B = len(idx)
        g = self.graph.batch(B, self.device)
        ii = torch.as_tensor(idx, device=self.device)
        kw = {k: v for k, v in g.items() if k != "cell_weight"}
        kw["params"] = X[ii]
        kw["params_latent"] = X[ii].repeat_interleave(self.graph.n_latent, dim=0)
        yb = Y[ii].reshape(B * self.n_cells, -1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = (self.net if train else self.model)(**kw)
        w = g["cell_weight"][:, None] * self.field_mask.repeat(B, 1)   # (B*cells, fields)
        err = pred.float() - yb
        if self.loss_kind == "mae":
            per = err.abs()
        elif self.loss_kind == "huber":
            per = torch.nn.functional.huber_loss(pred.float(), yb, reduction="none", delta=self.huber_delta)
        else:
            per = err ** 2
        if not (train and self.log_scale is not None):
            return (per * w).mean()
        lf = (per * w).mean(0)                                          # (fields,)
        s = self.log_scale
        if self.loss_kind == "mse":
            return (0.5 * torch.exp(-2 * s) * lf + s).mean()
        return (torch.exp(-s) * lf + s).mean()

    def _epoch_perm(self, epoch: int) -> np.ndarray:
        perm = np.random.default_rng(self.seed + epoch).permutation(self.n_train)
        n = (len(perm) // self.world) * self.world
        return perm[:n][self.rank::self.world]

    def _shard(self, n: int) -> np.ndarray:
        return np.arange(n)[self.rank::self.world]

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        tot, cnt = 0.0, 0
        sh = self._shard(self.n_val)
        for i in range(0, len(sh), self.batch):
            idx = sh[i:i + self.batch]
            tot += self._forward(self.X_val, self.Y_val, idx, train=False).item() * len(idx)
            cnt += len(idx)
        if self.ddp:
            t = torch.tensor([tot, cnt], dtype=torch.float64, device=self.device)
            torch.distributed.all_reduce(t)
            tot, cnt = t.tolist()
        return tot / max(cnt, 1)

    # ---- checkpointing ----------------------------------------------------------------
    def _export_dict(self, val_loss: float) -> dict:
        return {"state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                **self.export_meta, "epoch": self.epoch, "val_loss": val_loss,
                "epochs": self.epochs, "seed": self.seed}

    def save_last(self):
        if self.rank != 0:
            return
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "log_scale": None if self.log_scale is None else self.log_scale.detach().cpu(),
                    "sched": self.sched.state_dict(), "epoch": self.epoch, "best": self.best,
                    "history": self.history}, self.out_dir / "last.pt")

    def resume(self) -> bool:
        p = self.out_dir / "last.pt"
        if not p.exists():
            return False
        ck = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        if self.log_scale is not None and ck.get("log_scale") is not None:
            with torch.no_grad():
                self.log_scale.copy_(ck["log_scale"])
        self.epoch, self.best, self.history = ck["epoch"], ck["best"], ck.get("history", [])
        if self.rank == 0:
            print(f"resumed from {p} at epoch {self.epoch} (best val {self.best:.5f})", flush=True)
        return True

    # ---- loop -----------------------------------------------------------------------
    def train(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.resume()
        ckpt_every = int(self.cfg.get("ckpt_every", 10))
        log_every = int(self.cfg.get("log_every", 10))
        max_hours = self.cfg.get("max_hours")
        t_start = time.time()
        log = open(self.out_dir / "log.jsonl", "a") if self.rank == 0 else None
        while self.epoch < self.epochs:
            t0 = time.time()
            self.model.train()
            perm = self._epoch_perm(self.epoch)
            tot, cnt = 0.0, 0
            for i in range(0, len(perm), self.batch):
                idx = perm[i:i + self.batch]
                self.opt.zero_grad(set_to_none=True)
                loss = self._forward(self.X_tr, self.Y_tr, idx, train=True)
                loss.backward()
                if self.ddp and self.log_scale is not None:      # not inside DDP: average by hand
                    torch.distributed.all_reduce(self.log_scale.grad)
                    self.log_scale.grad /= self.world
                if self.clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.opt.step()
                tot += loss.item() * len(idx)
                cnt += len(idx)
            self.sched.step()
            val = self.validate()
            self.epoch += 1
            rec = {"epoch": self.epoch, "train_loss": tot / max(cnt, 1), "val_loss": val,
                   "lr": self.opt.param_groups[0]["lr"], "epoch_s": time.time() - t0}
            if self.log_scale is not None:
                rec["log_scale"] = dict(zip(self.field_names, self.log_scale.detach().cpu().tolist()))
            self.history.append(rec)
            if val < self.best:
                self.best = val
                if self.rank == 0:
                    torch.save(self._export_dict(val), self.out_dir / "best.pt")
            if self.rank == 0:
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if self.epoch % log_every == 0 or self.epoch == 1:
                    print(f"ep{self.epoch}: train {rec['train_loss']:.4f} val {val:.4f} "
                          f"lr {rec['lr']:.2e} ({rec['epoch_s']:.1f} s)", flush=True)
            if self.epoch % ckpt_every == 0 or self.epoch == self.epochs:
                self.save_last()
            if max_hours and (time.time() - t_start) / 3600 > float(max_hours):
                self.save_last()
                if self.rank == 0:
                    print(f"time limit reached at epoch {self.epoch}; resume with the same out_dir",
                          flush=True)
                break
        if log:
            log.close()
        best_path = self.out_dir / "best.pt"
        if self.ddp:
            torch.distributed.barrier()
        if best_path.exists():
            ck = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["state_dict"])
        return {"best_val": self.best, "epochs_done": self.epoch,
                "wall_s": time.time() - t_start}


def field_weights(names: list[str], spec: dict | None) -> np.ndarray:
    """Per-channel loss weights from {pattern: weight} (fnmatch on field names, later
    patterns win), renormalised to mean 1 so the loss scale and lr stay comparable."""
    w = np.ones(len(names), np.float32)
    for pat, val in (spec or {}).items():
        hit = [j for j, n in enumerate(names) if fnmatch.fnmatch(n, pat)]
        if not hit:
            raise ValueError(f"field_weights pattern {pat!r} matches no field of {names}")
        w[hit] = float(val)
    return w / w.mean()
