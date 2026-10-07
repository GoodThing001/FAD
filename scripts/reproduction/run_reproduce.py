"""End-to-end driver reproducing gkag145 (NAR 2026) on FAD aptamer GBSA.

Pipeline: (1) pretrain the 3-mer transformer encoder with MLM + triplet loss,
(2) freeze the encoder and train a deep ensemble of MLP regression heads on
GBSA, (3) evaluate (Spearman / MAE / RMSE), (4) optionally run one round of
Kriging-Believer batch selection over a held-out pool.

Default (--outer-seeds 1) reproduces the historical documented run
(spearman=0.123, seed=0 permutation split). With --outer-seeds N>1 the driver
uses the project split protocol (train_test_split test_size=0.2,
random_state=seed) and reports mean ± std across seeds — use that for any
quantitative claim.

Run on the server GPU env:
  /home/hzeng/miniconda3/envs/ZH/bin/python -m scripts.reproduction.run_reproduce \
      --smoke --epochs-pretrain 3 --epochs-head 5 --n-ensemble 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import tokenizer as tok
from .data import load_fad, tokenize_batch
from .encoder import SequenceEncoder
from .metrics import spearman
from .pretrain import pretrain, batched_indices
from .surrogate import DeepEnsemble, RegressionHead
from . import bo


def train_ensemble(encoder, seqs, y, device, n, epochs, batch_size, lr, seed):
    """Train n heads independently on cached embeddings (encoder frozen)."""
    ids, mask, _ = tokenize_batch(seqs)
    ids, mask = ids.to(device), mask.to(device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    mu_y, sd_y = yt.mean().item(), yt.std().item()
    yz = (yt - mu_y) / (sd_y + 1e-8)

    encoder.eval()
    with torch.no_grad():
        emb = encoder(ids, mask)  # (N, d_model), computed once

    heads = [RegressionHead(d_model=encoder.d_model).to(device) for _ in range(n)]
    for i, head in enumerate(heads):
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(seed + i)
        for ep in range(epochs):
            head.train()
            for idxs in batched_indices(len(emb), batch_size, gen):
                opt.zero_grad()
                pred = head(emb[idxs]).squeeze(-1)
                F.mse_loss(pred, yz[idxs]).backward()
                opt.step()
        print(f"  head {i+1}/{n} trained")
    return heads, mu_y, sd_y


def evaluate(encoder, heads, seqs, y, device, mu_y, sd_y, batch_size=256):
    ids, mask, _ = tokenize_batch(seqs)
    ids, mask = ids.to(device), mask.to(device)
    encoder.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            x, m = ids[i : i + batch_size], mask[i : i + batch_size]
            emb = encoder(x, m)
            stacked = torch.stack([h(emb) for h in heads], dim=0)  # (n,b,1)
            preds.append(stacked.mean(dim=0).squeeze(-1).cpu())
    mu = torch.cat(preds).numpy() * sd_y + mu_y
    return spearman(y, mu), float(np.mean(np.abs(y - mu))), float(np.sqrt(np.mean((y - mu) ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/proxy2000_v2/fad_proxy2000_v2_full.csv")
    p.add_argument("--target", default="gbsa")
    p.add_argument("--epochs-pretrain", type=int, default=30)
    p.add_argument("--epochs-head", type=int, default=50)
    p.add_argument("--n-ensemble", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-pretrain", type=float, default=8e-4)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outer-seeds", type=int, default=1,
                   help=">1: multi-seed eval with the project split protocol "
                        "(train_test_split, random_state=seed); reports mean±std")
    p.add_argument("--max-seqs", type=int, default=0, help="0 = all (subset for smoke)")
    p.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    p.add_argument("--bo", action="store_true", help="run one Kriging-Believer batch round")
    p.add_argument("--out", default="runs/reproduce/results.json")
    args = p.parse_args()

    if args.smoke:
        args.epochs_pretrain = min(args.epochs_pretrain, 3)
        args.epochs_head = min(args.epochs_head, 3)
        args.n_ensemble = min(args.n_ensemble, 5)
        args.max_seqs = args.max_seqs or 400
        args.batch_size = min(args.batch_size, 32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}")

    seqs, y = load_fad(args.data, args.target)
    if args.max_seqs:
        seqs, y = seqs[: args.max_seqs], y[: args.max_seqs]
    n = len(seqs)

    def run_one(tr_idx, te_idx, seed):
        tr_seqs = [seqs[i] for i in tr_idx]
        te_seqs = [seqs[i] for i in te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        n_train = len(tr_idx)
        print(f"data: {n} seqs, train {n_train}, test {n - n_train}")

        t0 = time.time()
        encoder = SequenceEncoder(tok.VOCAB_SIZE).to(device)
        print("== pretrain (MLM + triplet) ==")
        pretrain(encoder, tr_seqs, device, args.epochs_pretrain, args.batch_size,
                 args.lr_pretrain, args.gamma, seed)

        print(f"== train ensemble ({args.n_ensemble} heads) ==")
        heads, mu_y, sd_y = train_ensemble(encoder, tr_seqs, y_tr, device, args.n_ensemble,
                                           args.epochs_head, args.batch_size, args.lr_head, seed)

        print("== evaluate ==")
        sp, mae, rmse = evaluate(encoder, heads, te_seqs, y_te, device, mu_y, sd_y)
        print(f"TEST  spearman={sp:.4f}  mae={mae:.4f}  rmse={rmse:.4f}  ({time.time()-t0:.0f}s)")
        return {"spearman": sp, "mae": mae, "rmse": rmse}, encoder, heads, \
            tr_seqs, te_seqs, y_tr, y_te, mu_y, sd_y

    if args.outer_seeds > 1:
        from sklearn.model_selection import train_test_split
        per_seed = []
        for seed in range(1, args.outer_seeds + 1):
            tr_idx, te_idx = train_test_split(range(n), test_size=0.2, random_state=seed)
            res, _, _, _, _, _, _, _, _ = run_one(tr_idx, te_idx, seed)
            per_seed.append(res)
            print(f"  seed={seed}/{args.outer_seeds}  spearman={res['spearman']:.4f}")
        sps = np.array([r["spearman"] for r in per_seed])
        result = {"spearman_mean": float(sps.mean()), "spearman_std": float(sps.std()),
                  "per_seed": per_seed, "n": n, "n_train": int(n * 0.8),
                  "epochs_pretrain": args.epochs_pretrain,
                  "epochs_head": args.epochs_head, "n_ensemble": args.n_ensemble,
                  "device": device}
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"done -> {out}")
        return

    # Historical single-seed path (seed=0 permutation split) — reproduces the
    # documented sequence-only deep-ensemble baseline (0.123).
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    res, encoder, heads, tr_seqs, te_seqs, y_tr, y_te, mu_y, sd_y = run_one(tr_idx, te_idx, args.seed)
    sp, mae, rmse = res["spearman"], res["mae"], res["rmse"]

    result = {"spearman": sp, "mae": mae, "rmse": rmse, "n": n,
              "n_train": n_train, "epochs_pretrain": args.epochs_pretrain,
              "epochs_head": args.epochs_head, "n_ensemble": args.n_ensemble,
              "device": device}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if args.bo:
        # Demonstrate one KB batch over the test pool (labels hidden from the model).
        pool_ids, pool_mask, _ = tokenize_batch(te_seqs)
        pool_ids, pool_mask = pool_ids.to(device), pool_mask.to(device)
        tr_ids, tr_mask, tr_lab = tokenize_batch(tr_seqs, (y_tr - mu_y) / (sd_y + 1e-8))
        tr_ids, tr_mask = tr_ids.to(device), tr_mask.to(device)
        tr_lab = tr_lab.to(device)

        ensemble = DeepEnsemble(encoder, args.n_ensemble).to(device)
        ensemble.heads = torch.nn.ModuleList(heads)

        def refit(cur_ids, cur_mask, cur_lab):
            for h in ensemble.heads:
                opt = torch.optim.Adam(h.parameters(), lr=args.lr_head)
                for ep in range(2):  # cheap refit
                    h.train()
                    opt.zero_grad()
                    pred = h(encoder(cur_ids, cur_mask)).squeeze(-1)
                    F.mse_loss(pred, cur_lab).backward()
                    opt.step()

        chosen = bo.kriging_believer(ensemble, refit, tr_ids, tr_mask, tr_lab,
                                     pool_ids, pool_mask, n_batch=5, maximize=False)
        print("KB selected pool indices (minimize GBSA):", chosen)

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
