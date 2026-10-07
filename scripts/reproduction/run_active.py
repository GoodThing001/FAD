"""Retrospective active-learning closed loop for FAD GBSA (汇报 adaptation).

Simulates the expensive-GBSA oracle by hiding labels: from a small initial
labeled set, each round selects a batch via the acquisition policy, reveals the
true GBSA, and retrains the fused surrogate (sequence embedding + physics). A
parallel random-selection run with the same budget gives the control.

Learning curves (Spearman on a fixed holdout) let us check whether active
selection beats random at the same label budget, and whether the surrogate
improves as labels accumulate.

Run on the server GPU env:
  /home/hzeng/miniconda3/envs/ZH/bin/python -m scripts.reproduction.run_active \
      --smoke --n-ensemble 10 --epochs-head 10
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
from .active import select_batch
from .data import load_fad, tokenize_batch
from .encoder import SequenceEncoder
from .fusion import build_handcrafted, FusedHead
from .metrics import spearman
from .pretrain import pretrain


def train_fused_heads(emb, phy, y, device, n, epochs, lr, seed):
    """Train n FusedHeads on cached embeddings + physics; returns heads and y stats."""
    emb = torch.tensor(emb, dtype=torch.float32, device=device)
    phy = torch.tensor(phy, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    mu_y, sd_y = yt.mean().item(), yt.std().item() + 1e-8
    yz = (yt - mu_y) / sd_y

    heads = [FusedHead(d_model=emb.shape[1], d_phy=phy.shape[1]).to(device)
             for _ in range(n)]
    gen = torch.Generator().manual_seed(seed)
    n_data = len(emb)
    for i, head in enumerate(heads):
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for ep in range(epochs):
            head.train()
            perm = torch.randperm(n_data, generator=gen)
            for j in range(0, n_data, 64):
                idx = perm[j : j + 64]
                opt.zero_grad()
                pred = head(emb[idx], phy[idx]).squeeze(-1)
                F.mse_loss(pred, yz[idx]).backward()
                opt.step()
    return heads, mu_y, sd_y


def predict_fused(heads, emb, phy, device):
    emb = torch.tensor(emb, dtype=torch.float32, device=device)
    phy = torch.tensor(phy, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = torch.stack([h(emb, phy).squeeze(-1) for h in heads], dim=0)
    return preds.mean(0).cpu().numpy(), preds.std(0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/proxy2000_v2/fad_proxy2000_v2_full.csv")
    p.add_argument("--target", default="gbsa")
    p.add_argument("--n-ensemble", type=int, default=20)
    p.add_argument("--epochs-head", type=int, default=30)
    p.add_argument("--epochs-pretrain", type=int, default=20)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--n-init", type=int, default=400)
    p.add_argument("--n-holdout", type=int, default=400)
    p.add_argument("--n-batch", type=int, default=100)
    p.add_argument("--n-rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="runs/active/results.json")
    args = p.parse_args()

    if args.smoke:
        args.n_ensemble = min(args.n_ensemble, 10)
        args.epochs_head = min(args.epochs_head, 5)
        args.epochs_pretrain = min(args.epochs_pretrain, 3)
        args.n_init, args.n_holdout, args.n_batch, args.n_rounds = 200, 200, 50, 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    seqs, y = load_fad(args.data, args.target)
    n = len(seqs)
    phy = build_handcrafted(seqs)  # 303d

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    hold = perm[: args.n_holdout]
    rest = perm[args.n_holdout:]
    init = rest[: args.n_init]
    pool0 = rest[args.n_init:]
    print(f"n={n}  holdout={len(hold)}  init={len(init)}  pool={len(pool0)}")

    # Physics standardization is fit on the NON-HOLDOUT rows only (the
    # acquisition/training pool); holdout rows are transformed with the same
    # scaler. No holdout statistics enter the model (fixes the full-data
    # standardization leakage).
    phy_rest = phy[rest]
    phy_sd = phy_rest.std(axis=0)
    phy_sd[phy_sd == 0] = 1.0
    phy_std = (phy - phy_rest.mean(axis=0)) / phy_sd

    # Tokenize + pretrain encoder on the non-holdout sequences only
    # (transductive full-data pretraining is avoided: the holdout must be
    # unseen to the encoder too).
    ids, mask, _ = tokenize_batch(seqs)
    ids, mask = ids.to(device), mask.to(device)
    encoder = SequenceEncoder(tok.VOCAB_SIZE).to(device)
    t0 = time.time()
    print("== pretrain encoder (unsupervised, non-holdout sequences only) ==")
    pretrain(encoder, [seqs[i] for i in rest], device, args.epochs_pretrain,
             64, 8e-4, 0.98, args.seed)
    encoder.eval()
    with torch.no_grad():
        emb_all = encoder(ids, mask).cpu().numpy()
    print(f"embeddings cached ({emb_all.shape}) in {time.time()-t0:.0f}s")

    # y standardization is handled inside train_fused_heads per round.

    def run_loop(strategy, seed_off):
        labeled = init.copy()
        pool = pool0.copy()
        curve = []
        for rnd in range(args.n_rounds):
            heads, mu_y, sd_y = train_fused_heads(
                emb_all[labeled], phy_std[labeled], y[labeled],
                device, args.n_ensemble, args.epochs_head, args.lr_head, args.seed + seed_off)
            mu_h, _ = predict_fused(heads, emb_all[hold], phy_std[hold], device)
            sp = spearman(y[hold], mu_h * sd_y + mu_y)
            curve.append(sp)

            mu_p, sig_p = predict_fused(heads, emb_all[pool], phy_std[pool], device)
            if strategy == "active":
                sel = select_batch(mu_p, sig_p, phy_std[pool], phy_std[labeled],
                                   n_batch=args.n_batch, seed=args.seed + rnd)
            else:
                sel = np.random.RandomState(args.seed + rnd + seed_off).choice(
                    len(pool), size=min(args.n_batch, len(pool)), replace=False).tolist()
            labeled = np.concatenate([labeled, pool[sel]])
            pool = np.delete(pool, sel)
            print(f"  {strategy} round {rnd+1}/{args.n_rounds}: spearman={sp:.4f}  labeled={len(labeled)}")
        return curve

    print("== active learning loop ==")
    curve_al = run_loop("active", seed_off=1000)
    print("== random control loop ==")
    curve_rand = run_loop("random", seed_off=2000)

    out = {"curve_active": curve_al, "curve_random": curve_rand,
           "n_init": int(args.n_init), "n_batch": int(args.n_batch),
           "n_ensemble": args.n_ensemble, "epochs_head": args.epochs_head}
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nactive : {['%.4f' % v for v in curve_al]}")
    print(f"random : {['%.4f' % v for v in curve_rand]}")
    print(f"done -> {outp}")


if __name__ == "__main__":
    main()
