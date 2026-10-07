"""Proper two-stage paper method + feature fusion.

Stage 1 (paper's key): pretrain the transformer (MLM + triplet) on a large
unlabeled corpus:
  - corpus-mode=mutate (default): random mutants of our own aptamer sequences
    (in-domain corpus, 30,000 by default — matches the 9月 report). NOTE: the
    corpus is generated from ALL 2000 sequences, so the encoder is transductive
    w.r.t. future test folds. This is acceptable for the closed representation
    study (its conclusion is negative), but the caveat is printed at runtime and
    must be stated when citing numbers.
  - corpus-mode=riboswitch: the paper's 61,906 riboswitch NGS sequences
    (data/ngs_sequences.txt, server-only; 10k subset).

Stage 2: fuse the frozen transformer [CLS] embedding (128d) with NUPACK physics
features (the 7 useful blocks, ~192d) through a deep ensemble -> GBSA.

Runs in ZH env (torch).

Usage (server, ZH env):
  python -m scripts.reproduction.run_paper_fusion \
    --pretrain-epochs 20 --n-ensemble 20 --outer-seeds 10
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata
from sklearn.model_selection import train_test_split

from . import tokenizer as tok
from .data import load_fad, tokenize_batch
from .encoder import SequenceEncoder
from .metrics import spearman
from .pretrain import pretrain
from .surrogate import RegressionHead

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]

# 7 NUPACK "useful" blocks (positive marginal delta), ~192d
USEFUL_BLOCKS = {
    "contact_pca": [r"nupack_contact_"],
    "cluster_pair": [r"nupack_inter_cluster_\d+_\d+$", r"nupack_intra_cluster_\d+$",
                     r"nupack_mut_long_range$"],
    "energy_land": [r"nupack_n_very_stable_pairs$", r"nupack_n_moderate_pairs$",
                    r"nupack_stable_pair_ratio$", r"nupack_eig_ipr$", r"nupack_eig_n_eff_modes$"],
    "pos_entropy": [r"nupack_pos\d+_entropy$"],
    "motif_global": [r"nupack_motif_global_", r"nupack_motif_apt_"],
    "pos_pair_prob": [r"nupack_pos\d+_pair_prob$", r"nupack_pos\d+_likely_paired$",
                      r"nupack_c\d+_mean_pair_prob$", r"nupack_mean_pair_prob$",
                      r"nupack_std_pair_prob$", r"nupack_frac_paired_gt_0_5$"],
    "motif_positional": [r"nupack_motif_pos\d+_is_"],
}


def load_corpus(path, max_len=300, max_seqs=0, rng_seed=0):
    """Load + filter the pretraining corpus (T->U, N->random base, cap length)."""
    rng = np.random.RandomState(rng_seed)
    seqs = []
    with open(path) as f:
        for line in f:
            s = line.strip().upper().replace("T", "U")
            if not s or len(s) > max_len:
                continue
            s = "".join(c if c in "ACGU" else "ACGU"[rng.randint(4)] for c in s)
            seqs.append(s)
            if max_seqs and len(seqs) >= max_seqs:
                break
    return seqs


def generate_variant_corpus(seqs, n_variants, rng_seed=0, mutate_prob=0.5):
    """Generate an in-domain pretraining corpus by randomly mutating the 13
    variable positions of our own aptamer sequences.

    This is an in-domain pretraining hypothesis for the FAD task. The paper's
    riboswitch corpus belongs to a different task; that alone does not prove
    that it teaches an inferior representation. This historical generator
    uses whichever parents are passed (the old driver uses all 2000), so it
    must not be used unchanged for an inductive outer-fold claim. The controlled
    supplemental driver compares corpora and excludes all known test sequences.
    """
    rng = np.random.RandomState(rng_seed)
    idx = [p - 1 for p in MUT_POS]
    variants = []
    for _ in range(n_variants):
        s = list(seqs[rng.randint(len(seqs))])
        for i in idx:
            if rng.rand() < mutate_prob:
                s[i] = "ACGU"[rng.randint(4)]
        variants.append("".join(s))
    return variants


def load_nupack_useful(csv_path, seqs):
    """Load the 7 useful NUPACK blocks aligned to seqs."""
    df = pd.read_csv(csv_path)
    df["Sequence"] = df["Sequence"].astype(str)
    cols = [c for c in df.columns if c.startswith("nupack_")]
    sel = []
    for pats in USEFUL_BLOCKS.values():
        for i, c in enumerate(cols):
            if any(re.search(p, c) for p in pats):
                sel.append(i)
    sel = sorted(set(sel))
    cols = [cols[i] for i in sel]
    df = df.set_index("Sequence").loc[[str(s) for s in seqs]]
    return df[cols].to_numpy(dtype=np.float32), cols


class FusedNet(nn.Module):
    """Deep-ensemble member: [CLS] + physics -> paper's 9x256 MLP head -> scalar."""

    def __init__(self, d_emb, d_phy, hidden=256):
        super().__init__()
        self.head = RegressionHead(d_model=d_emb + d_phy, hidden=hidden, n_layers=9, out_dim=1)

    def forward(self, emb, phy):
        return self.head(torch.cat([emb, phy], dim=1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/ngs_sequences.txt")
    ap.add_argument("--corpus-mode", default="mutate", choices=["mutate", "riboswitch"],
                    help="mutate = random-mutant in-domain corpus; riboswitch = paper's NGS")
    ap.add_argument("--max-pretrain-seqs", type=int, default=30000)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--n-ensemble", type=int, default=20)
    ap.add_argument("--epochs-head", type=int, default=60)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--outer-seeds", type=int, default=10)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/paper_fusion/results.json")
    ap.add_argument("--log-dir", default="runs/paper_fusion")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # ── Stage 1: pretrain on large corpus ──
    data_dir = PROJECT_ROOT / args.data_dir
    if args.corpus_mode == "mutate":
        # in-domain corpus: random mutants of our own aptamer sequences
        seqs_all, _y = load_fad(data_dir / "fad_proxy2000_v2_full.csv", "gbsa")
        corpus = generate_variant_corpus(seqs_all, args.max_pretrain_seqs)
        print(f"pretrain corpus: {len(corpus)} in-domain random mutants")
        print("[caveat] corpus generated from ALL 2000 sequences (transductive w.r.t. "
              "the per-seed test folds); acceptable for this closed representation "
              "study, but must be stated when citing numbers.")
    else:
        corpus_path = PROJECT_ROOT / args.corpus
        if not corpus_path.is_file():
            raise FileNotFoundError(
                f"riboswitch corpus missing: {corpus_path}. The paper's NGS subset "
                "lives on the server as data/ngs_sequences.txt and is not committed "
                "to the mainline repo (raw external data). Use --corpus-mode mutate "
                "for the in-domain corpus, or restore the file on the server."
            )
        corpus = load_corpus(corpus_path, max_seqs=args.max_pretrain_seqs)
        print(f"pretrain corpus: {len(corpus)} riboswitch seqs")
    encoder = SequenceEncoder(tok.VOCAB_SIZE).to(device)
    print("== Stage 1: pretrain (MLM + triplet) on corpus ==")
    pretrain(encoder, corpus, device, args.pretrain_epochs, 64, 8e-4, 0.98, 0)
    encoder.eval()

    # ── Stage 2: fuse [CLS] + NUPACK physics -> deep ensemble ──
    seqs, y = load_fad(data_dir / "fad_proxy2000_v2_full.csv", "gbsa")
    X_phy, phy_cols = load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)
    print(f"physics: {len(phy_cols)}d (7 useful NUPACK blocks)")

    ids, mask, _ = tokenize_batch(seqs)
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        emb_all = encoder(ids, mask).cpu().numpy()  # (N, 128)
    print(f"[CLS] embeddings: {emb_all.shape}")

    all_res = []
    for seed in range(1, args.outer_seeds + 1):
        tr_idx, te_idx = train_test_split(range(len(seqs)), test_size=0.2, random_state=seed)
        y_tr_raw, y_te_raw = y[tr_idx], y[te_idx]
        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

        mu_p, sd_p = X_phy[tr_idx].mean(0), X_phy[tr_idx].std(0)
        sd_p[sd_p == 0] = 1.0
        phy_tr = (X_phy[tr_idx] - mu_p) / sd_p
        phy_te = (X_phy[te_idx] - mu_p) / sd_p

        emb_tr = torch.tensor(emb_all[tr_idx], dtype=torch.float32, device=device)
        emb_te = torch.tensor(emb_all[te_idx], dtype=torch.float32, device=device)
        phy_tr = torch.tensor(phy_tr, dtype=torch.float32, device=device)
        phy_te = torch.tensor(phy_te, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        mu_y, sd_y = y_tr_t.mean().item(), y_tr_t.std().item() + 1e-8
        yz = (y_tr_t - mu_y) / sd_y

        heads = [FusedNet(emb_all.shape[1], X_phy.shape[1]).to(device) for _ in range(args.n_ensemble)]
        n_tr = len(emb_tr)
        for i, h in enumerate(heads):
            opt = torch.optim.Adam(h.parameters(), lr=args.lr_head, weight_decay=1e-5)
            gen = torch.Generator().manual_seed(seed + i)
            for ep in range(args.epochs_head):
                h.train()
                perm = torch.randperm(n_tr, generator=gen)
                for j in range(0, n_tr, 64):
                    idx = perm[j:j + 64]
                    opt.zero_grad()
                    F.mse_loss(h(emb_tr[idx], phy_tr[idx]), yz[idx]).backward()
                    opt.step()

        with torch.no_grad():
            preds = torch.stack([h(emb_te, phy_te) for h in heads], dim=0)
        mu_te = preds.mean(0).cpu().numpy() * sd_y + mu_y
        sp = spearman(y_te_raw, mu_te)
        all_res.append({"seed": seed, "spearman": round(sp, 6)})
        print(f"  seed={seed:2d}  spearman={sp:.4f}")

        out_dir = PROJECT_ROOT / args.log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(all_res, open(out_dir / "partial.json", "w"), indent=2)

    sps = np.array([r["spearman"] for r in all_res])
    print(f"\n{'='*60}")
    print(f"paper fusion (pretrained transformer + NUPACK): {sps.mean():.4f} +/- {sps.std():.4f}")
    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"spearman_mean": float(sps.mean()), "spearman_std": float(sps.std()),
               "results": all_res, "n_physics": len(phy_cols)},
              open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
