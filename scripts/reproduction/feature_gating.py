"""Feature-level gated fusion (加权 + 反馈 + 优化).

Sequence (303d one-hot: mut/struct/wd) + gated NUPACK blocks -> MLP -> GBSA.

Each of the 18 NUPACK blocks gets a learnable sigmoid gate w_i = sigmoid(s_i).
Gates are optimized via feedback (regression loss) in a scale-sensitive MLP,
so blocks with negative marginal contribution are softly down-weighted instead
of hard-dropped, and blocks with latent signal can keep a small positive weight.

This is the feature-level complement to feature_weighting.py (prediction-level
fusion, whose weights collapsed to uniform). Runs in ZH env (torch).

Usage (server, ZH env):
  python -m scripts.reproduction.feature_gating --outer-seeds 10 --epochs 60
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")

BLOCKS = {
    "pos_pair_prob": [r"nupack_pos\d+_pair_prob$", r"nupack_pos\d+_likely_paired$",
                      r"nupack_c\d+_mean_pair_prob$", r"nupack_mean_pair_prob$",
                      r"nupack_std_pair_prob$", r"nupack_frac_paired_gt_0_5$"],
    "pos_entropy": [r"nupack_pos\d+_entropy$"],
    "win_stats": [r"nupack_pos\d+_win\d+_(mean|std)$"],
    "cluster_pair": [r"nupack_inter_cluster_\d+_\d+$", r"nupack_intra_cluster_\d+$",
                     r"nupack_mut_long_range$"],
    "pair_dist": [r"nupack_pos\d+_pair_dist$"],
    "seg_entropy": [r"nupack_seg_entropy\d+_(mean|std|min|max)$", r"nupack_pos\d+_seg3_entropy$"],
    "ensemble": [r"nupack_ensemble_entropy$", r"nupack_ensemble_entropy_norm$",
                 r"nupack_participation_ratio$", r"nupack_n_pairs_gt_0_1$"],
    "contact_pca": [r"nupack_contact_"],
    "expected_acc": [r"nupack_expected_accuracy_", r"nupack_frac_accurate_",
                     r"nupack_cl\d+_expected_acc$"],
    "mfe_basic": [r"nupack_mfe$", r"nupack_mfe_n_stems$", r"nupack_mfe_stem_(max|mean)_len$",
                  r"nupack_mfe_n_loops$", r"nupack_mfe_loop_max_len$", r"nupack_mfe_n_unpaired$",
                  r"nupack_mfe_frac_paired$", r"nupack_mfe_pos\d+_is_paired$", r"nupack_mfe_c\d+_paired$"],
    "squarna": [r"nupack_sq_"],
    "mfe_detailed": [r"nupack_mfe_n_bulges$", r"nupack_mfe_bulge_(max|mean|total)$",
                     r"nupack_mfe_n_internal_loops$", r"nupack_mfe_iloop_(max|asym_max|asym_mean)$",
                     r"nupack_mfe_n_helices$", r"nupack_mfe_n_hairpins$",
                     r"nupack_mfe_hairpin_(max|mean)$", r"nupack_mfe_c\d+_(bulges|iloops)$"],
    "ufold_blocks": [r"nupack_block_"],
    "dist_moments": [r"nupack_triu_", r"nupack_density_", r"nupack_dist_", r"nupack_frac_above_"],
    "gradient": [r"nupack_pos\d+_grad_", r"nupack_pos\d+_partner_entropy$",
                 r"nupack_pos\d+_best_partner_dist$"],
    "energy_land": [r"nupack_n_very_stable_pairs$", r"nupack_n_moderate_pairs$",
                    r"nupack_stable_pair_ratio$", r"nupack_eig_ipr$", r"nupack_eig_n_eff_modes$"],
    "motif_positional": [r"nupack_motif_pos\d+_is_"],
    "motif_global": [r"nupack_motif_global_", r"nupack_motif_apt_"],
}


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def build_seq(seqs):
    Xm = np.zeros((len(seqs), 52), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if idx < len(s) and s[idx] in BASES:
                Xm[i, j * 4 + BASES.index(s[idx])] = 1.0
    ROLES = {30: "stem", 31: "stem", 32: "stem", 33: "junction",
             89: "loop", 90: "loop", 91: "loop", 92: "junction",
             127: "stem", 128: "stem", 129: "stem", 130: "junction", 131: "loop"}
    PAIRS = {30: 33, 31: 32, 32: 31, 33: 30, 90: 92, 92: 90, 127: 131, 131: 127, 128: 130, 130: 128}
    Xs = np.zeros((len(seqs), 43), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U"); mc = sm = lm = 0
        for j, p in enumerate(MUT_POS):
            idx = p - 1; wt = WT_141[idx]; mb = s[idx]
            is_mut = 1.0 if mb != wt else 0.0; role = ROLES.get(p, "unknown")
            Xs[i, j * 3] = 1.0 if role == "stem" else 0.0
            Xs[i, j * 3 + 1] = 1.0 if role == "loop" else 0.0
            Xs[i, j * 3 + 2] = is_mut * (1.0 if role == "stem" else 0.0)
            mc += is_mut
            if is_mut:
                if role == "stem": sm += 1
                elif role == "loop": lm += 1
        Xs[i, -4] = mc; Xs[i, -3] = sm; Xs[i, -2] = lm; broken = 0
        for p in MUT_POS:
            part = PAIRS.get(p, 0)
            if part > 0 and s[p - 1] != WT_141[p - 1] and s[part - 1] == WT_141[part - 1]:
                broken += 1
        Xs[i, -1] = float(broken)
    all_d = [a + b for a in BASES for b in BASES]
    Xw = np.zeros((len(seqs), 208), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U"); n = len(s)
        for j, p in enumerate(MUT_POS):
            idx = p - 1; lo = max(0, idx - 5); hi = min(n, idx + 6)
            sub = s[lo:hi]
            for t in range(len(sub) - 1):
                d = sub[t:t + 2]
                if d[0] in BASES and d[1] in BASES:
                    Xw[i, j * 16 + all_d.index(d)] += 1
            total = max(1, len(sub) - 1)
            Xw[i, j * 16:j * 16 + 16] /= total
    return np.hstack([Xm, Xs, Xw])


class GatedNet(nn.Module):
    def __init__(self, d_seq, block_dims, hidden=256, gate_init=1.0):
        super().__init__()
        self.n_blocks = len(block_dims)
        self.gate_logits = nn.Parameter(torch.full((self.n_blocks,), gate_init))
        self.in_dim = d_seq + sum(block_dims)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq, blocks):
        gates = torch.sigmoid(self.gate_logits)
        gated = [gates[i] * blocks[i] for i in range(self.n_blocks)]
        x = torch.cat([seq] + gated, dim=1)
        return self.mlp(x).squeeze(-1)

    def gate_values(self):
        return torch.sigmoid(self.gate_logits).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--entropy-reg", type=float, default=0.0)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/feature_gating/results.json")
    ap.add_argument("--log-dir", default="runs/feature_gating")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    data_dir = PROJECT_ROOT / args.data_dir
    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in full_df["Sequence"]]
    y_all = full_df["gbsa"].to_numpy(dtype=np.float32)
    X_seq = build_seq(seqs)

    nup_df = pd.read_csv(data_dir / "nupack_features_full.csv")
    nup_df["Sequence"] = nup_df["Sequence"].astype(str)
    nup_df = nup_df.set_index("Sequence").loc[seqs]
    nup_cols = [c for c in nup_df.columns if c.startswith("nupack_")]
    X_nup = nup_df[nup_cols].to_numpy(dtype=np.float32)

    block_idx = {}
    used = set()
    for bname, pats in BLOCKS.items():
        idx = []
        for i, c in enumerate(nup_cols):
            if i in used:
                continue
            for p in pats:
                if re.search(p, c):
                    idx.append(i); used.add(i); break
        if idx:
            block_idx[bname] = idx
    block_names = list(block_idx.keys())
    print(f"NUPACK {len(nup_cols)}d -> {len(block_names)} blocks")

    all_res = []
    for seed in range(1, args.outer_seeds + 1):
        tr_idx, te_idx = train_test_split(range(len(full_df)), test_size=0.2, random_state=seed)
        Xs_tr, Xs_te = X_seq[tr_idx], X_seq[te_idx]
        y_tr_raw, y_te_raw = y_all[tr_idx], y_all[te_idx]
        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

        # standardize NUPACK features (fit on train)
        mu_n, sd_n = X_nup[tr_idx].mean(0), X_nup[tr_idx].std(0)
        sd_n[sd_n == 0] = 1.0
        blocks_tr = [(X_nup[tr_idx][:, idx] - mu_n[idx]) / sd_n[idx] for idx in block_idx.values()]
        blocks_te = [(X_nup[te_idx][:, idx] - mu_n[idx]) / sd_n[idx] for idx in block_idx.values()]

        # standardize 303d too (fit on train), so one-hot + struct counts are on a
        # comparable scale with the z-scored NUPACK blocks
        mu_s, sd_s = X_seq[tr_idx].mean(0), X_seq[tr_idx].std(0)
        sd_s[sd_s == 0] = 1.0
        Xs_tr = (Xs_tr - mu_s) / sd_s
        Xs_te = (Xs_te - mu_s) / sd_s

        seq_tr = torch.tensor(Xs_tr, dtype=torch.float32, device=device)
        seq_te = torch.tensor(Xs_te, dtype=torch.float32, device=device)
        blocks_tr = [torch.tensor(b, dtype=torch.float32, device=device) for b in blocks_tr]
        blocks_te = [torch.tensor(b, dtype=torch.float32, device=device) for b in blocks_te]
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

        model = GatedNet(X_seq.shape[1], [b.shape[1] for b in blocks_tr], hidden=args.hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        # Deterministic training per seed (torch.manual_seed covers weight init;
        # an explicit generator covers every minibatch permutation).
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)

        n_tr = len(seq_tr)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n_tr, generator=gen)
            for i in range(0, n_tr, 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                pred = model(seq_tr[idx], [b[idx] for b in blocks_tr])
                loss = F.mse_loss(pred, y_tr_t[idx])
                if args.entropy_reg > 0:
                    w = torch.sigmoid(model.gate_logits)
                    ent = -(w * torch.log(w + 1e-8) + (1 - w) * torch.log(1 - w + 1e-8))
                    # Minimizing entropy pushes gates toward 0/1 (sparse selection).
                    loss = loss + args.entropy_reg * ent.mean()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred_te = model(seq_te, blocks_te).cpu().numpy()
        sp = _spearman(y_te_raw, pred_te)
        gates = model.gate_values()

        all_res.append({"seed": seed, "spearman": round(sp, 6),
                        "gates": {n: round(float(gates[i]), 4) for i, n in enumerate(block_names)}})
        top = sorted(zip(block_names, gates), key=lambda kv: -kv[1])[:6]
        print(f"  seed={seed:2d}  sp={sp:.4f}  top_gates={[(n, round(float(g),3)) for n, g in top]}")

        out_dir = PROJECT_ROOT / args.log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(all_res, open(out_dir / "partial.json", "w"), indent=2)

    sps = np.array([r["spearman"] for r in all_res])
    agg = {n: float(np.mean([r["gates"].get(n, 0.0) for r in all_res]))
           for n in block_names}
    print(f"\n{'='*60}")
    print(f"gated fusion Spearman: {sps.mean():.4f} +/- {sps.std():.4f}")
    print("\nmean gate weights (sorted):")
    for n, w in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {n:<18} {w:.4f}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": all_res, "mean_gates": agg,
               "spearman_mean": float(sps.mean()), "spearman_std": float(sps.std())},
              open(out, "w"), indent=2)
    print(f"\n[OK] -> {out}")


if __name__ == "__main__":
    main()
