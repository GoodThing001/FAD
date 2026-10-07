"""Physics features + sequence/physics fusion for the FAD surrogate.

Physics branch has two sources:
  - hand-crafted mut/struct/wd (303d), a deterministic transform of the sequence;
  - NUPACK ensemble features (cached CSV), which carry secondary-structure /
    thermodynamic information not directly encoded by the raw sequence.

The fused surrogate concatenates the transformer [CLS] embedding (128d) with the
selected physics vector and feeds a regression head. This implements the
汇报's "Sequence + Selected Physics -> Feature Fusion" step (Early Fusion as the
first baseline; Gated Fusion can be added later).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")


def build_handcrafted(seqs: list[str]) -> np.ndarray:
    """mut (52) + struct (43) + wd (208) = 303d, matching phase3_fusion_v8."""
    n = len(seqs)
    Xm = np.zeros((n, 52), dtype=np.float32)
    Xs = np.zeros((n, 43), dtype=np.float32)
    Xw = np.zeros((n, 208), dtype=np.float32)

    ROLES = {30: "stem", 31: "stem", 32: "stem", 33: "junction",
             89: "loop", 90: "loop", 91: "loop", 92: "junction",
             127: "stem", 128: "stem", 129: "stem", 130: "junction", 131: "loop"}
    PAIRS = {30: 33, 31: 32, 32: 31, 33: 30, 90: 92, 92: 90,
             127: 131, 131: 127, 128: 130, 130: 128}
    all_d = [a + b for a in BASES for b in BASES]

    for i, s0 in enumerate(seqs):
        s = str(s0).upper().replace("T", "U")
        mc = sm = lm = 0
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if idx < len(s) and s[idx] in BASES:
                Xm[i, j * 4 + BASES.index(s[idx])] = 1.0
            wt = WT_141[idx]; mb = s[idx]
            is_mut = 1.0 if mb != wt else 0.0
            role = ROLES.get(p, "unknown")
            Xs[i, j * 3] = 1.0 if role == "stem" else 0.0
            Xs[i, j * 3 + 1] = 1.0 if role == "loop" else 0.0
            Xs[i, j * 3 + 2] = is_mut * (1.0 if role == "stem" else 0.0)
            mc += is_mut
            if is_mut:
                if role == "stem": sm += 1
                elif role == "loop": lm += 1
        Xs[i, -4], Xs[i, -3], Xs[i, -2] = mc, sm, lm
        broken = 0
        for p in MUT_POS:
            part = PAIRS.get(p, 0)
            if part > 0 and s[p - 1] != WT_141[p - 1] and s[part - 1] == WT_141[part - 1]:
                broken += 1
        Xs[i, -1] = float(broken)
        for j, p in enumerate(MUT_POS):
            idx = p - 1; lo = max(0, idx - 5); hi = min(len(s), idx + 6)
            sub = s[lo:hi]
            for t in range(len(sub) - 1):
                d = sub[t:t + 2]
                if d[0] in BASES and d[1] in BASES:
                    Xw[i, j * 16 + all_d.index(d)] += 1
            total = max(1, len(sub) - 1)
            Xw[i, j * 16:j * 16 + 16] /= total
    return np.hstack([Xm, Xs, Xw])


# 192d NUPACK "useful" blocks (see feature_selection_pipeline.NUPACK_USEFUL_PATTERNS).
# Anchored regexes over BASE nupack_* columns only; delta_nupack_* columns are
# excluded so that the selection matches the documented 7-block 192d total.
_NUPACK_USEFUL = [
    r"nupack_contact_",                      # contact_pca
    r"nupack_inter_cluster_\d+_\d+$",        # cluster_pair
    r"nupack_intra_cluster_\d+$",
    r"nupack_mut_long_range$",
    r"nupack_n_very_stable_pairs$",          # energy_land
    r"nupack_n_moderate_pairs$",
    r"nupack_stable_pair_ratio$",
    r"nupack_eig_ipr$",
    r"nupack_eig_n_eff_modes$",
    r"nupack_pos\d+_entropy$",               # pos_entropy
    r"nupack_motif_global_",                 # motif_global
    r"nupack_motif_apt_",
    r"nupack_pos\d+_pair_prob$",             # pos_pair_prob
    r"nupack_pos\d+_likely_paired$",
    r"nupack_c\d+_mean_pair_prob$",
    r"nupack_mean_pair_prob$",
    r"nupack_std_pair_prob$",
    r"nupack_frac_paired_gt_0_5$",
    r"nupack_motif_pos\d+_is_",              # motif_positional
]


def load_nupack(csv_path: str, seqs: list[str], blocks: str = "useful") -> np.ndarray:
    """Load NUPACK features aligned to seqs. blocks: 'useful' (192d) or 'all'."""
    df = pd.read_csv(csv_path)
    df["Sequence"] = df["Sequence"].astype(str)
    cols = [c for c in df.columns if c != "Sequence"]
    if blocks == "useful":
        keep = [i for i, c in enumerate(cols)
                if not c.startswith("delta_")
                and any(re.search(p, c) for p in _NUPACK_USEFUL)]
        cols = [cols[i] for i in keep]
    elif blocks != "all":
        raise ValueError(f"unknown blocks={blocks}")
    df = df.set_index("Sequence").loc[[str(s) for s in seqs]]
    X = df[cols].to_numpy(dtype=np.float32)
    return X, cols


class FusedHead(nn.Module):
    """Regression head on concat([CLS] embedding, physics features)."""

    def __init__(self, d_model: int = 128, d_phy: int = 192, hidden: int = 256,
                 n_layers: int = 3, out_dim: int = 1, gate: bool = False):
        super().__init__()
        self.gate = gate
        if gate:
            self.g = nn.Sequential(nn.Linear(d_model + d_phy, d_phy), nn.Sigmoid())
            in_dim = d_model + d_phy
        else:
            in_dim = d_model + d_phy
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers.append(nn.Linear(d_model + d_phy if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor, phy: torch.Tensor) -> torch.Tensor:
        if self.gate:
            h = torch.cat([emb, self.g(torch.cat([emb, phy], dim=-1)) * phy], dim=-1)
        else:
            h = torch.cat([emb, phy], dim=-1)
        return self.net(h)


class FusedEnsemble(nn.Module):
    def __init__(self, encoder, n: int = 30, d_phy: int = 192, head_kwargs: dict | None = None):
        super().__init__()
        head_kwargs = head_kwargs or {}
        self.encoder = encoder
        self.n = n
        self.heads = nn.ModuleList(
            [FusedHead(d_model=encoder.d_model, d_phy=d_phy, **head_kwargs) for _ in range(n)]
        )

    def predict(self, token_ids, pad_mask, phy):
        self.encoder.eval()
        emb = self.encoder(token_ids, pad_mask)
        preds = torch.stack([h(emb, phy) for h in self.heads], dim=0)
        return preds.mean(dim=0), preds.std(dim=0), preds
