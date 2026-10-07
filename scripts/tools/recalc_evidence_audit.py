# -*- coding: utf-8 -*-
"""独立统计复核：用证据目录中的 CSV 实际数据重算文档声明的统计量。
Spearman: rankdata + pearsonr（Windows 安全实现）。
Bootstrap: np.random.default_rng(20260903).choice 重采样 20000 次，CI 取 2.5/97.5 分位。
所有输入均为实际文件数据；文档数字只作为对照对象，不参与计算。
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr

RNG_SEED = 20260903
N_BOOT = 20000
OUT = {"rng_seed": RNG_SEED, "n_boot": N_BOOT}


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(pearsonr(rankdata(a), rankdata(b)).statistic)


def boot_mean_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(RNG_SEED)
    m = np.empty(N_BOOT)
    for i in range(N_BOOT):
        m[i] = x[rng.choice(n, n, replace=True)].mean()
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired_stats(d):
    d = np.asarray(d, float)
    m, lo, hi = boot_mean_ci(d)
    return {"n": int(len(d)), "mean_delta": m, "ci95_low": lo, "ci95_high": hi,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "losses": int((d < 0).sum())}


def f3(x):
    return round(float(x), 6)


# ---------------------------------------------------------------- 1. v8 fusion
def audit_v8(path, key):
    df = pd.read_csv(path)
    print(f"[{key}] rows={len(df)} seeds={sorted(df.seed.unique())[:3]}...{sorted(df.seed.unique())[-3:]}")
    piv = df.pivot_table(index="seed", columns=["method", "model"], values="gbsa_Sp")
    res = {}
    eq = piv[("equal_weight_rank", "fusion")]
    res["equal_weight_rank_mean"], res["equal_weight_rank_ci_low"], res["equal_weight_rank_ci_high"] = boot_mean_ci(eq)
    res["constrained_weight_mean"] = f3(piv[("constrained_weight", "fusion")].mean())
    res["single_means"] = {m: f3(piv[("single", m)].mean()) for m in ["RF", "ET", "GBR", "XGB", "SVR"]}
    res["single_std"] = {m: f3(piv[("single", m)].std(ddof=1)) for m in ["RF", "ET", "GBR", "XGB", "SVR"]}
    res["vs_xgb"] = paired_stats(eq - piv[("single", "XGB")])
    best = piv["single"].max(axis=1)
    res["vs_best_single"] = paired_stats(eq - best)
    res["n_seeds"] = int(len(eq))
    res["seeds_min"], res["seeds_max"] = int(eq.index.min()), int(eq.index.max())
    res["n_unique_seeds"] = int(eq.index.nunique())
    OUT[key] = res
    return df, piv


d100, p100 = audit_v8("evidence/server_sync_20260902/v8_fusion_100seed/all_results.csv", "v8_fusion_100seed")
d30, p30 = audit_v8("evidence/server_sync_20260902/v8_fusion_30seed/all_results.csv", "v8_fusion_30seed")

# 30-seed 与 100-seed 行级一致性
mrg = d30.merge(d100, on=["seed", "method", "model"], suffixes=("_30", "_100"))
same = (mrg.gbsa_Sp_30 - mrg.gbsa_Sp_100).abs().max()
OUT["v8_30_vs_100"] = {"overlap_rows": int(len(mrg)),
                       "max_abs_diff_shared_rows": float(same),
                       "rows_in_30_only": int(len(d30) - len(mrg))}

# ---------------------------------------------------------------- 2. phase1_v8
df1 = pd.read_csv("evidence/recovered_runs/phase1_v8/all_results.csv")
g = df1.groupby(["target", "loss", "features", "screen", "model"]).size()
OUT["phase1_v8"] = {
    "rows": int(len(df1)),
    "n_unique_seeds": int(df1.seed.nunique()),
    "seed_min": int(df1.seed.min()), "seed_max": int(df1.seed.max()),
    "combo_counts": {str(k): int(v) for k, v in g.value_counts().items()},
    "model_mean_gbsa_Sp": {m: f3(df1[df1.model == m].gbsa_Sp.mean()) for m in
                           ["RF", "ET", "SVR", "GBR", "XGB"]},
}
# 0.395 声明对应切片：winsor_2.5 + mut_struct_wd + GBR/XGB + MAE/Huber + nested_100d
slices = {}
for model in ["GBR", "XGB"]:
    for loss in ["mse", "mae", "huber_1", "huber_2"]:
        s = df1[(df1.target == "winsor_2.5") & (df1.features == "mut_struct_wd") &
                (df1.loss == loss) & (df1.screen == "nested_100d") & (df1.model == model)].gbsa_Sp
        if len(s):
            slices[f"{model}+{loss}@nested_100d"] = {"n": int(len(s)), "mean": f3(s.mean()),
                                                     "std": f3(s.std(ddof=1))}
# 每 seed 在 {GBR,XGB}×{mae,huber_1,huber_2}×nested_100d 上的最优
pool = df1[(df1.target == "winsor_2.5") & (df1.features == "mut_struct_wd") &
           (df1.screen == "nested_100d") & (df1.loss.isin(["mae", "huber_1", "huber_2"])) &
           (df1.model.isin(["GBR", "XGB"]))]
per_seed_best = pool.groupby("seed").gbsa_Sp.max()
OUT["phase1_v8"]["winsor_mut_struct_wd_nested100_slices"] = slices
OUT["phase1_v8"]["per_seed_best_GBR_XGB_mae_huber"] = {"mean": f3(per_seed_best.mean()),
                                                       "std": f3(per_seed_best.std(ddof=1)),
                                                       "n": int(len(per_seed_best))}

# ---------------------------------------------------------------- 3. recovered v10/v10b/v11/v12/v13
def paired_block(csv_path, seed_col, block_col, model_col, val_col, baseline, candidate, model=None):
    df = pd.read_csv(csv_path)
    if model is not None:
        df = df[df[model_col] == model]
    b = df[df[block_col] == baseline].set_index(seed_col)[val_col]
    c = df[df[block_col] == candidate].set_index(seed_col)[val_col]
    idx = b.index.intersection(c.index)
    d = (c.loc[idx] - b.loc[idx]).values
    return paired_stats(d), float(b.mean()), float(c.mean())


pairs = [
    ("v10b", "XGB", "baseline_seq_only", "contact_pca"),
    ("v11", "XGB", "seq_303d", "seq_303d+useful_192d"),
    ("v11", "GBR", "seq_303d", "seq_303d+useful_192d"),
    ("v13", "GBR", "seq_303d", "seq+nupack+rna_270d"),
    ("v13", "XGB", "seq_303d", "seq+nupack+rna_270d"),
    ("v13", "XGB", "seq_303d", "seq_303d+nupack_192d"),
    ("v13", "GBR", "seq_303d", "seq_303d+nupack_192d"),
    ("v10", "XGB", "baseline_seq_only", "contact_pca"),
    ("v10", "GBR", "baseline_seq_only", "contact_pca"),
    ("v12", "GBR", "baseline_seq", "rna_properties"),
    ("v12", "XGB", "baseline_seq", "local_window"),
    ("v12", "XGB", "baseline_seq", "rna_properties"),
]
colmap = {"v10": ("seed", "block", "model", "gbsa_Sp"),
          "v10b": ("seed", "block", "model", "gbsa_Sp"),
          "v11": ("seed", "features", "model", "gbsa_Sp"),
          "v12": ("seed", "feature", "model", "gbsa_Sp"),
          "v13": ("seed", "combo", "model", "gbsa_Sp")}
OUT["recovered_pairs"] = {}
for run, model, base, cand in pairs:
    sc, bc, mc, vc = colmap[run]
    st, bm, cm = paired_block(f"evidence/recovered_runs/{run}/all_results.csv",
                              sc, bc, mc, vc, base, cand, model)
    key = f"{run}|{model}|{base}->{cand}"
    OUT["recovered_pairs"][key] = {"paired": st, "baseline_mean": f3(bm), "candidate_mean": f3(cm)}
    print(key, "Δ=", round(st["mean_delta"], 6), "CI=[", round(st["ci95_low"], 6), ",",
          round(st["ci95_high"], 6), "] wins", st["wins"], "/", st["n"],
          "base_mean", round(bm, 6), "cand_mean", round(cm, 6))

# v10b 全部 block vs baseline（补充全景）
df10b = pd.read_csv("evidence/recovered_runs/v10b/all_results.csv")
b10b = df10b[df10b.block == "baseline_seq_only"].set_index("seed").gbsa_Sp
OUT["v10b_all_blocks_vs_baseline"] = {}
for blk in sorted(df10b.block.unique()):
    if blk == "baseline_seq_only":
        continue
    c = df10b[df10b.block == blk].set_index("seed").gbsa_Sp
    idx = b10b.index.intersection(c.index)
    OUT["v10b_all_blocks_vs_baseline"][blk] = paired_stats((c.loc[idx] - b10b.loc[idx]).values)

# ---------------------------------------------------------------- 4. feature_selection_all_safe_30seed
fs = pd.read_csv("evidence/server_sync_20260903/feature_selection_all_safe_30seed/metrics.csv")
ctrl = fs[fs.method == "all"].set_index(["seed", "model"]).Spearman
cells = {}
for method in ["mutual_info", "spearman", "tree_importance"]:
    for model in ["ET", "GBR", "XGB"]:
        sel = fs[(fs.method == method) & (fs.model == model)].set_index("seed").Spearman
        idx = sel.index.intersection(ctrl.index.get_level_values(0))
        d = (sel.loc[idx].values - ctrl.loc[idx, model].values)
        st = paired_stats(d)
        cells[f"{method}|{model}"] = {
            "paired": st,
            "sel_mean": f3(float(sel.mean())),
            "ctrl_mean": f3(float(ctrl.xs(model, level="model").mean())),
        }
OUT["fs_paired"] = {"rows": int(len(fs)),
                    "n_unique_seeds": int(fs.seed.nunique()),
                    "cells": cells}

cs = pd.read_csv("evidence/server_sync_20260903/feature_selection_all_safe_30seed/cluster_split_check.csv")
cc = cs[cs.method == "all"].set_index(["fold_id", "model"]).Spearman
cs_cells = {}
for method in ["mutual_info", "spearman", "tree_importance"]:
    for model in ["ET", "GBR", "XGB"]:
        sel = cs[(cs.method == method) & (cs.model == model)].set_index("fold_id").Spearman
        idx = sel.index.intersection(cc.index.get_level_values(0).unique())
        d = (sel.loc[idx].values - cc.loc[(idx, model)].values)
        cs_cells[f"{method}|{model}"] = {"n_folds": int(len(d)), "mean_delta": f3(float(d.mean())),
                                        "min_delta": f3(float(d.min())),
                                        "passes_max_drop_0.01": bool(d.min() > -0.01)}
OUT["fs_cluster_split"] = {"rows": int(len(cs)), "n_folds": int(cs.fold_id.nunique()), "cells": cs_cells}

# ---------------------------------------------------------------- 5. 数据文件验证
dat = pd.read_csv("data/proxy2000_v2/fad_proxy2000_v2_full.csv")
lab_pred = 0.961 * dat.Gap2 + 0.039 * dat.gbsa
resid = (dat.label - lab_pred)
# 无截距回归求精确系数
X = np.column_stack([dat.Gap2, dat.gbsa])
coef, *_ = np.linalg.lstsq(X, dat.label.values, rcond=None)
r2 = 1 - float(np.sum(resid ** 2) / np.sum((dat.label - dat.label.mean()) ** 2))
var_lab = float(dat.label.var(ddof=1))
var_g = float(dat.gbsa.var(ddof=1))
cov_term = 2 * 0.961 * 0.039 * float(np.cov(dat.Gap2, dat.gbsa, ddof=1)[0, 1])
OUT["data"] = {
    "rows": int(len(dat)),
    "cols": list(dat.columns),
    "nan_counts": {c: int(dat[c].isna().sum()) for c in dat.columns},
    "label_formula_max_abs_residual": float(resid.abs().max()),
    "label_formula_mean_abs_residual": float(resid.abs().mean()),
    "label_formula_r2_vs_actual": float(r2),
    "ols_nointercept_coefs": [float(coef[0]), float(coef[1])],
    "sp_Gap2_gbsa": spearman(dat.Gap2, dat.gbsa),
    "sp_gbsa_MFE": spearman(dat.gbsa, dat.MFE),
    "sp_label_Gap2": spearman(dat.label, dat.Gap2),
    "sp_label_gbsa": spearman(dat.label, dat.gbsa),
    "gbsa_mean": float(dat.gbsa.mean()), "gbsa_std": float(dat.gbsa.std(ddof=1)),
    "gbsa_min": float(dat.gbsa.min()), "gbsa_max": float(dat.gbsa.max()),
    "var_label": var_lab,
    "gbsa_variance_share_pure": float(0.039 ** 2 * var_g / var_lab),
    "gbsa_variance_share_with_cov": float((0.039 ** 2 * var_g + cov_term) / var_lab),
    "Gap2_variance_share_with_cov": float((0.961 ** 2 * float(dat.Gap2.var(ddof=1)) + cov_term) / var_lab),
    "decomp_check": float((0.961 ** 2 * dat.Gap2.var(ddof=1) + 0.039 ** 2 * var_g + cov_term)),
}

with open("evidence/_recalc_audit_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)
print("\n=== FINAL JSON ===")
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
