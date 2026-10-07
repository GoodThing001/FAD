"""Locked test predictions + release model card (execution plan v8 Phase 4).

Trains the release config (303d five-model rank fusion, winsor_2.5, screen-k
=100 in-fold) on the historical train.csv (1600 rows, seed-42 split) and
produces locked predictions for val.csv + test.csv (200+200, never used in
model selection). The historical split is the only fixed partition available,
so it serves as the locked evaluation set; formal claims remain the 100-seed
reference.

Outputs (runs/release_package/):
  locked_val_predictions.csv / locked_test_predictions.csv  (per-model + fused)
  model_card.json  (config, data hash, feature hash, environment, metrics)

Usage (server, FAD_env):
  python -m scripts.reproduction.lock_test_predict
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

try:
    from .eval_fusion_smoothing import build, get_model
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_fusion_smoothing import build, get_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    data_dir = PROJECT_ROOT / "data/proxy2000_v2"
    tr = pd.read_csv(data_dir / "train.csv")
    va = pd.read_csv(data_dir / "val.csv")
    te = pd.read_csv(data_dir / "test.csv")

    seqs_tr = [str(s) for s in tr["Sequence"]]
    y_tr_raw = tr["gbsa"].to_numpy(dtype=np.float64)
    X_tr = build(seqs_tr)

    lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
    y_w = np.clip(y_tr_raw, lo, hi).astype(np.float32)
    et_s = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                               min_samples_leaf=5, random_state=42, n_jobs=1)
    et_s.fit(X_tr, y_w)
    top = np.argsort(-et_s.feature_importances_)[:100]
    X_tr = X_tr[:, top]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)

    model_names = ["RF", "ET", "GBR", "XGB", "SVR"]
    models = {m: get_model(m) for m in model_names}
    for m in models.values():
        m.fit(X_tr_s, y_w)

    out_dir = PROJECT_ROOT / "runs/release_package"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, df in [("val", va), ("test", te)]:
        seqs_te = [str(s) for s in df["Sequence"]]
        X_te = build(seqs_te)[:, top]
        X_te_s = sc.transform(X_te)
        pred = pd.DataFrame({"Sequence": seqs_te})
        preds = {}
        for mname in model_names:
            p = models[mname].predict(X_te_s)
            preds[mname] = p
            pred[mname] = p
        pred["fused_rank"] = np.mean([rankdata(preds[m]) for m in model_names], axis=0)
        pred["gbsa_true"] = df["gbsa"].values
        pred.to_csv(out_dir / f"locked_{name}_predictions.csv", index=False)
        sp = _spearman(df["gbsa"].values, pred["fused_rank"])
        results[name] = {"n": len(df), "gbsa_spearman_fused": float(sp)}
        print(f"{name}: n={len(df)}  fused Spearman={sp:.4f}")

    card = {
        "model": "v8 five-model rank fusion (release config)",
        "features": "303d (mut 52 + struct 43 + wd 208), screen-k=100 (ET importance, "
                    "fit on train fold)",
        "target": "gbsa, winsor_2.5",
        "models": {"RF": "n=1747 leaf=14", "ET": "n=1596 leaf=7",
                   "GBR": "n=500 dp=3 lr=0.03 loss=absolute_error",
                   "XGB": "n=500 dp=3 lr=0.03", "SVR": "rbf C=0.57 eps=0.047"},
        "fusion": "equal-weight rank average",
        "training_split": "historical seed-42 train.csv (1600); locked eval = val/test 200+200",
        "data_sha256": sha256_file(data_dir / "fad_proxy2000_v2_full.csv"),
        "feature_sha256": sha256_file(Path(__file__).resolve().parent / "eval_fusion_smoothing.py"),
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "reference_metrics": {
            "100_seed_gbsa_spearman_fusion": 0.37477,
            "ci95": [0.36691, 0.38231],
            "locked_split_metrics": results,
        },
    }
    json.dump(card, open(out_dir / "model_card.json", "w"), indent=2)
    print(f"[OK] -> {out_dir}")


if __name__ == "__main__":
    main()
