"""Surrogate implementations for the closed loop (interface v0.2.0).

Both surrogates consume ONLY 141-nt sequences (they featurize internally with the
release 303d builder) and predict standalone gbsa (mu) with an optional uncertainty
estimate (sigma). No dock / Gap2 / label / gbsa column is ever used as input.

  XGBEnsembleSurrogate    bootstrap ensemble of XGB regressors (fast, sigma =
                          member std, `uncertainty_kind="ensemble_spread_uncalibrated"`).
                          Falls back to sklearn GBM when xgboost is unavailable
                          (Windows/local smoke runs) — the backend is recorded in
                          `model_id` (`xgb_boot{n}_xgboost` / `xgb_boot{n}_sklearn_gbm`),
                          so a local smoke run is never mistaken for the server backend.
  ReleaseFusionSurrogate  the five release model types (RF/ET/GBR/XGB/SVR) on
                          the same 303d features with in-fold screen-k=100;
                          sigma = member std. This is the five-model *mean*, not the
                          v8 rank fusion, so the 0.375 release number does not apply.

`model_id` and `feature_version` are written into every ledger row, which is how a
run can be traced back to the exact model that scored it.
"""

from __future__ import annotations

import pickle
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # package import (python -m scripts.loop...)
    from .interface import (Prediction, TargetSpec, validate_sequences)
    from scripts.reproduction.eval_fusion_smoothing import build, get_model
except ImportError:  # plain-script invocation
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.loop.interface import (Prediction, TargetSpec, validate_sequences)
    from scripts.reproduction.eval_fusion_smoothing import build, get_model


def _make_xgb(seed: int, n_estimators: int = 400):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=n_estimators, max_depth=4,
                            learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed,
                            verbosity=0, n_jobs=1)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                         learning_rate=0.05, random_state=seed)


def _winsor(y: np.ndarray, pct: float = 2.5):
    lo, hi = np.percentile(y, pct), np.percentile(y, 100 - pct)
    return np.clip(y, lo, hi), (float(lo), float(hi))


class XGBEnsembleSurrogate:
    """Bootstrap ensemble; sigma = std across members (predictive spread)."""

    feature_version = "seq303"

    def __init__(self, n_members: int = 10, seed: int = 42, subsample: float = 0.8,
                 screen_k: int | None = None, target: TargetSpec | None = None,
                 with_uncertainty: bool = True):
        self.n_members = int(n_members)
        self.seed = int(seed)
        self.subsample = float(subsample)
        self.screen_k = None if screen_k is None else int(screen_k)
        self.target = target or TargetSpec()
        self.with_uncertainty = bool(with_uncertainty)
        self.models: list = []
        self.top_idx: np.ndarray | None = None
        self.clip: tuple[float, float] | None = None
        self.n_train: int = 0
        self.backend: str = "unset"
        self._fit_id = "unfitted"

    @property
    def model_id(self) -> str:
        return f"xgb_boot{self.n_members}_{self.backend}_{self._fit_id}"

    @property
    def feature_version(self) -> str:
        if self.screen_k is not None:
            return f"seq303_screen{self.screen_k}"
        return "seq303"

    def fit(self, sequences: Sequence[str], y: np.ndarray) -> "XGBEnsembleSurrogate":
        seqs = validate_sequences(sequences, "fit sequences")
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(seqs) != len(y):
            raise ValueError(f"sequence/label length mismatch: {len(seqs)} vs {len(y)}")
        if not np.isfinite(y).all():
            raise ValueError("training labels contain non-finite values")
        y_w, self.clip = _winsor(y)
        X = build(seqs)
        if self.screen_k is not None and self.screen_k < X.shape[1]:
            from sklearn.ensemble import ExtraTreesRegressor
            et = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                                     min_samples_leaf=5, random_state=self.seed,
                                     n_jobs=1)
            et.fit(X, y_w)
            self.top_idx = np.argsort(-et.feature_importances_)[: self.screen_k]
            X = X[:, self.top_idx]
        rng = np.random.RandomState(self.seed)
        n = len(seqs)
        self.models = []
        for m in range(self.n_members):
            idx = rng.choice(n, size=max(2, int(round(self.subsample * n))),
                             replace=True)
            model = _make_xgb(self.seed + m)
            model.fit(X[idx], y_w[idx])
            self.models.append(model)
        self.backend = "xgboost" if type(self.models[0]).__module__.startswith("xgboost") \
            else "sklearn_gbm"
        self.n_train = n
        self._fit_id = hashlib.sha256(pickle.dumps((seqs, y, self.models,
                            self.top_idx), protocol=4)).hexdigest()[:16]
        return self

    def predict(self, sequences: Sequence[str]) -> Prediction:
        if not self.models:
            raise RuntimeError("surrogate is not fitted")
        seqs = validate_sequences(sequences, "predict sequences")
        X = build(seqs)
        if self.top_idx is not None:
            X = X[:, self.top_idx]
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        sigma = preds.std(axis=0) if self.with_uncertainty else None
        return Prediction(mu=preds.mean(axis=0), sigma=sigma,
                          uncertainty_kind="ensemble_spread_uncalibrated" if sigma is not None else None,
                          feature_version=self.feature_version,
                          model_id=self.model_id,
                          members=tuple(f"xgb_boot_{i}" for i in range(len(self.models))))

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "top_idx": self.top_idx,
                         "clip": self.clip, "seed": self.seed,
                         "model_id": self.model_id, "fit_id": self._fit_id,
                         "feature_version": self.feature_version,
                         "n_train": self.n_train, "backend": self.backend,
                         "target": self.target.describe()}, f)


class ReleaseFusionSurrogate:
    """Five release model types (RF/ET/GBR/XGB/SVR) on 303d with in-fold screen-k.

    NOTE (plan §10): this adapter averages the five models' raw GBSA-scale
    predictions. It is NOT the v8 rank fusion and must not inherit the v8 0.375
    reference; `model_id` records what it actually is.
    """

    feature_version = "seq303_screen100"

    def __init__(self, screen_k: int = 100, seed: int = 42,
                 target: TargetSpec | None = None, n_members: int | None = None,
                 with_uncertainty: bool = True):
        self.screen_k = int(screen_k)
        self.seed = seed
        self.target = target or TargetSpec()
        self.models: dict = {}
        self.top_idx: np.ndarray | None = None
        self.clip: tuple[float, float] | None = None
        self.n_train: int = 0
        self.with_uncertainty = bool(with_uncertainty)
        self._fit_id = "unfitted"
        if n_members is not None:
            print(f"[surrogate] ReleaseFusionSurrogate ignores n_members={n_members} "
                  "(five fixed model types)")

    @property
    def model_id(self) -> str:
        return "release_five_model_mean_v1_" + self._fit_id

    def fit(self, sequences: Sequence[str], y: np.ndarray) -> "ReleaseFusionSurrogate":
        seqs = validate_sequences(sequences, "fit sequences")
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(seqs) != len(y):
            raise ValueError(f"sequence/label length mismatch: {len(seqs)} vs {len(y)}")
        if not np.isfinite(y).all():
            raise ValueError("training labels contain non-finite values")
        y_w, self.clip = _winsor(y)
        X = build(seqs)

        if self.screen_k < X.shape[1]:
            from sklearn.ensemble import ExtraTreesRegressor
            et = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                                     min_samples_leaf=5, random_state=self.seed,
                                     n_jobs=1)
            et.fit(X, y_w)
            self.top_idx = np.argsort(-et.feature_importances_)[: self.screen_k]
            X = X[:, self.top_idx]

        self.models = {}
        for name in ("RF", "ET", "GBR", "XGB", "SVR"):
            m = get_model(name)
            m.fit(X, y_w)
            self.models[name] = m
        self.n_train = len(seqs)
        self._fit_id = hashlib.sha256(pickle.dumps((seqs, y, self.models,
                            self.top_idx), protocol=4)).hexdigest()[:16]
        return self

    def predict(self, sequences: Sequence[str]) -> Prediction:
        if not self.models:
            raise RuntimeError("surrogate is not fitted")
        seqs = validate_sequences(sequences, "predict sequences")
        X = build(seqs)
        if self.top_idx is not None:
            X = X[:, self.top_idx]
        preds = np.stack([m.predict(X) for m in self.models.values()], axis=0)
        sigma = preds.std(axis=0) if self.with_uncertainty else None
        return Prediction(mu=preds.mean(axis=0), sigma=sigma,
                          uncertainty_kind="ensemble_spread_uncalibrated" if sigma is not None else None,
                          feature_version=self.feature_version,
                          model_id=self.model_id,
                          members=tuple(self.models.keys()))

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "top_idx": self.top_idx,
                         "clip": self.clip, "n_train": self.n_train,
                         "model_id": self.model_id, "fit_id": self._fit_id,
                         "feature_version": self.feature_version,
                         "target": self.target.describe()}, f)


class RandomForestSurrogate(XGBEnsembleSurrogate):
    """Fixed RF diagnostic adapter, not the historical mutation-only baseline."""

    def __init__(self, seed=42, screen_k=100, n_members=None,
                 with_uncertainty=True, **kwargs):
        super().__init__(seed=seed, screen_k=screen_k,
                         with_uncertainty=with_uncertainty)

    @property
    def model_id(self):
        return "rf300_seq303_" + self._fit_id

    def fit(self, sequences, y):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        seqs = validate_sequences(sequences, "RF fit sequences")
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(seqs) != len(y) or not np.isfinite(y).all():
            raise ValueError("invalid RF training labels")
        y_w, self.clip = _winsor(y)
        X = build(seqs)
        self.top_idx = None
        if self.screen_k is not None and self.screen_k < X.shape[1]:
            screen = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                min_samples_leaf=5, random_state=self.seed, n_jobs=1).fit(X, y_w)
            self.top_idx = np.argsort(-screen.feature_importances_)[:self.screen_k]
            X = X[:, self.top_idx]
        self.models = [RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
            max_features="sqrt", random_state=self.seed, n_jobs=1).fit(X, y_w)]
        self.backend = "sklearn_rf"
        self.n_train = len(seqs)
        self._fit_id = hashlib.sha256(pickle.dumps((seqs, y, self.models,
                            self.top_idx), protocol=4)).hexdigest()[:16]
        return self

    def predict(self, sequences):
        if not self.models:
            raise RuntimeError("RF surrogate is not fitted")
        seqs = validate_sequences(sequences, "RF predict sequences")
        X = build(seqs)
        if self.top_idx is not None:
            X = X[:, self.top_idx]
        # Per-tree spread is diagnostic only and is not calibrated error.
        model = self.models[0]
        mu = model.predict(X)
        sigma = np.std([tree.predict(X) for tree in model.estimators_], axis=0) \
            if self.with_uncertainty else None
        return Prediction(mu=mu, sigma=sigma,
            uncertainty_kind="tree_spread_uncalibrated" if sigma is not None else None,
            model_id=self.model_id, feature_version=self.feature_version)


SURROGATES = {
    "xgb": XGBEnsembleSurrogate,
    "fusion": ReleaseFusionSurrogate,
    "rf": RandomForestSurrogate,
}


def make_surrogate(name: str, **kwargs):
    if name not in SURROGATES:
        raise ValueError(f"unknown surrogate '{name}'; choices: {sorted(SURROGATES)}")
    return SURROGATES[name](**kwargs)
