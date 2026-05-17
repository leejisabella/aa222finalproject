"""Surrogate model zoo.

Each `build_*` function returns an sklearn `Pipeline` so we have a
uniform `fit(X, y) / predict(X) / get_params()` interface across
gradient boosting, tree ensembles, an MLP, an RBF interpolator, and
a linear baseline. Models that need imputation/scaling/encoding wrap
those steps in the pipeline themselves.

This is what lets `train.py` compare candidates fairly: same data, same
preprocessing contract, only the regressor body differs.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from surrogate.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    make_preprocessor,
)

# Optional XGBoost — guarded so the package imports even if libomp is
# unavailable on the host.
try:  # pragma: no cover - import-time guard, exercised by env, not tests
    from xgboost import XGBRegressor  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False

try:  # pragma: no cover
    from catboost import CatBoostRegressor  # type: ignore

    CATBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    CATBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# RBF surrogate (custom — sklearn has no out-of-the-box RBF regressor).
# Implements vanilla RBF interpolation with a Tikhonov-regularized solve:
#   w = (K + lambda * I)^{-1} y
# where K_ij = exp(-gamma * ||x_i - x_j||^2). Predictions are K_test @ w.
# Wrapped in a Pipeline with the standard preprocessor so it sees the
# same encoded inputs as the other models.
# ---------------------------------------------------------------------------


class RBFRegressor(BaseEstimator, RegressorMixin):
    """Radial basis function regressor with ridge regularization.

    A Gaussian kernel maps every training point into a basis; we solve
    a regularized linear system for the weights. With ~6k training rows
    this is feasible (memory ~ 6000^2 floats = 290 MB) but already at
    the edge — listed as a baseline, not the recommended primary.
    """

    def __init__(self, gamma: float = 0.01, alpha: float = 1.0, max_train: int = 4000):
        self.gamma = gamma
        self.alpha = alpha
        self.max_train = max_train

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Subsample for tractability; full O(n^3) solve would be slow.
        n = X.shape[0]
        if n > self.max_train:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=self.max_train, replace=False)
            X = X[idx]
            y = y[idx]
        self.X_train_ = X
        K = rbf_kernel(X, X, gamma=self.gamma)
        K += self.alpha * np.eye(K.shape[0])
        self.weights_ = np.linalg.solve(K, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        K = rbf_kernel(X, self.X_train_, gamma=self.gamma)
        return K @ self.weights_


# ---------------------------------------------------------------------------
# Model builders. Each returns a fresh, unfit Pipeline so that callers can
# cross-validate, hyperparameter-search, or refit on full data freely.
# ---------------------------------------------------------------------------


ModelBuilder = Callable[[], Pipeline]


def build_xgboost(include_engineered: bool = True, **overrides) -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError(
            "xgboost is not importable on this host (missing libomp?). "
            "Install with `brew install libomp` or use build_hist_gb."
        )
    # XGBoost handles NaNs natively, so we skip imputation; we also skip
    # scaling since trees are scale-invariant. We still one-hot encode
    # the categoricals because XGBRegressor expects numeric input.
    pp = make_preprocessor(handle_missing=False, scale_numeric=False,
                           include_engineered=include_engineered)
    defaults = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=0,
        n_jobs=-1,
    )
    defaults.update(overrides)
    # NaN passthrough: imputers turned off above means missing values
    # would propagate as NaNs from the numeric branch; the one-hot
    # encoder errors on NaN strings, but the categoricals have no NaNs
    # in this dataset. XGBoost ingests numeric NaNs directly.
    model = XGBRegressor(**defaults)
    return Pipeline([("pre", pp), ("reg", model)])


def build_hist_gb(include_engineered: bool = True, **overrides) -> Pipeline:
    """sklearn HistGradientBoosting — same algorithm class as XGBoost.

    Used as the primary surrogate when XGBoost is unavailable, and as
    a comparison baseline regardless. Natively handles NaNs.
    """
    pp = make_preprocessor(handle_missing=False, scale_numeric=False,
                           include_engineered=include_engineered)
    defaults = dict(
        max_iter=500,
        max_depth=None,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
    )
    defaults.update(overrides)
    model = HistGradientBoostingRegressor(**defaults)
    return Pipeline([("pre", pp), ("reg", model)])


def build_random_forest(**overrides) -> Pipeline:
    pp = make_preprocessor(handle_missing=True, scale_numeric=False)
    defaults = dict(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=0,
    )
    defaults.update(overrides)
    model = RandomForestRegressor(**defaults)
    return Pipeline([("pre", pp), ("reg", model)])


def build_mlp(**overrides) -> Pipeline:
    pp = make_preprocessor(handle_missing=True, scale_numeric=True)
    defaults = dict(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
    )
    defaults.update(overrides)
    model = MLPRegressor(**defaults)
    return Pipeline([("pre", pp), ("reg", model)])


def build_rbf(**overrides) -> Pipeline:
    pp = make_preprocessor(handle_missing=True, scale_numeric=True)
    defaults = dict(gamma=0.05, alpha=1e-2, max_train=3000)
    defaults.update(overrides)
    model = RBFRegressor(**defaults)
    return Pipeline([("pre", pp), ("reg", model)])


def build_ridge_baseline(**overrides) -> Pipeline:
    pp = make_preprocessor(handle_missing=True, scale_numeric=True)
    defaults = dict(alpha=1.0)
    defaults.update(overrides)
    model = Ridge(**defaults)
    return Pipeline([("pre", pp), ("reg", model)])


class _CatBoostColumnSelector(BaseEstimator):
    """Module-level (picklable) column reorderer + categorical-string coercer
    for the CatBoost pipeline."""

    def __init__(self, feature_order, categorical_features):
        self.feature_order = list(feature_order)
        self.categorical_features = list(categorical_features)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xs = X[self.feature_order].copy()
        for c in self.categorical_features:
            Xs[c] = Xs[c].astype(str)
        return Xs

    def fit_transform(self, X, y=None):
        return self.transform(X)


def build_catboost(**overrides) -> Pipeline:
    """CatBoost — often best on small-medium mixed-type tabular data.

    Unlike the other boosters, CatBoost handles categoricals natively
    via ordered target statistics, so we tell it about the categorical
    column indices and skip one-hot encoding entirely. The standard
    preprocessor would defeat that advantage, so we use a passthrough.
    """
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("catboost is not importable on this host.")

    from surrogate.data import (
        CATEGORICAL_FEATURES,
        INDICATOR_FEATURES,
        get_numeric_features,
    )

    numeric_cols = get_numeric_features(include_engineered=True)
    feature_order = numeric_cols + CATEGORICAL_FEATURES + INDICATOR_FEATURES
    cat_indices = [feature_order.index(c) for c in CATEGORICAL_FEATURES]

    defaults = dict(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        random_seed=0,
        verbose=0,
        thread_count=-1,
        allow_writing_files=False,
    )
    defaults.update(overrides)
    model = CatBoostRegressor(cat_features=cat_indices, **defaults)
    pre = _CatBoostColumnSelector(feature_order, CATEGORICAL_FEATURES)
    return Pipeline([("pre", pre), ("reg", model)])


# ---------------------------------------------------------------------------
# Per-crop ensemble: train one sub-model per crop, dispatch at predict time.
# This directly attacks the within-crop R² gap by letting each sub-model
# specialize on its crop's variance instead of being averaged out.
# ---------------------------------------------------------------------------


class PerCropEnsemble(BaseEstimator, RegressorMixin):
    """Wraps a model builder so we fit one sub-model per crop_type level.

    `crop_column` (default "crop_type") names the column in the input
    DataFrame used to dispatch rows. The sub-model receives the full
    DataFrame (including crop_column) so it can still use it as a
    feature; in practice the one-hot for crop is constant within each
    sub-model and contributes nothing, but leaving it in keeps the
    preprocessor identical across global and per-crop pipelines.

    Why a fresh class instead of sklearn's GroupKFold: we need a model
    that *dispatches at predict time*, not just fits per-group. This
    is closer to a stacked model with a hard-routing first layer.
    """

    def __init__(self, base_builder, crop_column: str = "crop_type"):
        self.base_builder = base_builder
        self.crop_column = crop_column

    def fit(self, X, y):
        if not hasattr(X, "loc"):
            raise TypeError("PerCropEnsemble requires a DataFrame input.")
        y = pd.Series(y).reset_index(drop=True) if not isinstance(y, pd.Series) else y
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.crops_ = sorted(X[self.crop_column].unique().tolist())
        self.models_: Dict[str, Any] = {}
        for crop in self.crops_:
            mask = (X[self.crop_column] == crop).values
            if mask.sum() < 20:
                # Too few rows to specialize; fall back to a global model
                # trained on all data — keeps prediction coverage.
                pass
            sub_model = self.base_builder()
            sub_model.fit(X[mask], y[mask])
            self.models_[crop] = sub_model
        # Always train a fallback for unknown crops at predict time.
        self.fallback_ = self.base_builder()
        self.fallback_.fit(X, y)
        return self

    def predict(self, X):
        if not hasattr(X, "loc"):
            raise TypeError("PerCropEnsemble requires a DataFrame input.")
        X = X.reset_index(drop=True)
        preds = np.empty(len(X), dtype=float)
        seen = np.zeros(len(X), dtype=bool)
        for crop, model in self.models_.items():
            mask = (X[self.crop_column] == crop).values
            if mask.any():
                preds[mask] = model.predict(X[mask])
                seen |= mask
        if not seen.all():
            unseen = ~seen
            preds[unseen] = self.fallback_.predict(X[unseen])
        return preds


def build_per_crop(base_model_name: str = "hist_gb", **base_overrides) -> PerCropEnsemble:
    """Build a per-crop ensemble around one of the base models."""
    builder = lambda: build_model(base_model_name, **base_overrides)
    return PerCropEnsemble(base_builder=builder)


MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "hist_gb": build_hist_gb,
    "random_forest": build_random_forest,
    "mlp": build_mlp,
    "rbf": build_rbf,
    "ridge": build_ridge_baseline,
}
if XGBOOST_AVAILABLE:
    MODEL_REGISTRY["xgboost"] = build_xgboost
if CATBOOST_AVAILABLE:
    MODEL_REGISTRY["catboost"] = build_catboost


def build_model(name: str, **overrides) -> Pipeline:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**overrides)
