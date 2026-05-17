"""Training pipeline: cross-validation, hyperparameter tuning, comparison.

The contract is:
  1. Load the dataset.
  2. Stratified train/val/test split (test set never seen during tuning).
  3. K-fold CV on the train split to compare candidate models.
  4. Optional hyperparameter search on the top candidate.
  5. Refit the chosen model on train+val and report held-out test metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from surrogate.data import TARGET, load_dataset, split_data
from surrogate.models import MODEL_REGISTRY, build_model


@dataclass
class FoldResult:
    fold: int
    rmse: float
    mae: float
    r2: float
    fit_seconds: float


@dataclass
class CVResult:
    model_name: str
    fold_results: List[FoldResult] = field(default_factory=list)

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([f.rmse for f in self.fold_results]))

    @property
    def std_rmse(self) -> float:
        return float(np.std([f.rmse for f in self.fold_results]))

    @property
    def mean_r2(self) -> float:
        return float(np.mean([f.r2 for f in self.fold_results]))

    @property
    def mean_mae(self) -> float:
        return float(np.mean([f.mae for f in self.fold_results]))

    @property
    def mean_fit_seconds(self) -> float:
        return float(np.mean([f.fit_seconds for f in self.fold_results]))

    def to_row(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "rmse_mean": round(self.mean_rmse, 4),
            "rmse_std": round(self.std_rmse, 4),
            "mae_mean": round(self.mean_mae, 4),
            "r2_mean": round(self.mean_r2, 4),
            "fit_seconds_mean": round(self.mean_fit_seconds, 3),
        }


def _score(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 0,
    **model_overrides,
) -> CVResult:
    """K-fold CV. Returns per-fold and aggregated metrics.

    Stratification by crop_type would be ideal but KFold + a
    categorical column requires manual handling; since splits are
    already class-stratified at the train/val/test level and folds
    have ~1200 rows each, plain KFold is sufficient here.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = CVResult(model_name=model_name)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_va = X.iloc[val_idx]
        y_va = y.iloc[val_idx]
        pipe = build_model(model_name, **model_overrides)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - t0
        preds = pipe.predict(X_va)
        scores = _score(y_va, preds)
        result.fold_results.append(
            FoldResult(
                fold=fold_idx,
                rmse=scores["rmse"],
                mae=scores["mae"],
                r2=scores["r2"],
                fit_seconds=fit_seconds,
            )
        )
    return result


def compare_models(
    model_names: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Run K-fold CV for several models and return a leaderboard."""
    rows = []
    for name in model_names:
        cv = cross_validate_model(name, X, y, n_splits=n_splits, random_state=random_state)
        rows.append(cv.to_row())
    leaderboard = pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)
    return leaderboard


# ---------------------------------------------------------------------------
# Hyperparameter search. We use a small hand-written random search
# rather than sklearn's RandomizedSearchCV so the candidate grid stays
# explicit and the results table is easy to log/serialize.
# ---------------------------------------------------------------------------


def random_search(
    model_name: str,
    param_grid: Dict[str, List[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
    n_splits: int = 4,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample `n_iter` configurations and score each with K-fold CV.

    Returns a sorted DataFrame so the caller can `iloc[0]` for the best
    config.
    """
    rng = np.random.default_rng(random_state)
    keys = list(param_grid.keys())
    rows = []
    sampled = set()
    attempts = 0
    while len(rows) < n_iter and attempts < n_iter * 5:
        attempts += 1
        choice = {k: param_grid[k][rng.integers(0, len(param_grid[k]))] for k in keys}
        sig = tuple(sorted(choice.items()))
        if sig in sampled:
            continue
        sampled.add(sig)
        cv = cross_validate_model(
            model_name, X, y, n_splits=n_splits, random_state=random_state, **choice
        )
        rows.append({**choice, "rmse": cv.mean_rmse, "r2": cv.mean_r2})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


# Defaults that tend to be informative for the gradient-boosting class.
DEFAULT_HGB_GRID: Dict[str, List[Any]] = {
    "max_iter": [200, 400, 600, 1000],
    "learning_rate": [0.02, 0.05, 0.08, 0.12],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [5, 10, 20, 40],
    "l2_regularization": [0.0, 0.1, 1.0, 5.0],
}

DEFAULT_XGB_GRID: Dict[str, List[Any]] = {
    "n_estimators": [200, 400, 800, 1200],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.02, 0.05, 0.08, 0.12],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}


# ---------------------------------------------------------------------------
# End-to-end driver used by the CLI / notebook.
# ---------------------------------------------------------------------------


@dataclass
class TrainArtifacts:
    leaderboard: pd.DataFrame
    best_model_name: str
    best_params: Dict[str, Any]
    final_pipeline: Any
    test_metrics: Dict[str, float]
    per_crop_metrics: pd.DataFrame


def run_full_pipeline(
    candidate_models: Optional[List[str]] = None,
    do_hyperparam_search: bool = True,
    n_search_iter: int = 20,
    random_state: int = 0,
) -> TrainArtifacts:
    """Compare candidates, tune the winner, refit on train+val, score on test."""
    df = load_dataset()
    splits = split_data(df, random_state=random_state)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    if candidate_models is None:
        candidate_models = [m for m in ["xgboost", "hist_gb", "random_forest", "mlp", "ridge"]
                            if m in MODEL_REGISTRY]
        # rbf is intentionally optional in the default sweep due to cost.

    leaderboard = compare_models(candidate_models, X_train, y_train, random_state=random_state)
    best_name = leaderboard.iloc[0]["model"]

    best_params: Dict[str, Any] = {}
    if do_hyperparam_search:
        grid = None
        if best_name == "hist_gb":
            grid = DEFAULT_HGB_GRID
        elif best_name == "xgboost":
            grid = DEFAULT_XGB_GRID
        if grid is not None:
            search = random_search(
                best_name, grid, X_train, y_train, n_iter=n_search_iter,
                random_state=random_state,
            )
            best_params = {k: search.iloc[0][k] for k in grid.keys()}

    final_pipe = build_model(best_name, **best_params)
    # Refit on train+val so the test set remains untouched evaluation.
    X_fit = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_fit = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    final_pipe.fit(X_fit, y_fit)
    preds = final_pipe.predict(X_test)
    test_metrics = _score(y_test, preds)

    # Per-crop diagnostics: a single global R2 hides large per-crop swings.
    per_crop = (
        pd.DataFrame({
            "crop_type": X_test["crop_type"].values,
            "y_true": y_test.values,
            "y_pred": preds,
        })
        .groupby("crop_type")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "rmse": float(np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))),
            "mae": float(mean_absolute_error(g["y_true"], g["y_pred"])),
            "r2": float(r2_score(g["y_true"], g["y_pred"])),
        }))
        .reset_index()
    )

    return TrainArtifacts(
        leaderboard=leaderboard,
        best_model_name=best_name,
        best_params=best_params,
        final_pipeline=final_pipe,
        test_metrics=test_metrics,
        per_crop_metrics=per_crop,
    )
