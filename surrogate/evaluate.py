"""Evaluation utilities: diagnostic metrics + plots.

Plots are written to artifacts/ so they can be inspected after the
training script runs. All matplotlib code is import-guarded to a
non-interactive backend so the module works headless.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """RMSE / MAE / R² / MAPE.

    MAPE uses a 1e-3 epsilon so divisions by tiny lettuce yields don't
    blow up; values near zero (0.1 kg/m^2) are rare but real.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-3, None))))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


def per_crop_report(
    X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray
) -> pd.DataFrame:
    """Break metrics out by crop_type."""
    rows = []
    for crop, group in X.groupby("crop_type"):
        idx = group.index
        m = regression_metrics(y_true.loc[idx], y_pred[idx])
        rows.append({"crop_type": crop, "n": int(len(group)), **{k: round(v, 4) for k, v in m.items()}})
    return pd.DataFrame(rows).sort_values("crop_type").reset_index(drop=True)


def plot_pred_vs_actual(
    y_true,
    y_pred,
    out_path: Path,
    title: str = "Predicted vs actual yield (kg/m^2)",
    crops: Optional[Iterable[str]] = None,
) -> None:
    """Scatter with y=x reference. Color by crop if supplied."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if crops is not None:
        crops = list(crops)
        for crop in sorted(set(crops)):
            mask = np.array([c == crop for c in crops])
            ax.scatter(np.asarray(y_true)[mask], np.asarray(y_pred)[mask],
                       label=crop, alpha=0.5, s=10)
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Actual yield (kg/m²)")
    ax.set_ylabel("Predicted yield (kg/m²)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_residuals(y_true, y_pred, out_path: Path) -> None:
    """Residual scatter vs predicted, and residual histogram."""
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, resid, alpha=0.4, s=10)
    axes[0].axhline(0, color="k", linewidth=1)
    axes[0].set_xlabel("Predicted yield (kg/m²)")
    axes[0].set_ylabel("Residual (pred − actual)")
    axes[0].set_title("Residuals vs predicted")

    axes[1].hist(resid, bins=60, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="k", linewidth=1)
    axes[1].set_xlabel("Residual (kg/m²)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_feature_importance(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    out_path: Path,
    n_repeats: int = 5,
    random_state: int = 0,
    top_k: int = 15,
) -> pd.DataFrame:
    """Permutation importance on the held-out set.

    Permutation importance is model-agnostic and reflects how much
    yield predictions degrade when a feature is shuffled — more
    trustworthy than tree-internal `feature_importances_` here because
    the pipeline produces 30+ engineered columns from a small set of
    raw inputs.
    """
    result = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * top_k)))
    top = importances.head(top_k).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color="darkorange", edgecolor="black")
    ax.set_xlabel("Δ RMSE under permutation (higher = more important)")
    ax.set_title("Permutation feature importance (test set)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return importances


def plot_model_comparison(leaderboard: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of CV RMSE across candidate models."""
    fig, ax = plt.subplots(figsize=(7, 4))
    order = leaderboard.sort_values("rmse_mean")
    ax.barh(order["model"], order["rmse_mean"], xerr=order["rmse_std"],
            color="seagreen", edgecolor="black")
    ax.invert_yaxis()
    ax.set_xlabel("CV RMSE (kg/m²) — lower is better")
    ax.set_title("Candidate model comparison (5-fold CV)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_learning_curve(
    pipeline_builder,
    X: pd.DataFrame,
    y: pd.Series,
    out_path: Path,
    fractions: Iterable[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    random_state: int = 0,
) -> pd.DataFrame:
    """How does test-fold RMSE evolve as we feed in more training data?

    A flat curve says we have enough data; a steeply falling curve says
    we'd benefit from more harvests in the dataset.
    """
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(random_state)
    rows = []
    for frac in fractions:
        kf = KFold(n_splits=4, shuffle=True, random_state=random_state)
        fold_rmses = []
        for train_idx, val_idx in kf.split(X):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            n_sub = max(50, int(len(X_tr) * frac))
            sub_idx = rng.choice(len(X_tr), size=n_sub, replace=False)
            pipe = pipeline_builder()
            pipe.fit(X_tr.iloc[sub_idx], y_tr.iloc[sub_idx])
            preds = pipe.predict(X.iloc[val_idx])
            fold_rmses.append(float(np.sqrt(mean_squared_error(y.iloc[val_idx], preds))))
        rows.append({"frac": frac, "rmse_mean": np.mean(fold_rmses),
                     "rmse_std": np.std(fold_rmses)})
    curve = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(curve["frac"], curve["rmse_mean"], yerr=curve["rmse_std"],
                marker="o", capsize=4)
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("CV RMSE (kg/m²)")
    ax.set_title("Learning curve")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return curve
