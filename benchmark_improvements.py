"""Benchmark grid: quantify the marginal value of each proposed improvement.

Cells of the grid:
  A. HGB,        raw features only          ← prior baseline
  B. HGB,        + engineered features
  C. CatBoost,   + engineered features
  D. Per-crop HGB,      + engineered features
  E. Per-crop CatBoost, + engineered features
  F. XGBoost,    + engineered features      (sanity check vs HGB on equal terms)

For each cell we report:
  - 5-fold CV RMSE on the train split (with stratified-by-crop folds)
  - Held-out test RMSE / R² / MAPE
  - Per-crop test R² (this is the metric we're actually trying to lift)

A → B  ⇒ value of engineered features
B → C  ⇒ value of CatBoost over HGB
B → D  ⇒ value of per-crop modeling
B → E  ⇒ value of per-crop + CatBoost
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from surrogate.data import load_dataset, split_data
from surrogate.evaluate import per_crop_report, regression_metrics
from surrogate.models import (
    CATBOOST_AVAILABLE,
    XGBOOST_AVAILABLE,
    build_model,
    build_per_crop,
)

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def stratified_cv(
    build_fn: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """K-fold CV stratified by crop_type so per-crop coverage is balanced."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses, fit_times = [], []
    for train_idx, val_idx in skf.split(X, X["crop_type"]):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        model = build_fn()
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - t0)
        preds = model.predict(X_va)
        rmses.append(float(np.sqrt(mean_squared_error(y_va, preds))))
    return {
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "fit_seconds_mean": float(np.mean(fit_times)),
    }


def evaluate_cell(
    name: str,
    build_fn: Callable[[], Any],
    splits: Dict[str, Any],
) -> Dict[str, Any]:
    print(f"\n=== {name} ===")
    cv = stratified_cv(build_fn, splits["X_train"], splits["y_train"])
    print(f"  CV RMSE (5-fold strat): {cv['cv_rmse_mean']:.4f} ± {cv['cv_rmse_std']:.4f}  "
          f"(fit ~{cv['fit_seconds_mean']:.2f}s/fold)")

    # Refit on train+val, evaluate on held-out test
    X_fit = pd.concat([splits["X_train"], splits["X_val"]], axis=0).reset_index(drop=True)
    y_fit = pd.concat([splits["y_train"], splits["y_val"]], axis=0).reset_index(drop=True)
    model = build_fn()
    model.fit(X_fit, y_fit)
    preds = model.predict(splits["X_test"])
    test = regression_metrics(splits["y_test"], preds)
    per_crop = per_crop_report(splits["X_test"], splits["y_test"], preds)
    print(f"  Test RMSE {test['rmse']:.4f} | R² {test['r2']:.4f} | MAPE {test['mape']:.4f}")
    print(f"  Per-crop R²: " +
          " | ".join(f"{r.crop_type}={r.r2:.3f}" for r in per_crop.itertuples()))
    return {
        "name": name,
        "cv": cv,
        "test": test,
        "per_crop": per_crop.to_dict(orient="records"),
    }


def main() -> None:
    print("=" * 70)
    print("Benchmark grid — feature engineering / CatBoost / per-crop")
    print("=" * 70)

    # Two versions of the dataset: with and without engineered features.
    # Both are cleaned (dedup + temp repair + missingness indicators).
    df_raw = load_dataset(clean=True, engineer=False)
    df_eng = load_dataset(clean=True, engineer=True)
    splits_raw = split_data(df_raw, random_state=42)
    splits_eng = split_data(df_eng, random_state=42)

    n_train = len(splits_raw["X_train"])
    n_val = len(splits_raw["X_val"])
    n_test = len(splits_raw["X_test"])
    n_raw = sum(1 for c in splits_raw["X_train"].columns
                if splits_raw["X_train"][c].dtype != object)
    n_eng = sum(1 for c in splits_eng["X_train"].columns
                if splits_eng["X_train"][c].dtype != object)
    print(f"Train/Val/Test: {n_train}/{n_val}/{n_test}")
    print(f"Numeric features — raw: {n_raw} | engineered: {n_eng}")

    cells: List[Dict[str, Any]] = []

    cells.append(evaluate_cell(
        "A. HGB raw (baseline)",
        lambda: build_model("hist_gb", include_engineered=False),
        splits_raw,
    ))
    cells.append(evaluate_cell(
        "B. HGB + engineered features",
        lambda: build_model("hist_gb"),
        splits_eng,
    ))
    if CATBOOST_AVAILABLE:
        cells.append(evaluate_cell(
            "C. CatBoost + engineered features",
            lambda: build_model("catboost", iterations=600, verbose=0),
            splits_eng,
        ))
    cells.append(evaluate_cell(
        "D. Per-crop HGB + engineered features",
        lambda: build_per_crop("hist_gb"),
        splits_eng,
    ))
    if CATBOOST_AVAILABLE:
        cells.append(evaluate_cell(
            "E. Per-crop CatBoost + engineered features",
            lambda: build_per_crop("catboost", iterations=600, verbose=0),
            splits_eng,
        ))
    if XGBOOST_AVAILABLE:
        cells.append(evaluate_cell(
            "F. XGBoost + engineered features (sanity)",
            lambda: build_model("xgboost"),
            splits_eng,
        ))

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    summary = pd.DataFrame([{
        "cell": c["name"],
        "cv_rmse": round(c["cv"]["cv_rmse_mean"], 4),
        "cv_std": round(c["cv"]["cv_rmse_std"], 4),
        "test_rmse": round(c["test"]["rmse"], 4),
        "test_r2": round(c["test"]["r2"], 4),
        "test_mape": round(c["test"]["mape"], 4),
        **{f"r2_{r['crop_type']}": round(r["r2"], 3) for r in c["per_crop"]},
    } for c in cells])
    print(summary.to_string(index=False))

    summary.to_csv(ARTIFACTS / "benchmark_summary.csv", index=False)
    (ARTIFACTS / "benchmark_details.json").write_text(json.dumps(cells, indent=2, default=str))
    print(f"\nWrote artifacts/benchmark_summary.csv and benchmark_details.json")


if __name__ == "__main__":
    main()
