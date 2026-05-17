"""Train the TOMATO surrogate model and produce evaluation artifacts.

Usage:
    python train_surrogate.py

Writes:
    artifacts/surrogate.pkl                Pickled SurrogatePredictor
    artifacts/leaderboard.csv              CV comparison of candidate models
    artifacts/test_metrics.json            Final held-out test metrics
    artifacts/per_crop_metrics.csv         Per-crop test metrics
    artifacts/feature_importance.csv       Permutation importance
    artifacts/dataset_summary.json         Dataset diagnostics
    artifacts/best_params.json             Tuned hyperparameters
    artifacts/learning_curve.csv           Learning curve points
    artifacts/plots/*.png                  Diagnostic plots
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate.api import build_predictor_from_pipeline, save_predictor
from surrogate.data import load_dataset, split_data, summarize_dataset
from surrogate.evaluate import (
    per_crop_report,
    plot_feature_importance,
    plot_learning_curve,
    plot_model_comparison,
    plot_pred_vs_actual,
    plot_residuals,
    regression_metrics,
)
from surrogate.models import MODEL_REGISTRY, XGBOOST_AVAILABLE, build_model
from surrogate.train import (
    DEFAULT_HGB_GRID,
    DEFAULT_XGB_GRID,
    compare_models,
    random_search,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PLOTS = ARTIFACTS / "plots"


def _coerce_for_json(obj):
    if isinstance(obj, dict):
        return {k: _coerce_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_for_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-search", action="store_true",
                        help="Skip hyperparameter random search.")
    parser.add_argument("--search-iter", type=int, default=20)
    parser.add_argument("--no-learning-curve", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TOMATO surrogate training")
    print("=" * 70)

    # ---- 1. Load + summarize ----
    print("\n[1/6] Loading dataset...")
    t0 = time.perf_counter()
    df = load_dataset()
    splits = split_data(df, random_state=args.seed)
    summary = summarize_dataset(df)
    (ARTIFACTS / "dataset_summary.json").write_text(
        json.dumps(_coerce_for_json(summary), indent=2)
    )
    print(f"  Rows: {summary['n_rows']}  Crops: {summary['crop_counts']}")
    print(f"  Train/Val/Test: {len(splits['X_train'])}/{len(splits['X_val'])}/{len(splits['X_test'])}")

    # ---- 2. Compare candidate models ----
    print("\n[2/6] 5-fold CV comparison of candidate models...")
    candidates = [m for m in ["xgboost", "hist_gb", "random_forest", "mlp", "rbf", "ridge"]
                  if m in MODEL_REGISTRY]
    print(f"  Candidates: {candidates}")
    leaderboard = compare_models(
        candidates, splits["X_train"], splits["y_train"], n_splits=5,
        random_state=args.seed,
    )
    leaderboard.to_csv(ARTIFACTS / "leaderboard.csv", index=False)
    print("\n  Leaderboard (lower RMSE is better):")
    print(leaderboard.to_string(index=False))
    plot_model_comparison(leaderboard, PLOTS / "model_comparison.png")

    best_name = leaderboard.iloc[0]["model"]
    print(f"\n  Best by CV RMSE: {best_name}")

    # ---- 3. Hyperparameter search on the winner (if it has a grid) ----
    best_params = {}
    if not args.no_search:
        grid = None
        if best_name == "hist_gb":
            grid = DEFAULT_HGB_GRID
        elif best_name == "xgboost":
            grid = DEFAULT_XGB_GRID
        if grid is not None:
            print(f"\n[3/6] Random search ({args.search_iter} iters) for {best_name}...")
            search_results = random_search(
                best_name, grid, splits["X_train"], splits["y_train"],
                n_iter=args.search_iter, n_splits=4, random_state=args.seed,
            )
            search_results.to_csv(ARTIFACTS / f"random_search_{best_name}.csv", index=False)
            # Re-derive parameter values from the original grid using the
            # row index in the grid, so we preserve native int/float types
            # (pandas promotes ints to floats when columns are mixed).
            best_row = search_results.iloc[0]
            best_params = {}
            for k in grid.keys():
                target = best_row[k]
                # Find the matching element in the original grid (preserves type).
                for candidate in grid[k]:
                    if candidate == target or (
                        isinstance(candidate, float) and abs(candidate - float(target)) < 1e-12
                    ):
                        best_params[k] = candidate
                        break
                else:
                    best_params[k] = target
            print(f"  Best params: {best_params}")
            print(f"  Best CV RMSE under search: {search_results.iloc[0]['rmse']:.4f}")
        else:
            print(f"\n[3/6] No tuning grid for {best_name}; skipping.")
    (ARTIFACTS / "best_params.json").write_text(
        json.dumps(_coerce_for_json({"model": best_name, "params": best_params}), indent=2)
    )

    # ---- 4. Refit on train+val, evaluate on test ----
    print("\n[4/6] Refitting on train+val, evaluating on held-out test set...")
    X_fit = pd.concat([splits["X_train"], splits["X_val"]], axis=0).reset_index(drop=True)
    y_fit = pd.concat([splits["y_train"], splits["y_val"]], axis=0).reset_index(drop=True)
    final_pipe = build_model(best_name, **best_params)
    final_pipe.fit(X_fit, y_fit)
    preds = final_pipe.predict(splits["X_test"])
    test_metrics = regression_metrics(splits["y_test"], preds)
    print("  Test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")
    (ARTIFACTS / "test_metrics.json").write_text(
        json.dumps(_coerce_for_json(test_metrics), indent=2)
    )

    per_crop = per_crop_report(splits["X_test"], splits["y_test"], preds)
    per_crop.to_csv(ARTIFACTS / "per_crop_metrics.csv", index=False)
    print("\n  Per-crop metrics:")
    print(per_crop.to_string(index=False))

    # ---- 5. Diagnostic plots + feature importance ----
    print("\n[5/6] Producing diagnostic plots + feature importance...")
    plot_pred_vs_actual(
        splits["y_test"], preds, PLOTS / "pred_vs_actual.png",
        crops=splits["X_test"]["crop_type"].tolist(),
        title=f"{best_name}: predicted vs actual yield (test set)",
    )
    plot_residuals(splits["y_test"], preds, PLOTS / "residuals.png")

    importances = plot_feature_importance(
        final_pipe, splits["X_test"], splits["y_test"],
        PLOTS / "feature_importance.png", n_repeats=5,
    )
    importances.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    print("  Top 5 features by permutation importance:")
    print(importances.head(5).to_string(index=False))

    if not args.no_learning_curve:
        print("\n  Computing learning curve...")
        curve = plot_learning_curve(
            lambda: build_model(best_name, **best_params),
            X_fit, y_fit, PLOTS / "learning_curve.png",
        )
        curve.to_csv(ARTIFACTS / "learning_curve.csv", index=False)
        print(curve.to_string(index=False))

    # ---- 6. Persist the predictor ----
    print("\n[6/6] Saving SurrogatePredictor...")
    predictor = build_predictor_from_pipeline(
        final_pipe, model_name=best_name,
        metadata={"test_metrics": test_metrics, "best_params": best_params,
                  "leaderboard": leaderboard.to_dict(orient="records")},
    )
    path = save_predictor(predictor)
    print(f"  Wrote {path}")
    print(f"  Total time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
