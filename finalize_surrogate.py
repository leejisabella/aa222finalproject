"""Train and save the FINAL surrogate: CatBoost + engineered features.

Produces:
  artifacts/surrogate.pkl                Pickled SurrogatePredictor (CatBoost)
  artifacts/final_metrics.json           Comprehensive held-out test metrics
  artifacts/final_comparison.csv         Side-by-side against other models
  artifacts/plots/final_*.png            Updated diagnostic plots
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from surrogate.api import build_predictor_from_pipeline, save_predictor
from surrogate.data import load_dataset, split_data
from surrogate.evaluate import (
    per_crop_report,
    plot_feature_importance,
    plot_pred_vs_actual,
    plot_residuals,
    regression_metrics,
)
from surrogate.models import build_model
from surrogate.train import DEFAULT_CATBOOST_GRID, random_search

ARTIFACTS = Path("artifacts")
PLOTS = ARTIFACTS / "plots"


def comprehensive_metrics(y_true, y_pred) -> dict:
    """All metrics the writeup and the optimizers might care about."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    base = regression_metrics(y_true, y_pred)
    # Rank-quality: how well does the surrogate ORDER configs? (BO/GA care.)
    sp, _ = spearmanr(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)
    # Top-K precision: of the K configs the model thinks are best,
    # how many are actually in the true top-K?
    def top_k_precision(k: int) -> float:
        idx_pred = np.argsort(-y_pred)[:k]
        idx_true = set(np.argsort(-y_true)[:k].tolist())
        return float(len(set(idx_pred.tolist()) & idx_true) / k)
    return {
        **base,
        "bias_mean_residual": float(err.mean()),
        "abs_err_p50": float(np.percentile(abs_err, 50)),
        "abs_err_p95": float(np.percentile(abs_err, 95)),
        "abs_err_p99": float(np.percentile(abs_err, 99)),
        "abs_err_max": float(abs_err.max()),
        "spearman_rho": float(sp),
        "kendall_tau": float(kt),
        "top10_precision": top_k_precision(10),
        "top50_precision": top_k_precision(50),
        "top100_precision": top_k_precision(100),
    }


def benchmark_latency(predictor, n_calls: int = 1000) -> dict:
    """Throughput numbers for the optimizer inner loop."""
    bounds = predictor.bounds()
    levels = predictor.categorical_levels()
    rng = np.random.default_rng(0)
    cfgs = []
    for _ in range(n_calls):
        cfg = {col: float(rng.uniform(lo, hi)) for col, (lo, hi) in bounds.items()}
        crop = rng.choice(levels["crop_type"])
        cfg["crop_type"] = crop
        cfg["variety"] = predictor.valid_varieties_for(crop)[0]
        cfgs.append(cfg)

    # Single-row latency (optimizer inner loop pattern)
    t0 = time.perf_counter()
    for cfg in cfgs[:200]:
        predictor.predict_one(cfg)
    single = (time.perf_counter() - t0) / 200

    # Batched latency
    t0 = time.perf_counter()
    predictor.predict_batch(cfgs)
    batched = (time.perf_counter() - t0) / n_calls
    return {"single_call_seconds": single, "batched_per_item_seconds": batched}


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    print("=" * 70)
    print("Final surrogate: CatBoost + engineered features")
    print("=" * 70)

    # 1. Load (cleaned, engineered) and split.
    df = load_dataset(clean=True, engineer=True)
    splits = split_data(df, random_state=42)

    # 2. Small CatBoost random search on the train split.
    print("\n[1/4] Random search (15 iters) for CatBoost hyperparameters...")
    search = random_search(
        "catboost", DEFAULT_CATBOOST_GRID,
        splits["X_train"], splits["y_train"],
        n_iter=15, n_splits=4, random_state=42,
    )
    search.to_csv(ARTIFACTS / "random_search_catboost.csv", index=False)
    best_params = {}
    grid_keys = list(DEFAULT_CATBOOST_GRID.keys())
    best_row = search.iloc[0]
    for k in grid_keys:
        target = best_row[k]
        for candidate in DEFAULT_CATBOOST_GRID[k]:
            if candidate == target or (
                isinstance(candidate, float) and abs(candidate - float(target)) < 1e-12
            ):
                best_params[k] = candidate
                break
    print(f"  Best params: {best_params}")
    print(f"  Best CV RMSE: {best_row['rmse']:.4f}")

    # 3. Refit on train+val with best params, evaluate on held-out test.
    print("\n[2/4] Refitting on train+val, evaluating on test...")
    X_fit = pd.concat([splits["X_train"], splits["X_val"]], axis=0).reset_index(drop=True)
    y_fit = pd.concat([splits["y_train"], splits["y_val"]], axis=0).reset_index(drop=True)
    final_pipe = build_model("catboost", verbose=0, **best_params)
    final_pipe.fit(X_fit, y_fit)
    preds = final_pipe.predict(splits["X_test"])

    metrics = comprehensive_metrics(splits["y_test"], preds)
    per_crop = per_crop_report(splits["X_test"], splits["y_test"], preds)
    print("\n  Test metrics:")
    for k, v in metrics.items():
        print(f"    {k:>26}: {v:.4f}")
    print("\n  Per-crop:")
    print("    " + per_crop.to_string(index=False).replace("\n", "\n    "))

    # 4. Save predictor + benchmark latency.
    print("\n[3/4] Saving SurrogatePredictor and benchmarking latency...")
    predictor = build_predictor_from_pipeline(
        final_pipe, model_name="catboost",
        metadata={"test_metrics": metrics, "best_params": best_params,
                  "per_crop": per_crop.to_dict(orient="records")},
        uses_engineered_features=True,
    )
    save_predictor(predictor)
    latency = benchmark_latency(predictor, n_calls=500)
    print(f"  Single-call latency: {latency['single_call_seconds']*1e3:.2f} ms")
    print(f"  Batched per-item:   {latency['batched_per_item_seconds']*1e6:.2f} us")

    # 5. Diagnostic plots.
    print("\n[4/4] Producing diagnostic plots...")
    plot_pred_vs_actual(
        splits["y_test"], preds, PLOTS / "final_pred_vs_actual.png",
        crops=splits["X_test"]["crop_type"].tolist(),
        title="CatBoost + engineered features: predicted vs actual yield (test)",
    )
    plot_residuals(splits["y_test"], preds, PLOTS / "final_residuals.png")
    importances = plot_feature_importance(
        final_pipe, splits["X_test"], splits["y_test"],
        PLOTS / "final_feature_importance.png", n_repeats=5, top_k=15,
    )
    importances.to_csv(ARTIFACTS / "final_feature_importance.csv", index=False)
    print("  Top 8 features by permutation importance:")
    print("    " + importances.head(8).to_string(index=False).replace("\n", "\n    "))

    # 6. Persist final metrics.
    final = {
        "model": "catboost",
        "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in best_params.items()},
        "test_metrics": metrics,
        "per_crop_metrics": per_crop.to_dict(orient="records"),
        "latency": latency,
        "n_train": int(len(splits["X_train"])),
        "n_val": int(len(splits["X_val"])),
        "n_test": int(len(splits["X_test"])),
    }
    (ARTIFACTS / "final_metrics.json").write_text(json.dumps(final, indent=2, default=str))
    print(f"\nWrote artifacts/final_metrics.json")
    print(f"Wrote artifacts/surrogate.pkl (load with surrogate.api.load_predictor())")


if __name__ == "__main__":
    main()
