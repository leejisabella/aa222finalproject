"""Tests for the evaluation module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surrogate.data import load_dataset, split_data
from surrogate.evaluate import (
    per_crop_report,
    plot_feature_importance,
    plot_model_comparison,
    plot_pred_vs_actual,
    plot_residuals,
    regression_metrics,
)
from surrogate.models import build_model


@pytest.fixture(scope="module")
def fitted_pipeline():
    df = load_dataset()
    splits = split_data(df, random_state=42)
    pipe = build_model("hist_gb", max_iter=100, early_stopping=False)
    pipe.fit(splits["X_train"], splits["y_train"])
    return pipe, splits


def test_regression_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mape"] == pytest.approx(0.0)


def test_regression_metrics_shifted():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = y + 1.0
    m = regression_metrics(y, yhat)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(1.0)
    # Predicting mean+1 still has positive R² if shifts are constant
    assert m["r2"] > 0


def test_per_crop_report_covers_all_crops(fitted_pipeline):
    pipe, splits = fitted_pipeline
    preds = pipe.predict(splits["X_test"])
    rep = per_crop_report(splits["X_test"], splits["y_test"], preds)
    assert set(rep["crop_type"]) == {"Tomato", "Cucumber", "Lettuce", "Pepper"}
    # Reasonable per-crop R²
    for r2 in rep["r2"]:
        assert -1.0 < r2 < 1.0


def test_plots_create_files(fitted_pipeline):
    pipe, splits = fitted_pipeline
    preds = pipe.predict(splits["X_test"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        plot_pred_vs_actual(splits["y_test"], preds, tmp / "p1.png",
                             crops=splits["X_test"]["crop_type"].tolist())
        plot_residuals(splits["y_test"], preds, tmp / "p2.png")
        plot_model_comparison(
            pd.DataFrame({"model": ["a", "b"], "rmse_mean": [1.0, 2.0],
                          "rmse_std": [0.1, 0.1]}),
            tmp / "p3.png",
        )
        for f in ("p1.png", "p2.png", "p3.png"):
            assert (tmp / f).exists()
            assert (tmp / f).stat().st_size > 1000  # not an empty file


def test_feature_importance_smoke(fitted_pipeline):
    pipe, splits = fitted_pipeline
    # Subsample for speed
    X = splits["X_test"].head(200)
    y = splits["y_test"].head(200)
    with tempfile.TemporaryDirectory() as tmp:
        importances = plot_feature_importance(
            pipe, X, y, Path(tmp) / "fi.png", n_repeats=2,
        )
        assert (Path(tmp) / "fi.png").exists()
        # Returned dataframe has expected columns
        assert {"feature", "importance_mean", "importance_std"} <= set(importances.columns)
        # At least one feature has positive importance
        assert (importances["importance_mean"] > 0).any()
