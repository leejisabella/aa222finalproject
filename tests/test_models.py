"""Tests for the model zoo + training helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surrogate.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_dataset,
    split_data,
)
from surrogate.models import MODEL_REGISTRY, XGBOOST_AVAILABLE, build_model
from surrogate.train import (
    compare_models,
    cross_validate_model,
    random_search,
)


@pytest.fixture(scope="module")
def splits():
    df = load_dataset()
    return split_data(df, random_state=42)


@pytest.fixture(scope="module")
def small_train(splits):
    # Subsample for fast tests
    rng = np.random.default_rng(0)
    idx = rng.choice(len(splits["X_train"]), size=600, replace=False)
    return splits["X_train"].iloc[idx].reset_index(drop=True), \
           splits["y_train"].iloc[idx].reset_index(drop=True)


# ------------------------- model construction -------------------------


def test_registry_contains_minimum_models():
    for name in ("hist_gb", "random_forest", "mlp", "rbf", "ridge"):
        assert name in MODEL_REGISTRY


def test_build_model_unknown_raises():
    with pytest.raises(KeyError):
        build_model("does_not_exist")


@pytest.mark.parametrize("name", ["hist_gb", "ridge", "rbf"])
def test_models_fit_and_predict(name, small_train, splits):
    X_tr, y_tr = small_train
    pipe = build_model(name)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(splits["X_test"].head(50))
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()


def test_hist_gb_handles_nan_natively(small_train, splits):
    X_tr, y_tr = small_train
    # Inject NaNs into the training set
    X_tr = X_tr.copy()
    X_tr.loc[0:10, "avg_temperature_C"] = np.nan
    pipe = build_model("hist_gb")
    pipe.fit(X_tr, y_tr)
    # And NaNs at prediction time
    X_pred = splits["X_test"].head(5).copy()
    X_pred.loc[X_pred.index[0], "humidity_percent"] = np.nan
    preds = pipe.predict(X_pred)
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not importable")
def test_xgboost_handles_nan_natively(small_train, splits):
    X_tr, y_tr = small_train
    X_tr = X_tr.copy()
    X_tr.loc[0:10, "avg_temperature_C"] = np.nan
    pipe = build_model("xgboost")
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(splits["X_test"].head(5))
    assert np.isfinite(preds).all()


# ------------------------- baseline performance -------------------------


def test_hist_gb_beats_ridge_baseline(splits):
    # Boosting should clearly beat a linear baseline on this dataset.
    cv_hgb = cross_validate_model(
        "hist_gb", splits["X_train"], splits["y_train"], n_splits=3
    )
    cv_lin = cross_validate_model(
        "ridge", splits["X_train"], splits["y_train"], n_splits=3
    )
    assert cv_hgb.mean_rmse < cv_lin.mean_rmse, (
        f"HGB RMSE {cv_hgb.mean_rmse:.3f} should beat ridge {cv_lin.mean_rmse:.3f}"
    )


def test_hist_gb_achieves_target_r2(splits):
    # Loose sanity check: with default params, R² on CV should be >0.75.
    cv = cross_validate_model("hist_gb", splits["X_train"], splits["y_train"], n_splits=3)
    assert cv.mean_r2 > 0.75


def test_compare_models_sorted(splits):
    lb = compare_models(["ridge", "hist_gb"], splits["X_train"], splits["y_train"], n_splits=3)
    # Lower RMSE first.
    assert lb["rmse_mean"].is_monotonic_increasing
    assert set(lb["model"]) == {"ridge", "hist_gb"}


def test_cv_result_aggregates_correctly():
    from surrogate.train import CVResult, FoldResult
    cv = CVResult("toy", [
        FoldResult(0, 1.0, 0.5, 0.9, 0.1),
        FoldResult(1, 2.0, 0.5, 0.9, 0.1),
    ])
    assert cv.mean_rmse == 1.5
    assert cv.std_rmse == 0.5


def test_random_search_smoke(small_train):
    X_tr, y_tr = small_train
    grid = {"max_iter": [50, 100], "learning_rate": [0.1, 0.2]}
    results = random_search("hist_gb", grid, X_tr, y_tr, n_iter=3, n_splits=2)
    assert len(results) >= 1
    # Returned sorted by RMSE
    assert results["rmse"].is_monotonic_increasing
