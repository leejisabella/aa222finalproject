"""Tests for engineered features and per-crop ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surrogate.data import load_dataset, split_data
from surrogate.features import (
    ENGINEERED_NUMERIC_FEATURES,
    add_engineered_features,
    daily_light_integral_mol,
    growing_degree_days,
    vapor_pressure_deficit_kpa,
)
from surrogate.models import CATBOOST_AVAILABLE, build_model, build_per_crop


def test_vpd_at_full_humidity_is_zero():
    # 100% RH → no vapor pressure deficit, regardless of temperature.
    t = pd.Series([10.0, 20.0, 30.0])
    h = pd.Series([100.0, 100.0, 100.0])
    vpd = vapor_pressure_deficit_kpa(t, h)
    np.testing.assert_allclose(vpd.values, 0.0, atol=1e-12)


def test_vpd_increases_with_temperature_at_fixed_rh():
    # Hotter air holds more water; VPD at fixed RH grows with T.
    t = pd.Series([15.0, 25.0, 35.0])
    h = pd.Series([60.0, 60.0, 60.0])
    vpd = vapor_pressure_deficit_kpa(t, h)
    assert vpd.iloc[2] > vpd.iloc[1] > vpd.iloc[0]


def test_vpd_realistic_range_for_greenhouse():
    # Tomato/cucumber greenhouse: ~25°C, 70% RH → VPD around 0.95 kPa.
    vpd = float(vapor_pressure_deficit_kpa(pd.Series([25.0]), pd.Series([70.0])).iloc[0])
    assert 0.7 < vpd < 1.2


def test_dli_realistic_range():
    # 30,000 lux × 16h photoperiod → roughly 32 mol/m^2/day (sunny greenhouse).
    dli = float(daily_light_integral_mol(pd.Series([30000.0]), pd.Series([16.0])).iloc[0])
    assert 25 < dli < 40


def test_gdd_zero_below_base_temp():
    # 5°C is below the 10°C base; GDD should be exactly zero.
    gdd = growing_degree_days(pd.Series([5.0]), pd.Series([60.0]))
    assert float(gdd.iloc[0]) == 0.0


def test_add_engineered_features_no_mutation():
    df = load_dataset(engineer=False).head(20)
    original_cols = set(df.columns)
    out = add_engineered_features(df)
    # Original frame is unchanged
    assert set(df.columns) == original_cols
    # All engineered cols are present in the output
    for c in ENGINEERED_NUMERIC_FEATURES:
        assert c in out.columns


def test_add_engineered_features_handles_missing_fertilizer():
    df = load_dataset(engineer=False).head(50).copy()
    # Force a row where P is missing → n_to_p_ratio must be NaN, not inf
    df.loc[df.index[0], "fertilizer_P_kg_ha"] = np.nan
    out = add_engineered_features(df)
    assert pd.isna(out.loc[df.index[0], "n_to_p_ratio"])
    # And no inf anywhere
    eng_cols = ENGINEERED_NUMERIC_FEATURES
    assert not np.isinf(out[eng_cols].to_numpy(dtype=float)).any()


# ------------------------- per-crop ensemble -------------------------


@pytest.fixture(scope="module")
def splits():
    return split_data(load_dataset(), random_state=42)


def test_per_crop_dispatch_uses_correct_submodel(splits):
    pc = build_per_crop("hist_gb", max_iter=50, early_stopping=False)
    pc.fit(splits["X_train"], splits["y_train"])
    assert set(pc.crops_) == {"Tomato", "Cucumber", "Lettuce", "Pepper"}
    # A Tomato config and a Lettuce config get routed to different models.
    tomato = splits["X_test"][splits["X_test"]["crop_type"] == "Tomato"].head(1)
    lettuce = splits["X_test"][splits["X_test"]["crop_type"] == "Lettuce"].head(1)
    p_t = pc.predict(tomato)[0]
    p_l = pc.predict(lettuce)[0]
    # Tomato yields are roughly 2x lettuce yields — sanity check that the
    # ensemble produced crop-appropriate predictions.
    assert p_t > p_l


def test_per_crop_handles_unseen_crop(splits):
    pc = build_per_crop("hist_gb", max_iter=50, early_stopping=False)
    # Train on only 3 of the 4 crops, predict on the held-out crop.
    mask = splits["X_train"]["crop_type"] != "Pepper"
    X_tr = splits["X_train"][mask]
    y_tr = splits["y_train"][mask]
    pc.fit(X_tr, y_tr)
    pepper = splits["X_test"][splits["X_test"]["crop_type"] == "Pepper"].head(3)
    # Should fall back to the global model rather than crash.
    preds = pc.predict(pepper)
    assert preds.shape == (3,)
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost not installed")
def test_catboost_fits_and_predicts(splits):
    cb = build_model("catboost", iterations=50, verbose=0)
    cb.fit(splits["X_train"], splits["y_train"])
    preds = cb.predict(splits["X_test"].head(10))
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()
    # CatBoost predictions are in a plausible yield range.
    assert (preds > 0).all() and (preds < 50).all()
