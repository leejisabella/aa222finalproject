"""Tests for the inference API and persistence."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surrogate.api import (
    SurrogatePredictor,
    build_predictor_from_pipeline,
    load_predictor,
    save_predictor,
)
from surrogate.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_dataset, split_data
from surrogate.models import build_model


@pytest.fixture(scope="module")
def predictor() -> SurrogatePredictor:
    df = load_dataset()
    splits = split_data(df, random_state=42)
    # Train a small but legitimate model so tests exercise real predictions.
    pipe = build_model("hist_gb", max_iter=100, early_stopping=False)
    pipe.fit(splits["X_train"], splits["y_train"])
    return build_predictor_from_pipeline(pipe, model_name="hist_gb")


@pytest.fixture
def valid_config(predictor: SurrogatePredictor) -> dict:
    # A realistic Tomato Beefsteak config in the middle of each range.
    cfg = {}
    for col, (lo, hi) in predictor.bounds().items():
        cfg[col] = (lo + hi) / 2
    cfg["crop_type"] = "Tomato"
    cfg["variety"] = "Beefsteak"
    return cfg


def test_predict_one_returns_float(predictor, valid_config):
    y = predictor.predict_one(valid_config)
    assert isinstance(y, float)
    assert np.isfinite(y)


def test_predict_one_in_plausible_range(predictor, valid_config):
    # Tomatoes range 6-31 kg/m^2; mid-config should land somewhere
    # in the realistic spread, not at -50 or 500.
    y = predictor.predict_one(valid_config)
    assert 0 < y < 60


def test_predict_batch_shape(predictor, valid_config):
    configs = [valid_config, valid_config, valid_config]
    preds = predictor.predict_batch(configs)
    assert preds.shape == (3,)


def test_predict_batch_dataframe(predictor, valid_config):
    df = pd.DataFrame([valid_config] * 4)
    preds = predictor.predict_batch(df)
    assert preds.shape == (4,)


def test_predict_missing_column_raises(predictor, valid_config):
    bad = dict(valid_config)
    del bad["avg_temperature_C"]
    with pytest.raises(ValueError, match="Missing required feature"):
        predictor.predict_one(bad)


def test_predict_extra_columns_ignored(predictor, valid_config):
    extra = dict(valid_config)
    extra["completely_random_column"] = 12345
    # Should not raise; extras are silently dropped.
    y = predictor.predict_one(extra)
    assert np.isfinite(y)


def test_strict_validation_rejects_invalid_crop(predictor, valid_config):
    bad = dict(valid_config)
    bad["crop_type"] = "Mango"
    with pytest.raises(ValueError, match="unknown levels"):
        predictor.predict_one(bad, strict=True)


def test_strict_validation_rejects_out_of_range(predictor, valid_config):
    bad = dict(valid_config)
    bad["avg_temperature_C"] = 500.0
    with pytest.raises(ValueError, match="above training range"):
        predictor.predict_one(bad, strict=True)


def test_strict_validation_rejects_crop_variety_mismatch(predictor, valid_config):
    bad = dict(valid_config)
    bad["crop_type"] = "Lettuce"
    bad["variety"] = "Beefsteak"  # tomato variety on lettuce
    with pytest.raises(ValueError, match="invalid variety"):
        predictor.predict_one(bad, strict=True)


def test_nonstrict_warns_but_predicts(predictor, valid_config):
    bad = dict(valid_config)
    bad["avg_temperature_C"] = 500.0  # extrapolation
    # Should not raise; the OneHotEncoder + tree handle this even though
    # the prediction is meaningless.
    y = predictor.predict_one(bad, strict=False)
    assert np.isfinite(y)


def test_bounds_match_feature_spec(predictor):
    bounds = predictor.bounds()
    for col in NUMERIC_FEATURES:
        assert col in bounds
        lo, hi = bounds[col]
        assert lo < hi


def test_categorical_levels(predictor):
    levels = predictor.categorical_levels()
    assert set(levels["crop_type"]) == {"Tomato", "Cucumber", "Lettuce", "Pepper"}
    assert "Beefsteak" in levels["variety"]


def test_valid_varieties_for_crop(predictor):
    tomato_varieties = predictor.valid_varieties_for("Tomato")
    assert "Beefsteak" in tomato_varieties
    assert "Iceberg" not in tomato_varieties  # lettuce variety
    pepper_varieties = predictor.valid_varieties_for("Pepper")
    assert "Bell" in pepper_varieties


def test_save_load_roundtrip(predictor, valid_config):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        save_predictor(predictor, path)
        assert path.exists()
        loaded = load_predictor(path)
        assert loaded.model_name == predictor.model_name
        # Predictions match exactly after pickling
        y0 = predictor.predict_one(valid_config)
        y1 = loaded.predict_one(valid_config)
        assert y0 == pytest.approx(y1)


def test_predict_one_accepts_series(predictor, valid_config):
    s = pd.Series(valid_config)
    y = predictor.predict_one(s)
    assert np.isfinite(y)


def test_predict_batch_matches_individual_calls(predictor, valid_config):
    configs = [dict(valid_config) for _ in range(5)]
    # Vary one column so predictions differ
    for i, cfg in enumerate(configs):
        cfg["co2_ppm"] = 500 + i * 100
    batch = predictor.predict_batch(configs)
    individual = np.array([predictor.predict_one(cfg) for cfg in configs])
    np.testing.assert_allclose(batch, individual, rtol=1e-9)
