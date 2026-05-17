"""Tests for the data module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surrogate.data import (
    CATEGORICAL_FEATURES,
    INDICATOR_FEATURES,
    MISSINGNESS_INDICATOR_GROUPS,
    NUMERIC_FEATURES,
    TARGET,
    build_feature_spec,
    clean_dataset,
    load_dataset,
    make_preprocessor,
    split_data,
    summarize_dataset,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_dataset()


def test_load_drops_id_and_date_columns(df: pd.DataFrame) -> None:
    for dropped in ("greenhouse_id", "planting_date", "harvest_date"):
        assert dropped not in df.columns, f"{dropped} should be dropped"


def test_load_has_all_features_and_target(df: pd.DataFrame) -> None:
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]:
        assert col in df.columns


def test_load_size_after_cleaning(df: pd.DataFrame) -> None:
    # Raw CSV is 10,400 rows. Cleaning drops 200 exact duplicates → 10,200.
    assert len(df) == 10200


def test_load_raw_keeps_duplicates() -> None:
    raw = load_dataset(clean=False)
    assert len(raw) == 10400
    assert raw.duplicated().sum() == 200


def test_clean_dataset_audit() -> None:
    raw = load_dataset(clean=False)
    cleaned, audit = clean_dataset(raw)
    assert audit["duplicates_dropped"] == 200
    assert audit["temp_rows_repaired"] == 1
    assert audit["rows_in"] == 10400
    assert audit["rows_out"] == 10200
    # No duplicates remain
    assert cleaned.duplicated().sum() == 0
    # Temp ordering invariant holds after repair
    bad = (cleaned["min_temperature_C"] > cleaned["avg_temperature_C"]) | (
        cleaned["avg_temperature_C"] > cleaned["max_temperature_C"]
    )
    # Allow NaN rows (env block missing): NaN comparisons return False so .any() is safe
    assert int(bad.sum()) == 0


def test_missingness_indicators_present(df: pd.DataFrame) -> None:
    for ind in INDICATOR_FEATURES:
        assert ind in df.columns
        assert set(df[ind].unique()) <= {0.0, 1.0}


def test_missingness_indicators_match_block_pattern(df: pd.DataFrame) -> None:
    for indicator, cols in MISSINGNESS_INDICATOR_GROUPS.items():
        # Indicator = 1 iff ALL cols in the block are NaN
        expected = df[cols].isna().all(axis=1).astype(float)
        pd.testing.assert_series_equal(
            df[indicator].rename(None), expected.rename(None),
            check_names=False,
        )


def test_missingness_indicator_counts(df: pd.DataFrame) -> None:
    # After dropping 200 duplicates, fertilizer block-missing count should still be present.
    # Raw count is 1058; duplicates could overlap with missing rows, so we just check >0.
    assert int(df["fertilizer_was_missing"].sum()) > 0
    assert int(df["env_was_missing"].sum()) > 0


def test_no_missing_target(df: pd.DataFrame) -> None:
    assert df[TARGET].isna().sum() == 0


def test_feature_spec_categories_match(df: pd.DataFrame) -> None:
    spec = build_feature_spec(df)
    assert set(spec.categorical["crop_type"]) == {"Tomato", "Cucumber", "Lettuce", "Pepper"}
    # Each crop has at least one variety
    for crop in spec.categorical["crop_type"]:
        assert len(spec.valid_varieties_per_crop[crop]) >= 1


def test_feature_spec_numeric_ranges(df: pd.DataFrame) -> None:
    spec = build_feature_spec(df)
    for col, (lo, hi) in spec.numeric_ranges.items():
        assert lo <= hi
        # Should match actual min/max
        actual_lo = float(df[col].min())
        actual_hi = float(df[col].max())
        assert lo == pytest.approx(actual_lo)
        assert hi == pytest.approx(actual_hi)


def test_split_shapes_are_stratified(df: pd.DataFrame) -> None:
    splits = split_data(df, test_size=0.2, val_size=0.2, random_state=42)
    total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
    assert total == len(df)

    # Within 1% of the global crop proportions for each split
    global_pcts = df["crop_type"].value_counts(normalize=True)
    for split_name in ("X_train", "X_val", "X_test"):
        split_pcts = splits[split_name]["crop_type"].value_counts(normalize=True)
        for crop in global_pcts.index:
            assert abs(split_pcts[crop] - global_pcts[crop]) < 0.01


def test_split_reproducible(df: pd.DataFrame) -> None:
    s1 = split_data(df, random_state=7)
    s2 = split_data(df, random_state=7)
    pd.testing.assert_frame_equal(s1["X_train"], s2["X_train"])
    pd.testing.assert_series_equal(s1["y_test"], s2["y_test"])


def test_split_validates_sizes(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        split_data(df, test_size=0.6, val_size=0.6)
    with pytest.raises(ValueError):
        split_data(df, test_size=0.0, val_size=0.2)


def test_preprocessor_handles_missing(df: pd.DataFrame) -> None:
    pp = make_preprocessor(handle_missing=True, scale_numeric=True, include_engineered=False)
    # Inject a NaN to exercise the imputer path
    sample = df.head(50).copy()
    sample.loc[0, "avg_temperature_C"] = np.nan
    Xt = pp.fit_transform(sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES + INDICATOR_FEATURES])
    assert not np.isnan(Xt).any()


def test_preprocessor_passthrough_keeps_nan(df: pd.DataFrame) -> None:
    pp = make_preprocessor(handle_missing=False, scale_numeric=False, include_engineered=False)
    sample = df.head(50).copy()
    sample.loc[0, "avg_temperature_C"] = np.nan
    Xt = pp.fit_transform(sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES + INDICATOR_FEATURES])
    # NaN in numeric branch should propagate (this is what XGBoost/HGB want)
    assert np.isnan(Xt).any()


def test_preprocessor_unknown_category_ignored(df: pd.DataFrame) -> None:
    pp = make_preprocessor(handle_missing=False, scale_numeric=False, include_engineered=False)
    train = df.head(200)[NUMERIC_FEATURES + CATEGORICAL_FEATURES + INDICATOR_FEATURES]
    pp.fit(train)
    novel = train.head(1).copy()
    novel.loc[novel.index[0], "variety"] = "MutantStrain99"
    Xt = pp.transform(novel)
    # All variety one-hots should be zero for the unknown category;
    # the row should still produce a finite vector of expected width.
    assert Xt.shape[0] == 1
    assert np.isfinite(Xt).all()


def test_summary_has_expected_keys(df: pd.DataFrame) -> None:
    summary = summarize_dataset(df)
    assert summary["n_rows"] == 10200  # after dedup
    assert set(summary["crop_counts"].keys()) == {"Tomato", "Cucumber", "Lettuce", "Pepper"}
