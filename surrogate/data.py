"""Data loading and preprocessing for the greenhouse yield dataset.

The Kaggle dataset has 10,400 rows of greenhouse harvest records with
mixed continuous and categorical features and ~5% missing values in
several environmental columns. The yield (kg/m^2) is the regression
target the surrogate must predict so the downstream optimizers can
evaluate candidate greenhouse configurations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "greenhouse_crop_yields.csv"

TARGET = "yield_kg_per_m2"

CATEGORICAL_FEATURES: List[str] = ["crop_type", "variety"]

NUMERIC_FEATURES: List[str] = [
    "days_to_maturity",
    "avg_temperature_C",
    "min_temperature_C",
    "max_temperature_C",
    "humidity_percent",
    "co2_ppm",
    "light_intensity_lux",
    "photoperiod_hours",
    "irrigation_mm",
    "fertilizer_N_kg_ha",
    "fertilizer_P_kg_ha",
    "fertilizer_K_kg_ha",
    "pest_severity",
    "soil_pH",
]

# Columns dropped before modeling. greenhouse_id is just a site identifier
# with no causal meaning; the dates are redundant once days_to_maturity
# captures growth duration and including them risks date-leakage between
# train/test splits.
DROP_COLUMNS: List[str] = ["greenhouse_id", "planting_date", "harvest_date"]

# Columns whose values are missing in informative blocks (e.g. NPK is
# always missing as a group → "no fertilizer recorded"; the 5 env
# columns are always missing as a group → "site without env sensors").
# We add a single 0/1 indicator per block so the non-tree baselines
# (MLP, RBF, Ridge) can learn that NaN means something rather than
# silently absorbing a median-imputed value.
MISSINGNESS_INDICATOR_GROUPS: Dict[str, List[str]] = {
    "fertilizer_was_missing": [
        "fertilizer_N_kg_ha",
        "fertilizer_P_kg_ha",
        "fertilizer_K_kg_ha",
    ],
    "env_was_missing": [
        "avg_temperature_C",
        "min_temperature_C",
        "max_temperature_C",
        "humidity_percent",
        "light_intensity_lux",
    ],
}
INDICATOR_FEATURES: List[str] = list(MISSINGNESS_INDICATOR_GROUPS.keys())


@dataclass(frozen=True)
class FeatureSpec:
    """Schema + valid ranges for the surrogate's inputs.

    Ranges come from the empirical min/max of the training data and
    are used by the inference API to bound optimizer search and by
    tests to validate that synthetic inputs are realistic.
    """

    categorical: Dict[str, List[str]] = field(default_factory=dict)
    numeric_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    valid_varieties_per_crop: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def numeric_names(self) -> List[str]:
        return list(self.numeric_ranges.keys())

    @property
    def categorical_names(self) -> List[str]:
        return list(self.categorical.keys())


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply data cleaning. Returns the cleaned frame plus an audit dict.

    Cleaning steps:
      1. Drop exact duplicate rows. The raw CSV has 200 fully repeated
         harvest records; leaving them in lets the same row land in both
         train and test, biasing test metrics optimistically.
      2. Repair temperature-consistency violations. A single row has
         avg > max by 0.1°C (rounding artifact); we clip avg into
         [min, max] instead of dropping the row.
      3. Add per-block missingness indicators. Fertilizer N/P/K are
         missing as a group ("no fertilizer recorded") and the 5 env
         columns are missing as a group ("no env sensors"). The
         indicators let non-tree models learn what NaN means.
    """
    audit: Dict[str, int] = {}
    audit["rows_in"] = len(df)

    # 1. Drop exact duplicates.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    audit["duplicates_dropped"] = before - len(df)

    # 2. Repair temperature ordering.
    bad_mask = (df["min_temperature_C"] > df["avg_temperature_C"]) | (
        df["avg_temperature_C"] > df["max_temperature_C"]
    )
    audit["temp_rows_repaired"] = int(bad_mask.sum())
    if bad_mask.any():
        df.loc[bad_mask, "avg_temperature_C"] = df.loc[bad_mask, ["min_temperature_C", "max_temperature_C"]].mean(axis=1)

    # 3. Add missingness indicators (computed BEFORE any imputation).
    for indicator_name, cols in MISSINGNESS_INDICATOR_GROUPS.items():
        df[indicator_name] = df[cols].isna().all(axis=1).astype(float)

    audit["rows_out"] = len(df)
    return df, audit


def load_dataset(
    path: Optional[Path] = None,
    clean: bool = True,
    engineer: bool = True,
) -> pd.DataFrame:
    """Load the raw CSV, drop ID/date columns, optionally clean + engineer.

    With `clean=True` (default): drops 200 duplicate rows, repairs the
    one temperature-ordering violation, and adds the missingness
    indicator features.
    With `engineer=True` (default): adds domain-derived features (VPD,
    DLI, GDD, fertilizer ratios, season totals) from `features.py`.

    `engineer=False` is used for the A/B benchmark that quantifies how
    much the engineered features help.
    """
    src = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found at {src}")
    df = pd.read_csv(src)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if clean:
        df, _ = clean_dataset(df)
    if engineer:
        # Imported here to avoid a circular import (features uses no data helpers).
        from surrogate.features import add_engineered_features
        df = add_engineered_features(df)
    return df


def build_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    """Derive feature schema (categories and numeric ranges) from data."""
    categorical = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}
    numeric_ranges = {
        col: (float(df[col].min()), float(df[col].max())) for col in NUMERIC_FEATURES
    }
    valid_varieties_per_crop = {
        crop: sorted(df.loc[df["crop_type"] == crop, "variety"].dropna().unique().tolist())
        for crop in categorical["crop_type"]
    }
    return FeatureSpec(
        categorical=categorical,
        numeric_ranges=numeric_ranges,
        valid_varieties_per_crop=valid_varieties_per_crop,
    )


# Built lazily so importing the module does not require the dataset.
_FEATURE_SPEC_CACHE: Dict[str, FeatureSpec] = {}


def _get_default_spec() -> FeatureSpec:
    key = str(DEFAULT_DATA_PATH)
    if key not in _FEATURE_SPEC_CACHE:
        _FEATURE_SPEC_CACHE[key] = build_feature_spec(load_dataset())
    return _FEATURE_SPEC_CACHE[key]


class _LazyFeatureSpec:
    """Module-level proxy so `FEATURE_SPEC.numeric_names` works without
    forcing dataset load at import time (which would break tests on
    machines without the CSV)."""

    def __getattr__(self, item):
        return getattr(_get_default_spec(), item)


FEATURE_SPEC = _LazyFeatureSpec()


def get_numeric_features(include_engineered: bool = True) -> List[str]:
    """Numeric feature list, optionally including engineered ones.

    Held separate from the module-level NUMERIC_FEATURES so the A/B
    benchmark can build a preprocessor that intentionally excludes the
    engineered features.
    """
    cols = list(NUMERIC_FEATURES)
    if include_engineered:
        from surrogate.features import ENGINEERED_NUMERIC_FEATURES
        cols += ENGINEERED_NUMERIC_FEATURES
    return cols


def make_preprocessor(
    handle_missing: bool = True,
    scale_numeric: bool = True,
    sparse: bool = False,
    include_engineered: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer for models that need imputed/encoded inputs.

    - Numeric: median-impute (if `handle_missing`) then standard-scale
      (if `scale_numeric`). Median is robust to the outliers visible in
      pest_severity and fertilizer columns.
    - Categorical: most-frequent impute (rare, since these are never
      missing in practice) then one-hot encode with unknown categories
      ignored at predict time so the optimizers can still produce a
      result for any in-spec input.
    """
    numeric_steps = []
    if handle_missing:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    cat_steps = []
    if handle_missing:
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    cat_steps.append(
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=sparse),
        )
    )
    categorical_pipeline = Pipeline(cat_steps)

    numeric_cols = get_numeric_features(include_engineered=include_engineered)
    # Indicators are 0/1 and never NaN — pass through unchanged.
    transformers = [
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
    transformers.append(("ind", "passthrough", INDICATOR_FEATURES))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Stratified train/val/test split (by crop_type).

    Stratification keeps the four crops' proportions matched across
    splits so per-crop metrics on the held-out set are meaningful.
    Default: 60/20/20.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be in (0,1) and sum to < 1")

    strat = df["crop_type"]
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )
    # Adjust val_size relative to the remaining fraction.
    rel_val = val_size / (1.0 - test_size)
    strat_tv = train_val["crop_type"]
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=random_state, stratify=strat_tv
    )

    feature_cols = get_numeric_features(include_engineered=True) + CATEGORICAL_FEATURES + INDICATOR_FEATURES
    # Keep only columns that actually exist (handles engineer=False case).
    feature_cols = [c for c in feature_cols if c in df.columns]
    return {
        "X_train": train[feature_cols].reset_index(drop=True),
        "y_train": train[TARGET].reset_index(drop=True),
        "X_val": val[feature_cols].reset_index(drop=True),
        "y_val": val[TARGET].reset_index(drop=True),
        "X_test": test[feature_cols].reset_index(drop=True),
        "y_test": test[TARGET].reset_index(drop=True),
    }


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Diagnostic summary used by the training script's report."""
    return {
        "n_rows": int(len(df)),
        "crop_counts": df["crop_type"].value_counts().to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "target_stats": df.groupby("crop_type")[TARGET].describe().round(3).to_dict(),
    }
