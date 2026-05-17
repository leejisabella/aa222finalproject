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


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw CSV and drop ID/date columns.

    Returns a DataFrame with one row per harvest, columns in
    FEATURE_SPEC.categorical + FEATURE_SPEC.numeric_ranges + [TARGET].
    Missing values in feature columns are left as NaN; gradient-boosted
    models consume them directly, and the sklearn pipeline imputes for
    models that cannot.
    """
    src = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found at {src}")
    df = pd.read_csv(src)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    # Target must be present; drop rows where it is missing.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
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


def make_preprocessor(
    handle_missing: bool = True,
    scale_numeric: bool = True,
    sparse: bool = False,
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
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

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
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
