"""Inference API consumed by downstream optimizers.

The optimizers (Bayesian Opt, GA, CMA-ES, PSO) need to evaluate
candidate greenhouse configurations at high rates inside their inner
loops. This module wraps the trained pipeline behind:

  - `SurrogatePredictor.predict_one(config_dict) -> float`
  - `SurrogatePredictor.predict_batch(configs_df) -> np.ndarray`
  - `SurrogatePredictor.bounds()` so the optimizers can sample inside
    the training distribution and not extrapolate into nonsense.

Predictors are pickled to artifacts/ so optimizers don't pay the
training cost on every run.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from surrogate.data import (
    CATEGORICAL_FEATURES,
    INDICATOR_FEATURES,
    MISSINGNESS_INDICATOR_GROUPS,
    NUMERIC_FEATURES,
    FeatureSpec,
    build_feature_spec,
    load_dataset,
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "surrogate.pkl"

ConfigLike = Union[Mapping[str, Any], pd.Series]


@dataclass
class SurrogatePredictor:
    """Trained surrogate + the feature schema needed to validate inputs."""

    pipeline: Any  # an sklearn Pipeline
    feature_spec: FeatureSpec
    model_name: str = "unknown"
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    # ----------------------- validation helpers -----------------------

    def _coerce_to_frame(self, configs: Union[ConfigLike, Iterable[ConfigLike], pd.DataFrame]
                         ) -> pd.DataFrame:
        if isinstance(configs, pd.DataFrame):
            df = configs.copy()
        elif isinstance(configs, Mapping):
            df = pd.DataFrame([dict(configs)])
        elif isinstance(configs, pd.Series):
            df = pd.DataFrame([configs.to_dict()])
        else:
            # iterable of dict/series
            rows = []
            for item in configs:
                if isinstance(item, Mapping):
                    rows.append(dict(item))
                elif isinstance(item, pd.Series):
                    rows.append(item.to_dict())
                else:
                    raise TypeError(f"Unsupported config item type: {type(item)}")
            df = pd.DataFrame(rows)

        missing = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        # Auto-fill missingness indicators from the actual NaN pattern in
        # the supplied numeric columns. Optimizers supply fully-specified
        # configs (no NaNs), so these indicators end up as 0 — which
        # matches the "measurement was recorded" branch the model trained on.
        for indicator, cols in MISSINGNESS_INDICATOR_GROUPS.items():
            if indicator not in df.columns:
                df[indicator] = df[cols].isna().all(axis=1).astype(float)
        # Reorder + drop extras so the preprocessor sees its expected schema.
        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + INDICATOR_FEATURES]

    def _validate(self, df: pd.DataFrame, strict: bool) -> List[str]:
        """Return list of human-readable warnings/errors. Raise if strict."""
        problems: List[str] = []
        for col in CATEGORICAL_FEATURES:
            valid = set(self.feature_spec.categorical[col])
            bad = set(df[col].dropna().astype(str).unique()) - valid
            if bad:
                problems.append(f"{col} has unknown levels {sorted(bad)}; valid: {sorted(valid)}")
        for col in NUMERIC_FEATURES:
            lo, hi = self.feature_spec.numeric_ranges[col]
            # 10% slack — optimizers may push slightly past training extremes;
            # beyond that the surrogate is extrapolating.
            slack = 0.1 * max(abs(lo), abs(hi), 1.0)
            below = df[col].dropna() < (lo - slack)
            above = df[col].dropna() > (hi + slack)
            if below.any() or above.any():
                problems.append(
                    f"{col} has {int(below.sum())} below and {int(above.sum())} above "
                    f"training range [{lo:.2f}, {hi:.2f}] (+/-10%)"
                )

        # Crop / variety consistency: e.g. "Bell" only makes sense for Pepper.
        for crop, varieties in self.feature_spec.valid_varieties_per_crop.items():
            mask = df["crop_type"] == crop
            bad = set(df.loc[mask, "variety"].dropna().astype(str).unique()) - set(varieties)
            if bad:
                problems.append(
                    f"crop_type={crop} paired with invalid variety {sorted(bad)}; "
                    f"valid: {varieties}"
                )

        if strict and problems:
            raise ValueError("Config validation failed:\n  - " + "\n  - ".join(problems))
        return problems

    # ----------------------- public API -----------------------

    def predict_one(self, config: ConfigLike, strict: bool = False) -> float:
        df = self._coerce_to_frame(config)
        self._validate(df, strict=strict)
        return float(self.pipeline.predict(df)[0])

    def predict_batch(self, configs: Union[Iterable[ConfigLike], pd.DataFrame],
                      strict: bool = False) -> np.ndarray:
        df = self._coerce_to_frame(configs)
        self._validate(df, strict=strict)
        return np.asarray(self.pipeline.predict(df), dtype=float)

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Numeric feature bounds for optimizer search-space construction."""
        return dict(self.feature_spec.numeric_ranges)

    def categorical_levels(self) -> Dict[str, List[str]]:
        """Discrete choices for the optimizer (crop_type, variety)."""
        return {k: list(v) for k, v in self.feature_spec.categorical.items()}

    def valid_varieties_for(self, crop_type: str) -> List[str]:
        return list(self.feature_spec.valid_varieties_per_crop[crop_type])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_predictor(predictor: SurrogatePredictor, path: Optional[Path] = None) -> Path:
    target = Path(path) if path is not None else DEFAULT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump(predictor, f)
    return target


def load_predictor(path: Optional[Path] = None) -> SurrogatePredictor:
    src = Path(path) if path is not None else DEFAULT_MODEL_PATH
    with src.open("rb") as f:
        return pickle.load(f)


def build_predictor_from_pipeline(
    pipeline: Any,
    model_name: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> SurrogatePredictor:
    spec = build_feature_spec(load_dataset())
    return SurrogatePredictor(
        pipeline=pipeline,
        feature_spec=spec,
        model_name=model_name,
        metadata=dict(metadata or {}),
    )
