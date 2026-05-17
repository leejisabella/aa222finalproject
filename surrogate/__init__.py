"""TOMATO surrogate model package.

Provides a fitted regression model that maps greenhouse configurations
(crop_type, env, fertilizer, ...) to predicted yield (kg/m^2). Downstream
optimizers (Bayesian Opt, GA, CMA-ES, PSO) call `predict_yield` from `api`.
"""

"""TOMATO surrogate package — convenience re-exports."""

from surrogate.api import (
    SurrogatePredictor,
    build_predictor_from_pipeline,
    load_predictor,
    save_predictor,
)
from surrogate.data import (
    CATEGORICAL_FEATURES,
    FEATURE_SPEC,
    NUMERIC_FEATURES,
    TARGET,
    build_feature_spec,
    load_dataset,
    make_preprocessor,
    split_data,
)
from surrogate.models import MODEL_REGISTRY, XGBOOST_AVAILABLE, build_model

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURE_SPEC",
    "MODEL_REGISTRY",
    "NUMERIC_FEATURES",
    "SurrogatePredictor",
    "TARGET",
    "XGBOOST_AVAILABLE",
    "build_feature_spec",
    "build_model",
    "build_predictor_from_pipeline",
    "load_dataset",
    "load_predictor",
    "make_preprocessor",
    "save_predictor",
    "split_data",
]
