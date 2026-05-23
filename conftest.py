"""Pytest configuration — ensures repository root is on sys.path."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Silence two categories of sklearn noise that fire heavily in the BO tests:
#   - ConvergenceWarning: GP kernel-MLE inner optimizer hitting its iter cap.
#   - InconsistentVersionWarning: surrogate.pkl was trained with sklearn 1.5.0;
#     a newer sklearn in the env still loads it fine.
try:
    from sklearn.exceptions import (
        ConvergenceWarning,
        InconsistentVersionWarning,
    )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
