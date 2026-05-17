"""Constrained black-box optimization problem for TOMATO.

A `GreenhouseProblem` instance bundles everything an optimizer needs:
  - the decision-variable vector + box bounds
  - the objective (surrogate-predicted yield)
  - the list of coupling constraints
  - sampling utilities (random, random-feasible)
  - encode/decode between the optimization vector and a surrogate config

One problem per crop. The optimizer (BO/GA/CMA-ES/PSO) is responsible
only for the continuous-vector search; crop_type and variety are fixed
inputs so the discrete combinatorial layer doesn't tangle with the
continuous one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from optimization.constraints import (
    Constraint,
    default_constraints,
    evaluate_constraints,
    max_violation,
    violation,
)
from optimization.cost_model import (
    TOTAL_AREA_M2,
    electricity_cost_usd,
    electricity_kwh_per_m2_per_cycle,
    fertilizer_cost_usd,
    total_operating_cost_usd,
    total_yield_kg,
    water_cost_usd,
    water_l_per_m2_per_cycle,
)
from optimization.crop_specs import (
    CROP_SPECS,
    DECISION_VARIABLES,
    MAX_PEST_SEVERITY,
)
from surrogate.api import SurrogatePredictor, load_predictor
from surrogate.data import load_dataset


# Bounds the Appendix doesn't specify directly (days_to_maturity and
# pest_severity). days bound is data-derived per crop; pest is [0, cap].
def _derive_days_bounds(crop: str) -> Tuple[float, float]:
    """[5th, 95th] percentile of days_to_maturity for `crop` in the data."""
    df = load_dataset(clean=True, engineer=False)
    days = df.loc[df["crop_type"] == crop, "days_to_maturity"].dropna()
    if len(days) < 30:
        return (30.0, 90.0)  # fallback if a crop is missing/under-represented
    return (float(days.quantile(0.05)), float(days.quantile(0.95)))


def _default_variety(crop: str) -> str:
    """Use the data-modal variety as the default (no other info to pick)."""
    df = load_dataset(clean=True, engineer=False)
    return str(df.loc[df["crop_type"] == crop, "variety"].mode().iloc[0])


@dataclass
class GreenhouseProblem:
    """Single-crop constrained yield-maximization problem.

    Attributes
    ----------
    crop : str               One of "Tomato", "Cucumber", "Lettuce", "Pepper".
    variety : str            Variety name (must be valid for `crop`).
    predictor : SurrogatePredictor   The trained yield surrogate.
    constraints : List[Constraint]   Coupling constraints; default per-milestone.
    area_m2 : float          Total growing area (default = 30-acre vertical farm).
    var_names : list         Decision-variable names, in vector order.
    lower_bounds, upper_bounds : np.ndarray  Box bounds, shape (dim,).
    """

    crop: str
    variety: Optional[str] = None
    predictor: Optional[SurrogatePredictor] = None
    constraints: List[Constraint] = field(default_factory=default_constraints)
    area_m2: float = TOTAL_AREA_M2

    var_names: List[str] = field(init=False)
    lower_bounds: np.ndarray = field(init=False)
    upper_bounds: np.ndarray = field(init=False)
    _days_bounds: Tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        if self.crop not in CROP_SPECS:
            raise ValueError(
                f"Unknown crop {self.crop!r}; valid: {sorted(CROP_SPECS)}"
            )

        if self.predictor is None:
            self.predictor = load_predictor()

        if self.variety is None:
            self.variety = _default_variety(self.crop)
        valid_varieties = self.predictor.valid_varieties_for(self.crop)
        if self.variety not in valid_varieties:
            raise ValueError(
                f"variety {self.variety!r} not valid for {self.crop}; "
                f"options: {valid_varieties}"
            )

        self._days_bounds = _derive_days_bounds(self.crop)
        self.var_names = list(DECISION_VARIABLES)
        spec = CROP_SPECS[self.crop]
        max_pest = MAX_PEST_SEVERITY[self.crop]

        lows, highs = [], []
        for name in self.var_names:
            if name == "days_to_maturity":
                lo, hi = self._days_bounds
            elif name == "pest_severity":
                lo, hi = 0.0, max_pest
            else:
                lo, hi = spec[name]
            lows.append(lo)
            highs.append(hi)
        self.lower_bounds = np.array(lows, dtype=float)
        self.upper_bounds = np.array(highs, dtype=float)

    # ------------------------------------------------------------------
    # Shape + bounds helpers
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self.var_names)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """(lower, upper) numpy arrays of shape (dim,)."""
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def midpoint(self) -> np.ndarray:
        """Center of the box. Used as a deterministic warm-start."""
        return 0.5 * (self.lower_bounds + self.upper_bounds)

    def clip_to_box(self, x: np.ndarray) -> np.ndarray:
        """Clamp `x` element-wise into the box [lower, upper]."""
        return np.minimum(np.maximum(x, self.lower_bounds), self.upper_bounds)

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert a decision vector to a full surrogate-ready config dict."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"x has shape {x.shape}; expected ({self.dim},)")
        cfg: Dict[str, Any] = dict(zip(self.var_names, x.tolist()))
        cfg["crop_type"] = self.crop
        cfg["variety"] = self.variety
        return cfg

    def decode_batch(self, X: np.ndarray) -> pd.DataFrame:
        """Vectorized decode for batch evaluation (used inside optimizer loops)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f"X has shape {X.shape}; expected (n, {self.dim})")
        df = pd.DataFrame(X, columns=self.var_names)
        df["crop_type"] = self.crop
        df["variety"] = self.variety
        return df

    def encode(self, config: Mapping[str, float]) -> np.ndarray:
        """Inverse of `decode`. Drops crop_type/variety; pulls decision vars."""
        missing = [n for n in self.var_names if n not in config]
        if missing:
            raise KeyError(f"config missing decision vars: {missing}")
        return np.array([float(config[n]) for n in self.var_names], dtype=float)

    # ------------------------------------------------------------------
    # Objective + constraints
    # ------------------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """Surrogate-predicted yield (kg/m²). Higher is better."""
        cfg = self.decode(x)
        return float(self.predictor.predict_one(cfg))

    def objective_min(self, x: np.ndarray) -> float:
        """Minimization-form objective (= -yield) for algorithms that minimize."""
        return -self.objective(x)

    def objective_batch(self, X: np.ndarray) -> np.ndarray:
        """Batched objective; ~390× faster than looping `objective(x)`."""
        df = self.decode_batch(X)
        return self.predictor.predict_batch(df)

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        """g(x) values for all coupling constraints (g ≤ 0 means feasible)."""
        cfg = self.decode(x)
        return evaluate_constraints(self.constraints, cfg)

    def constraint_values_batch(self, X: np.ndarray) -> np.ndarray:
        """Per-row constraint values; shape (n, n_constraints)."""
        df = self.decode_batch(X)
        rows = np.array([
            evaluate_constraints(self.constraints, df.iloc[i].to_dict())
            for i in range(len(df))
        ], dtype=float)
        return rows

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Feasibility check: box bounds AND every coupling constraint."""
        x = np.asarray(x, dtype=float)
        if (x < self.lower_bounds - tol).any() or (x > self.upper_bounds + tol).any():
            return False
        g = self.constraint_values(x)
        return bool((g <= tol).all())

    def violation(self, x: np.ndarray) -> float:
        """Quadratic-penalty value (sum of squared positive g_i)."""
        return violation(self.constraint_values(x))

    def max_violation(self, x: np.ndarray) -> float:
        """∞-norm of constraint violations (largest single breach)."""
        return max_violation(self.constraint_values(x))

    def penalized_objective(
        self, x: np.ndarray, penalty_weight: float = 1e3, minimize: bool = True
    ) -> float:
        """Single scalar with quadratic constraint penalty added in.

        Suitable for the unconstrained variants of CMA-ES / PSO / GA.
        `minimize=True` returns -yield + λ·violation (i.e. minimize me);
        `minimize=False` returns yield - λ·violation (maximize me).
        """
        y = self.objective(x)
        v = self.violation(x)
        return (-y if minimize else y) + (penalty_weight * v if minimize
                                          else -penalty_weight * v)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_random(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Uniform sample inside the box (ignores coupling constraints)."""
        rng = rng if rng is not None else np.random.default_rng()
        return rng.uniform(self.lower_bounds, self.upper_bounds)

    def sample_random_batch(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.uniform(self.lower_bounds, self.upper_bounds, size=(n, self.dim))

    def sample_random_feasible(
        self,
        max_tries: int = 1000,
        rng: Optional[np.random.Generator] = None,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Rejection-sample inside the FEASIBLE region (box + coupling).

        Useful for initializing population-based methods. Raises
        RuntimeError if the feasible region is too small to hit in
        `max_tries` random box samples.
        """
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(max_tries):
            x = self.sample_random(rng=rng)
            # Cheap pre-check: enforce temp ordering by sorting so we
            # waste fewer draws on obviously-bad temp triples.
            t = sorted([x[1], x[2], x[3]])  # avg, min, max  → sorted = (min, avg, max)
            x[1], x[2], x[3] = t[1], t[0], t[2]
            x = self.clip_to_box(x)
            if self.is_feasible(x, tol=tol):
                return x
        raise RuntimeError(
            f"Could not find a feasible sample for crop={self.crop!r} in "
            f"{max_tries} tries. The constraint set may be infeasible "
            f"(e.g. electricity budget too low for the photoperiod/light box)."
        )

    # ------------------------------------------------------------------
    # Diagnostics + accounting helpers
    # ------------------------------------------------------------------

    def electricity_kwh_per_m2(self, x: np.ndarray) -> float:
        return electricity_kwh_per_m2_per_cycle(self.decode(x))

    def water_l_per_m2(self, x: np.ndarray) -> float:
        return water_l_per_m2_per_cycle(self.decode(x))

    def total_yield_kg(self, x: np.ndarray) -> float:
        return total_yield_kg(self.objective(x), area_m2=self.area_m2)

    def total_cost_usd(self, x: np.ndarray) -> float:
        return total_operating_cost_usd(self.decode(x), area_m2=self.area_m2)

    def cost_breakdown_usd(self, x: np.ndarray) -> Dict[str, float]:
        cfg = self.decode(x)
        return {
            "electricity": electricity_cost_usd(cfg, area_m2=self.area_m2),
            "water":       water_cost_usd(cfg, area_m2=self.area_m2),
            "fertilizer":  fertilizer_cost_usd(cfg, area_m2=self.area_m2),
        }

    def summary(self, x: np.ndarray) -> Dict[str, Any]:
        """One-stop report on a single decision vector."""
        cfg = self.decode(x)
        g = self.constraint_values(x)
        feasible = bool((g <= 1e-6).all()) and not (
            (x < self.lower_bounds - 1e-6).any()
            or (x > self.upper_bounds + 1e-6).any()
        )
        return {
            "crop": self.crop,
            "variety": self.variety,
            "feasible": feasible,
            "yield_kg_per_m2": self.objective(x),
            "total_yield_kg": self.total_yield_kg(x),
            "electricity_kwh_per_m2": self.electricity_kwh_per_m2(x),
            "water_l_per_m2": self.water_l_per_m2(x),
            "total_cost_usd": self.total_cost_usd(x),
            "cost_breakdown_usd": self.cost_breakdown_usd(x),
            "max_constraint_violation": self.max_violation(x),
            "constraints": {
                c.name: float(val) for c, val in zip(self.constraints, g)
            },
            "config": cfg,
        }


# Convenience constructors for the four crops.

def make_all_problems(
    predictor: Optional[SurrogatePredictor] = None,
) -> Dict[str, GreenhouseProblem]:
    """One GreenhouseProblem per crop, sharing the same surrogate."""
    if predictor is None:
        predictor = load_predictor()
    return {
        crop: GreenhouseProblem(crop=crop, predictor=predictor)
        for crop in CROP_SPECS
    }
