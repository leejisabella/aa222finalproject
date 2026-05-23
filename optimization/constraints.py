"""Coupling constraints for the greenhouse optimization problem.

Convention: every constraint function returns g(x) such that
    g(x) ≤ 0  ⇔  feasible.
This matches how AA222/CS361 algorithms (penalty methods, GA repair
operators, BO constrained acquisition) typically consume constraints.

Box bounds on individual variables are handled by the Problem class's
lower_bounds/upper_bounds; the functions in this module cover the
*coupling* constraints that link variables together — the ones that
make the feasible region non-rectangular and the optimization actually
interesting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping

import numpy as np

from optimization.cost_model import (
    electricity_kwh_per_m2_per_cycle,
    total_operating_cost_usd,
    water_l_per_m2_per_cycle,
)

# Default electricity budget per the milestone: 100–300 kWh/m²/cycle.
# Both bounds enforced by default — the milestone states the band as a
# range, not just a ceiling.
DEFAULT_ELECTRICITY_BUDGET_KWH_PER_M2: tuple[float, float] = (100.0, 300.0)

# VPD horticultural band from the milestone. < 0.5 kPa → fungal disease
# risk; > 1.2 kPa → plant water stress.
VPD_LOWER_KPA = 0.5
VPD_UPPER_KPA = 1.2

# Water budget: cycle-total irrigation per m² (irrigation_mm × days).
DEFAULT_WATER_BUDGET_L_PER_M2 = 400.0

# Operating cost budget: electricity + water + fertilizer per m² per cycle.
DEFAULT_COST_BUDGET_USD_PER_M2 = 300.0


def vapor_pressure_deficit_kpa(avg_temp_c: float, humidity_percent: float) -> float:
    """VPD in kPa. Magnus-Tetens form of saturation vapor pressure.

    Same formula as `surrogate.features.vapor_pressure_deficit_kpa` but
    on a single config (not a Series), since constraint evaluation is
    scalar-by-scalar inside the optimizer's inner loop.
    """
    es = 0.6108 * np.exp(17.27 * avg_temp_c / (avg_temp_c + 237.3))
    return (1.0 - humidity_percent / 100.0) * es


# ---------------------------------------------------------------------------
# Constraint definitions. Each returns g(config) ≤ 0 ⇔ feasible.
# ---------------------------------------------------------------------------


@dataclass
class Constraint:
    """A scalar-valued constraint g(config) ≤ 0."""

    name: str
    g: Callable[[Mapping[str, float]], float]
    description: str

    def __call__(self, config: Mapping[str, float]) -> float:
        return float(self.g(config))


def _g_min_le_avg(config: Mapping[str, float]) -> float:
    return float(config["min_temperature_C"]) - float(config["avg_temperature_C"])


def _g_avg_le_max(config: Mapping[str, float]) -> float:
    return float(config["avg_temperature_C"]) - float(config["max_temperature_C"])


def _make_vpd_low_constraint(lower_kpa: float):
    def _g(config: Mapping[str, float]) -> float:
        vpd = vapor_pressure_deficit_kpa(
            float(config["avg_temperature_C"]),
            float(config["humidity_percent"]),
        )
        return lower_kpa - vpd
    return _g


def _make_vpd_high_constraint(upper_kpa: float):
    def _g(config: Mapping[str, float]) -> float:
        vpd = vapor_pressure_deficit_kpa(
            float(config["avg_temperature_C"]),
            float(config["humidity_percent"]),
        )
        return vpd - upper_kpa
    return _g


def _make_electricity_upper(budget: float):
    def _g(config: Mapping[str, float]) -> float:
        return electricity_kwh_per_m2_per_cycle(config) - budget
    return _g


def _make_electricity_lower(floor: float):
    def _g(config: Mapping[str, float]) -> float:
        return floor - electricity_kwh_per_m2_per_cycle(config)
    return _g


def _make_water_upper(budget_l_per_m2: float):
    def _g(config: Mapping[str, float]) -> float:
        return water_l_per_m2_per_cycle(config) - budget_l_per_m2
    return _g


def _make_cost_upper(budget_usd_per_m2: float):
    # Operating cost scales linearly with area, so total_operating_cost_usd
    # evaluated at area_m2=1.0 returns $/m²/cycle directly.
    def _g(config: Mapping[str, float]) -> float:
        return total_operating_cost_usd(config, area_m2=1.0) - budget_usd_per_m2
    return _g


def default_constraints(
    electricity_budget_kwh_per_m2: float | tuple[float, float] = DEFAULT_ELECTRICITY_BUDGET_KWH_PER_M2,
    vpd_band_kpa: tuple[float, float] = (VPD_LOWER_KPA, VPD_UPPER_KPA),
    water_budget_l_per_m2: float | None = DEFAULT_WATER_BUDGET_L_PER_M2,
    cost_budget_usd_per_m2: float | None = DEFAULT_COST_BUDGET_USD_PER_M2,
) -> List[Constraint]:
    """Build the standard coupling-constraint list for the milestone problem.

    Parameters
    ----------
    electricity_budget_kwh_per_m2
        Scalar (upper bound only) or (lo, hi) tuple. Default is
        (100.0, 300.0) per the milestone.
    vpd_band_kpa
        (lower, upper) VPD band in kPa. Default (0.5, 1.2).
    water_budget_l_per_m2
        Cycle-total irrigation cap (L/m²/cycle). `None` disables this
        constraint; default 400.0.
    cost_budget_usd_per_m2
        Cycle-total operating-cost cap ($/m²/cycle), summing electricity,
        water, and fertilizer. `None` disables; default $300/m²/cycle.

    Returns a list of `Constraint` objects in a fixed order so the
    output of `Problem.constraints(x)` is reproducible across runs.
    """
    cons: List[Constraint] = []

    # Temperature ordering: min ≤ avg ≤ max.
    cons.append(Constraint(
        name="temp_min_le_avg",
        g=_g_min_le_avg,
        description="min_temperature_C ≤ avg_temperature_C",
    ))
    cons.append(Constraint(
        name="temp_avg_le_max",
        g=_g_avg_le_max,
        description="avg_temperature_C ≤ max_temperature_C",
    ))

    # Vapor pressure deficit band.
    lo_vpd, hi_vpd = vpd_band_kpa
    cons.append(Constraint(
        name="vpd_lower",
        g=_make_vpd_low_constraint(lo_vpd),
        description=f"VPD(T,H) ≥ {lo_vpd} kPa (avoid fungal disease)",
    ))
    cons.append(Constraint(
        name="vpd_upper",
        g=_make_vpd_high_constraint(hi_vpd),
        description=f"VPD(T,H) ≤ {hi_vpd} kPa (avoid plant water stress)",
    ))

    # Electricity budget.
    if isinstance(electricity_budget_kwh_per_m2, tuple):
        lo_e, hi_e = electricity_budget_kwh_per_m2
        cons.append(Constraint(
            name="electricity_lower",
            g=_make_electricity_lower(lo_e),
            description=f"electricity ≥ {lo_e} kWh/m²/cycle",
        ))
        cons.append(Constraint(
            name="electricity_upper",
            g=_make_electricity_upper(hi_e),
            description=f"electricity ≤ {hi_e} kWh/m²/cycle",
        ))
    else:
        cons.append(Constraint(
            name="electricity_upper",
            g=_make_electricity_upper(float(electricity_budget_kwh_per_m2)),
            description=f"electricity ≤ {electricity_budget_kwh_per_m2} kWh/m²/cycle",
        ))

    # Water budget (cycle-total irrigation per m²).
    if water_budget_l_per_m2 is not None:
        cons.append(Constraint(
            name="water_upper",
            g=_make_water_upper(float(water_budget_l_per_m2)),
            description=f"water ≤ {water_budget_l_per_m2} L/m²/cycle",
        ))

    # Operating-cost budget (electricity + water + fertilizer per m²).
    if cost_budget_usd_per_m2 is not None:
        cons.append(Constraint(
            name="cost_upper",
            g=_make_cost_upper(float(cost_budget_usd_per_m2)),
            description=f"op cost ≤ ${cost_budget_usd_per_m2}/m²/cycle",
        ))

    return cons


def evaluate_constraints(
    constraints: List[Constraint], config: Mapping[str, float]
) -> np.ndarray:
    """Evaluate every constraint at a single config; return the g-values."""
    return np.array([c(config) for c in constraints], dtype=float)


def violation(g_values: np.ndarray) -> float:
    """Sum-of-squared positive parts of g — a quadratic penalty.

    Useful both as a feasibility-distance metric and as the standard
    quadratic penalty term for penalty-method optimizers.
    """
    return float(np.sum(np.maximum(0.0, g_values) ** 2))


def max_violation(g_values: np.ndarray) -> float:
    """Largest single-constraint violation (∞-norm of positive part)."""
    if len(g_values) == 0:
        return 0.0
    return float(max(0.0, g_values.max()))
