"""Tests for the optimization problem definition.

Covers cost_model, constraints, and GreenhouseProblem. The optimization
algorithms themselves are not yet implemented; these tests pin down the
problem so future optimizer code has a stable contract to call against.
"""
from __future__ import annotations

import numpy as np
import pytest

from optimization import (
    CROP_SPECS,
    DECISION_VARIABLES,
    DEFAULT_ELECTRICITY_BUDGET_KWH_PER_M2,
    ELECTRICITY_PRICE_USD_PER_KWH,
    GreenhouseProblem,
    LAYER_AREA_M2,
    MAX_PEST_SEVERITY,
    NUM_LAYERS,
    TOTAL_AREA_M2,
    VPD_LOWER_KPA,
    VPD_UPPER_KPA,
    default_constraints,
    electricity_kwh_per_m2_per_cycle,
    make_all_problems,
    total_operating_cost_usd,
    vapor_pressure_deficit_kpa,
)


# ============================================================
# Crop specs sanity
# ============================================================


def test_all_four_crops_present():
    assert set(CROP_SPECS) == {"Tomato", "Cucumber", "Lettuce", "Pepper"}


def test_appendix_a_tomato_values():
    spec = CROP_SPECS["Tomato"]
    assert spec["min_temperature_C"][0] == 15.0
    assert spec["max_temperature_C"][1] == 32.0
    assert spec["humidity_percent"] == (60.0, 75.0)
    assert spec["light_intensity_lux"] == (20000.0, 50000.0)
    assert spec["photoperiod_hours"][1] == 16.0  # tomato chlorosis cap


def test_lettuce_photoperiod_cap():
    # Milestone biology: lettuce bolts under long days → cap at 14.
    assert CROP_SPECS["Lettuce"]["photoperiod_hours"][1] == 14.0


def test_pest_severity_caps():
    # Lettuce has a tighter pest cap (2) than the other crops (3).
    assert MAX_PEST_SEVERITY["Lettuce"] == 2.0
    assert MAX_PEST_SEVERITY["Tomato"] == 3.0


# ============================================================
# Cost model
# ============================================================


def test_total_area_is_30_acres():
    # 3 acres × 10 layers = 30 acres.
    assert LAYER_AREA_M2 == pytest.approx(3 * 4046.86)
    assert NUM_LAYERS == 10
    assert TOTAL_AREA_M2 == pytest.approx(LAYER_AREA_M2 * 10)


def test_electricity_proxy_top_of_band():
    """Milestone sanity check: 30 000 lux × 14 h × 70 days ≈ 294 kWh/m²/cycle."""
    cfg = {"light_intensity_lux": 30000.0, "photoperiod_hours": 14.0,
           "days_to_maturity": 70.0}
    e = electricity_kwh_per_m2_per_cycle(cfg)
    assert 280 < e < 310  # top of milestone 100-300 band


def test_electricity_scales_linearly_in_all_three_factors():
    base = {"light_intensity_lux": 20000.0, "photoperiod_hours": 10.0,
            "days_to_maturity": 50.0}
    e0 = electricity_kwh_per_m2_per_cycle(base)
    # Doubling lux, photoperiod, or days each doubles electricity.
    for col in base:
        modified = dict(base); modified[col] *= 2
        assert electricity_kwh_per_m2_per_cycle(modified) == pytest.approx(2 * e0)


def test_electricity_price_is_san_francisco_milestone_value():
    assert ELECTRICITY_PRICE_USD_PER_KWH == 0.41


def test_total_cost_sums_electricity_water_fertilizer():
    from optimization.cost_model import (
        electricity_cost_usd,
        fertilizer_cost_usd,
        water_cost_usd,
    )
    cfg = {"light_intensity_lux": 30000.0, "photoperiod_hours": 14.0,
           "days_to_maturity": 70.0, "irrigation_mm": 6.0,
           "fertilizer_N_kg_ha": 150.0, "fertilizer_P_kg_ha": 70.0,
           "fertilizer_K_kg_ha": 200.0}
    total = total_operating_cost_usd(cfg)
    parts = electricity_cost_usd(cfg) + water_cost_usd(cfg) + fertilizer_cost_usd(cfg)
    assert total == pytest.approx(parts)


# ============================================================
# VPD + constraints
# ============================================================


def test_vpd_at_full_humidity_is_zero():
    assert vapor_pressure_deficit_kpa(25.0, 100.0) == pytest.approx(0.0, abs=1e-9)


def test_vpd_matches_horticulture_reference_at_25c_70pct():
    # Standard greenhouse reference: 25°C, 70% RH → VPD ≈ 0.95 kPa.
    vpd = vapor_pressure_deficit_kpa(25.0, 70.0)
    assert 0.85 < vpd < 1.05


def test_default_constraints_have_5_when_scalar_budget():
    cons = default_constraints()
    # 2 temp-ordering + 2 VPD + 1 electricity upper = 5
    assert len(cons) == 5
    names = [c.name for c in cons]
    assert {"temp_min_le_avg", "temp_avg_le_max", "vpd_lower",
            "vpd_upper", "electricity_upper"} == set(names)


def test_default_constraints_have_6_when_tuple_budget():
    cons = default_constraints(electricity_budget_kwh_per_m2=(100, 300))
    assert len(cons) == 6
    names = [c.name for c in cons]
    assert "electricity_lower" in names
    assert "electricity_upper" in names


def test_vpd_constraints_sign_convention():
    """g ≤ 0 = feasible. 25°C / 70% RH gives VPD ≈ 0.95, in [0.5, 1.2]."""
    cons = {c.name: c for c in default_constraints()}
    cfg = {"avg_temperature_C": 25.0, "humidity_percent": 70.0,
           "min_temperature_C": 20.0, "max_temperature_C": 28.0,
           "light_intensity_lux": 25000.0, "photoperiod_hours": 14.0,
           "days_to_maturity": 65.0}
    assert cons["vpd_lower"](cfg) < 0  # 0.5 - 0.95 < 0 ✓
    assert cons["vpd_upper"](cfg) < 0  # 0.95 - 1.2 < 0 ✓


def test_temp_ordering_constraints_detect_inversion():
    cons = {c.name: c for c in default_constraints()}
    bad = {"min_temperature_C": 30, "avg_temperature_C": 20, "max_temperature_C": 25,
           "humidity_percent": 70, "light_intensity_lux": 25000,
           "photoperiod_hours": 14, "days_to_maturity": 65}
    # min > avg → g > 0 (violated)
    assert cons["temp_min_le_avg"](bad) > 0


# ============================================================
# GreenhouseProblem — structure
# ============================================================


@pytest.fixture(scope="module")
def problems():
    return make_all_problems()


def test_one_problem_per_crop(problems):
    assert set(problems) == {"Tomato", "Cucumber", "Lettuce", "Pepper"}


def test_problem_dimension_is_14(problems):
    for p in problems.values():
        assert p.dim == 14
        assert len(p.var_names) == 14
        assert p.var_names == DECISION_VARIABLES


def test_bounds_are_finite_and_ordered(problems):
    for crop, p in problems.items():
        lo, hi = p.bounds()
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
        assert np.all(lo <= hi), f"{crop}: bounds not ordered"


def test_pest_severity_bound_matches_crop_cap(problems):
    for crop, p in problems.items():
        idx = p.var_names.index("pest_severity")
        assert p.lower_bounds[idx] == 0.0
        assert p.upper_bounds[idx] == MAX_PEST_SEVERITY[crop]


def test_invalid_crop_raises():
    with pytest.raises(ValueError, match="Unknown crop"):
        GreenhouseProblem(crop="Banana")


def test_invalid_variety_raises():
    with pytest.raises(ValueError, match="not valid"):
        GreenhouseProblem(crop="Tomato", variety="Iceberg")  # lettuce variety


# ============================================================
# GreenhouseProblem — encode / decode
# ============================================================


def test_decode_produces_surrogate_compatible_dict(problems):
    p = problems["Tomato"]
    x = p.midpoint()
    cfg = p.decode(x)
    # Surrogate needs crop_type + variety + all decision vars.
    assert cfg["crop_type"] == "Tomato"
    assert cfg["variety"] == p.variety
    for v in DECISION_VARIABLES:
        assert v in cfg


def test_encode_decode_roundtrip(problems):
    p = problems["Cucumber"]
    rng = np.random.default_rng(42)
    x = p.sample_random(rng=rng)
    x_back = p.encode(p.decode(x))
    np.testing.assert_array_equal(x, x_back)


def test_decode_wrong_shape_raises(problems):
    p = problems["Lettuce"]
    with pytest.raises(ValueError, match="shape"):
        p.decode(np.zeros(13))


# ============================================================
# GreenhouseProblem — objective + constraints
# ============================================================


def test_objective_returns_finite_yield(problems):
    for crop, p in problems.items():
        y = p.objective(p.midpoint())
        assert np.isfinite(y)
        # Realistic per-crop spread (data has up to ~30 kg/m^2 for tomato).
        assert 0 < y < 60


def test_objective_min_is_negative_objective(problems):
    p = problems["Tomato"]
    x = p.midpoint()
    assert p.objective_min(x) == pytest.approx(-p.objective(x))


def test_objective_batch_matches_loop(problems):
    p = problems["Cucumber"]
    rng = np.random.default_rng(0)
    X = p.sample_random_batch(8, rng=rng)
    batch = p.objective_batch(X)
    loop = np.array([p.objective(x) for x in X])
    np.testing.assert_allclose(batch, loop, rtol=1e-9)


def test_objective_batch_is_faster_than_loop(problems):
    """Optimizer inner loops should batch — verify the API delivers it."""
    import time
    p = problems["Tomato"]
    rng = np.random.default_rng(0)
    X = p.sample_random_batch(100, rng=rng)
    t0 = time.perf_counter()
    p.objective_batch(X)
    t_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    for x in X:
        p.objective(x)
    t_loop = time.perf_counter() - t0
    # Batch should be at least 5× faster (typical ~50×).
    assert t_batch < t_loop / 5, f"batch={t_batch:.3f}s vs loop={t_loop:.3f}s"


def test_constraint_values_shape(problems):
    for p in problems.values():
        g = p.constraint_values(p.midpoint())
        assert g.shape == (len(p.constraints),)
        assert np.all(np.isfinite(g))


def test_is_feasible_handles_violation(problems):
    p = problems["Tomato"]
    # Midpoint of Tomato violates the electricity budget (smoke-tested earlier).
    assert not p.is_feasible(p.midpoint())


def test_is_feasible_detects_box_violation(problems):
    p = problems["Lettuce"]
    x = p.midpoint().copy()
    x[0] = p.upper_bounds[0] + 100.0
    assert not p.is_feasible(x)


def test_clip_to_box_works(problems):
    p = problems["Pepper"]
    x = p.upper_bounds + 50.0
    clipped = p.clip_to_box(x)
    assert np.all(clipped <= p.upper_bounds + 1e-9)
    assert np.all(clipped >= p.lower_bounds - 1e-9)


# ============================================================
# GreenhouseProblem — sampling
# ============================================================


def test_random_sample_in_box(problems):
    rng = np.random.default_rng(0)
    for p in problems.values():
        for _ in range(50):
            x = p.sample_random(rng=rng)
            assert np.all(x >= p.lower_bounds) and np.all(x <= p.upper_bounds)


def test_random_batch_shape(problems):
    p = problems["Tomato"]
    rng = np.random.default_rng(0)
    X = p.sample_random_batch(25, rng=rng)
    assert X.shape == (25, p.dim)


def test_feasible_sample_satisfies_constraints(problems):
    rng = np.random.default_rng(1)
    for crop, p in problems.items():
        x = p.sample_random_feasible(rng=rng, max_tries=2000)
        assert p.is_feasible(x), f"{crop} feasible sample reports infeasible"
        # Box-bounds check
        assert np.all(x >= p.lower_bounds - 1e-9)
        assert np.all(x <= p.upper_bounds + 1e-9)


def test_feasible_sampling_reproducible_with_seed(problems):
    p = problems["Cucumber"]
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    np.testing.assert_array_equal(
        p.sample_random_feasible(rng=rng1),
        p.sample_random_feasible(rng=rng2),
    )


# ============================================================
# Penalty + summary helpers
# ============================================================


def test_penalty_objective_penalizes_infeasibility(problems):
    p = problems["Tomato"]
    x_infeas = p.midpoint()  # violates electricity
    rng = np.random.default_rng(2)
    x_feas = p.sample_random_feasible(rng=rng)
    # Both yield about the same magnitude; penalty pushes infeasible higher (worse).
    p_inf = p.penalized_objective(x_infeas, penalty_weight=1e3, minimize=True)
    p_fea = p.penalized_objective(x_feas, penalty_weight=1e3, minimize=True)
    assert p_inf > p_fea


def test_summary_keys(problems):
    p = problems["Lettuce"]
    rng = np.random.default_rng(3)
    x = p.sample_random_feasible(rng=rng)
    s = p.summary(x)
    expected = {
        "crop", "variety", "feasible", "yield_kg_per_m2", "total_yield_kg",
        "electricity_kwh_per_m2", "water_l_per_m2", "total_cost_usd",
        "cost_breakdown_usd", "max_constraint_violation",
        "constraints", "config",
    }
    assert expected <= set(s)


def test_total_yield_scales_with_area(problems):
    p = problems["Tomato"]
    x = p.midpoint()
    per_m2 = p.objective(x)
    assert p.total_yield_kg(x) == pytest.approx(per_m2 * TOTAL_AREA_M2)
