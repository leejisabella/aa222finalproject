"""Electricity proxy and operating-cost computations.

The dataset doesn't measure greenhouse electricity directly, so per
the milestone we approximate it from `light_intensity_lux ×
photoperiod_hours × days_to_maturity`. We use a simple, defensible
lux → W/m² conversion (LED-equivalent: 1 W/m² ≈ 100 lux) so cycle-
total electricity comes out in the 100–300 kWh/m² band reported in
the milestone for typical greenhouse operation.

Cost numbers:
  - $0.41/kWh (San Francisco residential average, per milestone)
  - Vertical farm scale: 3 acres × 10 layers (12,141 m² per layer)
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

# 1 acre = 4046.86 m^2 (US survey acre). Three acres × ten vertical layers.
ACRE_M2 = 4046.86
LAYER_AREA_M2 = 3.0 * ACRE_M2          # 12,140.6 m^2 per layer
NUM_LAYERS = 10
TOTAL_AREA_M2 = LAYER_AREA_M2 * NUM_LAYERS  # 121,406 m^2 (vertical farm)

# Electricity price ($ / kWh). San Francisco average per milestone.
ELECTRICITY_PRICE_USD_PER_KWH = 0.41

# Lux-to-W/m^2 conversion for LED greenhouse lighting. 100 lux per W/m^2
# is a defensible round number for white-LED grow lights (the actual
# range is ~75–150 lux/W/m^2 depending on spectrum); using a single
# constant keeps the proxy interpretable and the milestone's 100–300
# kWh/m^2 band achievable for realistic configs.
LUX_PER_WATT_PER_M2 = 100.0


def electricity_kwh_per_m2_per_cycle(config: Mapping[str, float]) -> float:
    """Total lighting electricity consumed by one harvest cycle, per m^2.

    Formula:
        kWh/m²/cycle
          = (lux / LUX_PER_WATT_PER_M2)      ← lighting power density W/m²
          × photoperiod_hours                  ← daily on-time
          × days_to_maturity                   ← cycle length
          × 1e-3                               ← Wh → kWh

    Sanity check: tomato at 30000 lux, 14 h, 70 days
      = (30000/100) × 14 × 70 × 1e-3 = 294 kWh/m²/cycle  ← top of milestone band ✓
    """
    lux = float(config["light_intensity_lux"])
    photoperiod = float(config["photoperiod_hours"])
    days = float(config["days_to_maturity"])
    watts_per_m2 = lux / LUX_PER_WATT_PER_M2
    return watts_per_m2 * photoperiod * days * 1e-3


def electricity_cost_usd(
    config: Mapping[str, float],
    area_m2: float = TOTAL_AREA_M2,
    price_per_kwh: float = ELECTRICITY_PRICE_USD_PER_KWH,
) -> float:
    """Electricity cost for one cycle over the given area."""
    return electricity_kwh_per_m2_per_cycle(config) * area_m2 * price_per_kwh


def water_l_per_m2_per_cycle(config: Mapping[str, float]) -> float:
    """Total irrigation water per m^2 per cycle (mm × days = L/m²)."""
    return float(config["irrigation_mm"]) * float(config["days_to_maturity"])


def water_cost_usd(
    config: Mapping[str, float],
    area_m2: float = TOTAL_AREA_M2,
    price_per_m3: float = 3.0,  # US municipal treated-water midpoint
) -> float:
    """Irrigation water cost. 1 mm × 1 m² = 1 L = 1e-3 m³."""
    liters = water_l_per_m2_per_cycle(config) * area_m2
    return liters * 1e-3 * price_per_m3


def fertilizer_kg_per_ha_total(config: Mapping[str, float]) -> float:
    """Total N+P+K applied (kg/ha)."""
    return (float(config["fertilizer_N_kg_ha"])
            + float(config["fertilizer_P_kg_ha"])
            + float(config["fertilizer_K_kg_ha"]))


def fertilizer_cost_usd(
    config: Mapping[str, float],
    area_m2: float = TOTAL_AREA_M2,
    price_per_kg: float = 1.50,  # blended NPK fertilizer midpoint
) -> float:
    """Fertilizer cost. Convert kg/ha to absolute kg given area_m2."""
    area_ha = area_m2 / 10_000.0
    return fertilizer_kg_per_ha_total(config) * area_ha * price_per_kg


def total_yield_kg(
    yield_kg_per_m2: float,
    area_m2: float = TOTAL_AREA_M2,
) -> float:
    """Cycle-total yield over the vertical farm (kg)."""
    return float(yield_kg_per_m2) * area_m2


def total_operating_cost_usd(
    config: Mapping[str, float],
    area_m2: float = TOTAL_AREA_M2,
) -> float:
    """Sum of electricity + water + fertilizer for one cycle."""
    return (electricity_cost_usd(config, area_m2=area_m2)
            + water_cost_usd(config, area_m2=area_m2)
            + fertilizer_cost_usd(config, area_m2=area_m2))


# Batch versions for optimizer inner loops. All accept a DataFrame and
# return a numpy array of length len(df).
def electricity_kwh_per_m2_batch(df) -> np.ndarray:
    watts = df["light_intensity_lux"].to_numpy(dtype=float) / LUX_PER_WATT_PER_M2
    return watts * df["photoperiod_hours"].to_numpy(dtype=float) \
                 * df["days_to_maturity"].to_numpy(dtype=float) * 1e-3
