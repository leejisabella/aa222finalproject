"""Domain-derived feature engineering for the greenhouse surrogate.

The raw dataset hands the model temperature, humidity, lux, etc. as
independent columns. But plants integrate those signals in known,
nonlinear ways: vapor pressure deficit ties humidity and temperature,
daily light integral ties intensity and photoperiod, and so on. Trees
*can* approximate these through axis-aligned splits, but giving the
features directly lets a much smaller model capture them cleanly.

All formulas come from standard horticulture references:
  - VPD: Murray (1967) / FAO-56 Penman-Monteith
  - DLI: Faust & Logan (2018), HortScience 53(9)
  - GDD: McMaster & Wilhelm (1997), Agric. For. Meteorol. 87(4)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Approximate conversion: PAR (photosynthetically active radiation) in
# umol/m^2/s ≈ lux × 0.0185 for sunlight. Wrong for narrow-band LEDs but
# this dataset doesn't specify lamp type, so we use the sunlight conv.
LUX_TO_PPFD = 0.0185

# Base temperature for growing degree days. 10°C is the standard reference
# for warm-season vegetables (tomato/pepper/cucumber); lettuce uses 4°C
# but we use 10 as a single number since the model can adjust per-crop.
GDD_BASE_TEMP_C = 10.0


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN where the denominator is
    zero/NaN rather than inf — keeps downstream models from seeing inf."""
    den = den.where((den != 0) & den.notna(), np.nan)
    return num / den


def vapor_pressure_deficit_kpa(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """VPD in kPa. Magnus-Tetens form of saturation vapor pressure."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return (1 - humidity_pct / 100.0) * es


def daily_light_integral_mol(light_lux: pd.Series, photoperiod_h: pd.Series) -> pd.Series:
    """DLI in mol/m^2/day. Standard horticultural target metric."""
    ppfd = light_lux * LUX_TO_PPFD  # umol/m^2/s
    return ppfd * photoperiod_h * 3600.0 / 1.0e6


def growing_degree_days(avg_temp_c: pd.Series, days_to_maturity: pd.Series) -> pd.Series:
    """Total heat accumulation over the growing period (°C·days)."""
    daily = (avg_temp_c - GDD_BASE_TEMP_C).clip(lower=0)
    return daily * days_to_maturity


# Names of features added by `add_engineered_features` — exported so
# `data.py` can include them in the numeric feature list and the
# preprocessor knows about them.
ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "vpd_kpa",
    "dli_mol_m2_day",
    "gdd_degC_days",
    "temp_range_C",
    "n_to_p_ratio",
    "k_to_n_ratio",
    "n_share_npk",
    "total_npk_kg_ha",
    "total_irrigation_mm",
    "total_dli_mol_m2",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-derived features. Returns a new DataFrame (does not mutate).

    Engineered features:
      vpd_kpa            — vapor pressure deficit; couples temp + humidity.
      dli_mol_m2_day     — daily light integral; couples lux + photoperiod.
      gdd_degC_days      — growing degree days over the season.
      temp_range_C       — diurnal temperature swing (max - min).
      n_to_p_ratio       — fertilizer ratio (plants respond to ratios).
      k_to_n_ratio       — fertilizer ratio.
      n_share_npk        — N / (N+P+K).
      total_npk_kg_ha    — total fertilizer load.
      total_irrigation_mm — irrigation rate × days (season total).
      total_dli_mol_m2   — DLI × days (cumulative light over season).
    """
    out = df.copy()

    # Coupling features
    out["vpd_kpa"] = vapor_pressure_deficit_kpa(out["avg_temperature_C"], out["humidity_percent"])
    out["dli_mol_m2_day"] = daily_light_integral_mol(out["light_intensity_lux"], out["photoperiod_hours"])
    out["gdd_degC_days"] = growing_degree_days(out["avg_temperature_C"], out["days_to_maturity"])
    out["temp_range_C"] = out["max_temperature_C"] - out["min_temperature_C"]

    # Fertilizer ratios. Use safe division so missing P (or N) yields NaN.
    out["n_to_p_ratio"] = _safe_div(out["fertilizer_N_kg_ha"], out["fertilizer_P_kg_ha"])
    out["k_to_n_ratio"] = _safe_div(out["fertilizer_K_kg_ha"], out["fertilizer_N_kg_ha"])
    total_npk = (out["fertilizer_N_kg_ha"]
                 + out["fertilizer_P_kg_ha"]
                 + out["fertilizer_K_kg_ha"])
    out["total_npk_kg_ha"] = total_npk
    out["n_share_npk"] = _safe_div(out["fertilizer_N_kg_ha"], total_npk)

    # Cumulative season totals
    out["total_irrigation_mm"] = out["irrigation_mm"] * out["days_to_maturity"]
    out["total_dli_mol_m2"] = out["dli_mol_m2_day"] * out["days_to_maturity"]

    return out
