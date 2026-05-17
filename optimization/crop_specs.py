"""Per-crop greenhouse bounds from the project milestone Appendix A.

All bounds come from published greenhouse guidelines for commercial
crop growth conditions. See AA222_CS361_Project_Milestone.pdf,
Appendix A, Table 1.

The Appendix gives bounds on *each individual variable*; coupling
constraints (VPD, temperature ordering, electricity budget) are added
on top in `constraints.py`. Photoperiod upper bounds already encode the
"chlorosis on tomatoes / bolting on lettuce" biology constraint from
the milestone text.
"""
from __future__ import annotations

from typing import Dict, Tuple

# Per-crop box bounds. Each value is (lower_bound, upper_bound) for the
# decision variable named by the key. Units match the surrogate's input
# features exactly so the decoded config can be fed straight to the
# trained SurrogatePredictor.
CROP_SPECS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Tomato": {
        "min_temperature_C":   (15.0, 32.0),
        "max_temperature_C":   (15.0, 32.0),
        "avg_temperature_C":   (15.0, 32.0),
        "humidity_percent":    (60.0, 75.0),
        "co2_ppm":             (800.0, 1200.0),
        "light_intensity_lux": (20000.0, 50000.0),
        "photoperiod_hours":   (12.0, 16.0),   # ≤ 16 = chlorosis safety
        "irrigation_mm":       (4.0, 8.0),
        "soil_pH":             (5.8, 6.8),
        "fertilizer_N_kg_ha":  (100.0, 200.0),
        "fertilizer_P_kg_ha":  (40.0, 100.0),
        "fertilizer_K_kg_ha":  (150.0, 300.0),
    },
    "Cucumber": {
        "min_temperature_C":   (18.0, 35.0),
        "max_temperature_C":   (18.0, 35.0),
        "avg_temperature_C":   (18.0, 35.0),
        "humidity_percent":    (70.0, 90.0),
        "co2_ppm":             (800.0, 1200.0),
        "light_intensity_lux": (20000.0, 40000.0),
        "photoperiod_hours":   (12.0, 16.0),
        "irrigation_mm":       (5.0, 10.0),
        "soil_pH":             (5.5, 7.0),
        "fertilizer_N_kg_ha":  (80.0, 180.0),
        "fertilizer_P_kg_ha":  (40.0, 90.0),
        "fertilizer_K_kg_ha":  (120.0, 250.0),
    },
    "Lettuce": {
        "min_temperature_C":   (10.0, 27.0),
        "max_temperature_C":   (10.0, 27.0),
        "avg_temperature_C":   (10.0, 27.0),
        "humidity_percent":    (50.0, 70.0),
        "co2_ppm":             (600.0, 1000.0),
        "light_intensity_lux": (10000.0, 20000.0),
        "photoperiod_hours":   (10.0, 14.0),   # ≤ 14 = bolting safety
        "irrigation_mm":       (2.0, 5.0),
        "soil_pH":             (6.0, 7.0),
        "fertilizer_N_kg_ha":  (50.0, 120.0),
        "fertilizer_P_kg_ha":  (30.0, 80.0),
        "fertilizer_K_kg_ha":  (80.0, 180.0),
    },
    "Pepper": {
        "min_temperature_C":   (16.0, 35.0),
        "max_temperature_C":   (16.0, 35.0),
        "avg_temperature_C":   (16.0, 35.0),
        "humidity_percent":    (50.0, 70.0),
        "co2_ppm":             (800.0, 1200.0),
        "light_intensity_lux": (25000.0, 50000.0),
        "photoperiod_hours":   (12.0, 16.0),
        "irrigation_mm":       (4.0, 8.0),
        "soil_pH":             (5.8, 6.8),
        "fertilizer_N_kg_ha":  (100.0, 180.0),
        "fertilizer_P_kg_ha":  (40.0, 100.0),
        "fertilizer_K_kg_ha":  (150.0, 300.0),
    },
}

# Maximum allowed pest_severity per crop (Appendix A last row).
# Lower is better for yield, so the optimizer will naturally drive
# this to 0; the bound caps how bad we let it get for sensitivity
# analysis.
MAX_PEST_SEVERITY: Dict[str, float] = {
    "Tomato":   3.0,
    "Cucumber": 3.0,
    "Lettuce":  2.0,
    "Pepper":   3.0,
}

# Decision-variable order. Fixed once so encode/decode is deterministic
# and the same x vector means the same thing across every optimizer run.
DECISION_VARIABLES = [
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
