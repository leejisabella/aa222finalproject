"""Pick the best hyperparameters from a per-sweep tuning JSON.

Both `tune_bo.py` and `tune_ga.py` do one-at-a-time (OAT) sweeps, e.g.
sweeping `kernel` while holding everything else at its default, then
sweeping `acquisition` the same way. This module provides
`select_best_config(tuning_json)`:

  1. For each sweep, score every config by the median of its per-seed
     best-feasible final yield.
  2. Pick the highest-scoring config per sweep.
  3. Merge the winners across sweeps into one combined config — this is
     the OAT independence assumption (well-known limitation; documented
     in the writeup).
  4. Return both the merged config and the per-sweep winners (for
     diagnostics).

Why median: it's the robust summary statistic from the convergence plots
already shown in §C of the plan. Mean would over-weight outlier seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _score_run(run: dict) -> float:
    """Final best-feasible yield for a single seeded run."""
    bfy = run.get("best_feasible_y", [])
    if not bfy:
        return float("-inf")
    # last finite value in the running best-feasible array
    arr = np.asarray(bfy, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("-inf")
    return float(finite[-1])


def _score_config(entry: dict) -> float:
    """Median final best-feasible yield across seeds for one config."""
    runs = entry.get("runs", [])
    if not runs:
        return float("-inf")
    return float(np.median([_score_run(r) for r in runs]))


def select_best_config(
    tuning_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Read a tuning JSON, return (best_combined_config, per_sweep_winners).

    Parameters
    ----------
    tuning_path
        Path to results/tuning_bo.json or results/tuning_ga.json.

    Returns
    -------
    best_combined
        dict ready to spread into a *Optimizer constructor.
        E.g. for BO: {"kernel": "matern52", "acquisition": "cei", "n_init": 20}.
    per_sweep
        dict mapping sweep_name → {"winner": cfg, "score": float, "alternatives": [...]}.
        Useful for the writeup.
    """
    with Path(tuning_path).open("r") as f:
        data = json.load(f)
    sweeps = data.get("sweeps", {})

    per_sweep: Dict[str, Dict[str, Any]] = {}

    # First pass: pick the winner of each sweep.
    for sweep_name, entries in sweeps.items():
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry in entries:
            score = _score_config(entry)
            scored.append((score, entry["config"]))
        if not scored:
            continue
        scored.sort(key=lambda t: t[0], reverse=True)
        winner_score, winner_cfg = scored[0]
        per_sweep[sweep_name] = {
            "winner": winner_cfg,
            "score": winner_score,
            "alternatives": [
                {"config": c, "score": s} for s, c in scored[1:]
            ],
        }

    # Second pass: merge winners into one combined config. For each
    # knob key, take the value from the sweep whose WINNER has the
    # highest score. This prevents a narrow sub-sweep (e.g.
    # "ucb_kappa" — which tunes a UCB-specific knob in isolation
    # from cEI) from overriding a globally-better choice picked by
    # a broader sweep (e.g. "acquisition", which compares cEI vs
    # UCB vs PI at default settings).
    best_combined: Dict[str, Any] = {}
    key_source_score: Dict[str, float] = {}
    for sweep_name, info in per_sweep.items():
        score = info["score"]
        for k, v in info["winner"].items():
            if k not in best_combined or score > key_source_score[k]:
                best_combined[k] = v
                key_source_score[k] = score

    return best_combined, per_sweep
