"""Shared optimizer-history dataclass used by BO and GA.

Both optimizers log per-evaluation data into an OptimizerHistory so the
plotting code in `plot_results.py` can be algorithm-agnostic. GA-only
fields (full populations and parent-selection masks per generation) are
kept on the same object and default to None for BO.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class OptimizerHistory:
    algorithm: str
    crop: str
    seed: int

    xs: List[np.ndarray] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    gs: List[np.ndarray] = field(default_factory=list)
    feasible: List[bool] = field(default_factory=list)
    best_feasible_y: List[float] = field(default_factory=list)
    iter_index: List[int] = field(default_factory=list)
    wall_time_s: List[float] = field(default_factory=list)

    final_x: Optional[np.ndarray] = None
    final_summary: Dict[str, Any] = field(default_factory=dict)
    converged: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    # GA-only — per-generation snapshots for the inner-mechanics plot.
    populations: Optional[List[np.ndarray]] = None
    fitnesses: Optional[List[np.ndarray]] = None
    feasibility_masks: Optional[List[np.ndarray]] = None
    parent_indices: Optional[List[np.ndarray]] = None

    # ----------------------------- helpers ---------------------------------

    def log_eval(
        self,
        x: np.ndarray,
        y: float,
        g: np.ndarray,
        feasible: bool,
        iter_index: int,
        wall_time_s: float,
    ) -> None:
        """Append one evaluation and update the running best-feasible."""
        self.xs.append(np.asarray(x, dtype=float).copy())
        self.ys.append(float(y))
        self.gs.append(np.asarray(g, dtype=float).copy())
        self.feasible.append(bool(feasible))
        self.iter_index.append(int(iter_index))
        self.wall_time_s.append(float(wall_time_s))

        prev = self.best_feasible_y[-1] if self.best_feasible_y else float("-inf")
        if not np.isfinite(prev):
            prev = float("-inf")
        if feasible and y > prev:
            self.best_feasible_y.append(float(y))
        else:
            self.best_feasible_y.append(prev if np.isfinite(prev) else float("nan"))

    def best_feasible_so_far(self) -> float:
        """Latest running best-feasible objective; nan if no feasible point yet."""
        return self.best_feasible_y[-1] if self.best_feasible_y else float("nan")

    def best_feasible_x(self) -> Optional[np.ndarray]:
        """Decision vector of the best-feasible evaluation (or None)."""
        best_y = float("-inf")
        best_x: Optional[np.ndarray] = None
        for x, y, feas in zip(self.xs, self.ys, self.feasible):
            if feas and y > best_y:
                best_y = y
                best_x = x
        return best_x

    # ----------------------------- I/O -------------------------------------

    def to_json(self, path: Path) -> None:
        """Serialize this history to JSON. Arrays become nested lists."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        # Convert numpy arrays / lists-of-arrays to plain lists.
        d["xs"] = [x.tolist() for x in self.xs]
        d["gs"] = [g.tolist() for g in self.gs]
        if self.final_x is not None:
            d["final_x"] = self.final_x.tolist()
        for k in ("populations", "fitnesses", "feasibility_masks", "parent_indices"):
            v = getattr(self, k)
            d[k] = None if v is None else [np.asarray(a).tolist() for a in v]
        with path.open("w") as f:
            json.dump(d, f, indent=2, default=_json_default)

    @classmethod
    def from_json(cls, path: Path) -> "OptimizerHistory":
        with Path(path).open("r") as f:
            d = json.load(f)
        d["xs"] = [np.asarray(x, dtype=float) for x in d["xs"]]
        d["gs"] = [np.asarray(g, dtype=float) for g in d["gs"]]
        if d.get("final_x") is not None:
            d["final_x"] = np.asarray(d["final_x"], dtype=float)
        for k in ("populations", "fitnesses", "feasibility_masks", "parent_indices"):
            if d.get(k) is not None:
                d[k] = [np.asarray(a) for a in d[k]]
        return cls(**d)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
