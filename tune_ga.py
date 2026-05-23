"""Hyperparameter sweep for the Genetic Algorithm on Tomato.

One-at-a-time sweeps over pop_size, pc, pm, eta_c, eta_m.
Each config run with 3 seeds. Saves results to results/tuning_ga.json
for plot_results.py.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

from optimization import GeneticAlgorithm, GreenhouseProblem
from optimization.tuning_select import select_best_config
from surrogate.api import load_predictor


# Decision-vector dim is 14 → "default" pm = 1/14.
SWEEPS: Dict[str, List[Dict[str, Any]]] = {
    "pop_size": [
        {"pop_size": 20},
        {"pop_size": 50},
        {"pop_size": 100},
        {"pop_size": 200},
    ],
    "pc": [
        {"pc": 0.6},
        {"pc": 0.8},
        {"pc": 0.9},
        {"pc": 1.0},
    ],
    "pm": [
        {"pm": 1.0 / 28.0},   # 1/(2·dim)
        {"pm": 1.0 / 14.0},   # 1/dim   (default)
        {"pm": 2.0 / 14.0},   # 2/dim
    ],
    "eta_c": [
        {"eta_c": 2.0},
        {"eta_c": 15.0},
        {"eta_c": 30.0},
    ],
    "eta_m": [
        {"eta_m": 5.0},
        {"eta_m": 20.0},
        {"eta_m": 50.0},
    ],
}


def history_to_summary(hist) -> Dict[str, Any]:
    return {
        "best_feasible_y": list(hist.best_feasible_y),
        "wall_time_s": list(hist.wall_time_s),
        "iter_index": list(hist.iter_index),
        "feasible_mask": list(hist.feasible),
        "final_y": hist.final_summary.get("yield_kg_per_m2"),
        "final_feasible": hist.final_summary.get("feasible"),
        "converged": hist.converged,
        "config": hist.config,
        "n_evals": len(hist.ys),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-gens", type=int, default=60,
                   help="generations per run (default: 60, ~3000 evals at pop=50)")
    p.add_argument("--patience-gens", type=int, default=10)
    p.add_argument("--crop", default="Tomato")
    p.add_argument("--out", type=Path, default=Path("results") / "tuning_ga.json")
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor()
    problem = GreenhouseProblem(crop=args.crop, predictor=predictor)

    out: Dict[str, Any] = {"crop": args.crop, "seeds": args.seeds,
                           "max_gens": args.max_gens,
                           "patience_gens": args.patience_gens,
                           "sweeps": {}}

    t0 = time.perf_counter()
    for sweep_name, configs in SWEEPS.items():
        print(f"\n--- Sweeping {sweep_name} ({len(configs)} configs) ---")
        out["sweeps"][sweep_name] = []
        for cfg in configs:
            runs = []
            for seed in range(args.seeds):
                t_run = time.perf_counter()
                ga = GeneticAlgorithm(
                    problem=problem,
                    max_gens=args.max_gens,
                    patience_gens=args.patience_gens,
                    seed=seed,
                    **cfg,
                )
                hist = ga.run()
                runs.append(history_to_summary(hist))
                print(f"  cfg={cfg} seed={seed}: "
                      f"best={hist.best_feasible_so_far():.3f}, "
                      f"{time.perf_counter() - t_run:.1f}s")
            out["sweeps"][sweep_name].append({"config": cfg, "runs": runs})

    out["wall_time_total_s"] = time.perf_counter() - t0
    with args.out.open("w") as f:
        json.dump(out, f, indent=2, default=lambda v: float(v)
                  if hasattr(v, "__float__") else None)
    print(f"\nSaved → {args.out}  ({out['wall_time_total_s']:.1f}s total)")

    # ---- Promote: pick the best config per sweep, write best_ga.json ----
    best, per_sweep = select_best_config(args.out)
    best_path = args.out.parent / "best_ga.json"
    payload = {
        "best_combined_config": best,
        "per_sweep_winners": per_sweep,
        "source_tuning_file": str(args.out),
    }
    with best_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nPromoted best config → {best_path}")
    print(f"  best_combined: {best}")
    for sweep_name, info in per_sweep.items():
        print(f"  [{sweep_name}] winner={info['winner']} "
              f"score={info['score']:.3f}")


if __name__ == "__main__":
    main()
