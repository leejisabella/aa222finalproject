"""Run the Genetic Algorithm on all 4 crops, multiple seeds.

Saves one OptimizerHistory JSON per (crop, seed) under results/ga/.

Usage:
    python run_ga.py                              # textbook defaults
    python run_ga.py --from-best results/best_ga.json   # promoted-from-tuning
    python run_ga.py --crop Tomato --seeds 5
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from optimization import GeneticAlgorithm, GreenhouseProblem
from surrogate.api import load_predictor


CROPS = ("Tomato", "Cucumber", "Lettuce", "Pepper")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--crop", type=str, default=None, choices=CROPS)
    p.add_argument("--pop-size", type=int, default=50)
    p.add_argument("--max-gens", type=int, default=100)
    p.add_argument("--patience-gens", type=int, default=10)
    p.add_argument("--pc", type=float, default=0.9)
    p.add_argument("--pm", type=float, default=None,
                   help="mutation probability (default: 1/dim)")
    p.add_argument("--eta-c", type=float, default=15.0)
    p.add_argument("--eta-m", type=float, default=20.0)
    p.add_argument("--from-best", type=Path, default=None,
                   help="path to best_ga.json from tune_ga.py; if present, "
                        "overrides --pop-size/--pc/--pm/--eta-c/--eta-m")
    p.add_argument("--out-dir", type=Path, default=Path("results") / "ga")
    return p.parse_args()


def _load_best_config(path: Path) -> dict:
    with path.open("r") as f:
        payload = json.load(f)
    cfg = payload.get("best_combined_config", {})
    if not cfg:
        raise ValueError(f"{path} has empty best_combined_config")
    return cfg


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    optimizer_kwargs = {
        "pop_size": args.pop_size,
        "pc": args.pc,
        "pm": args.pm,
        "eta_c": args.eta_c,
        "eta_m": args.eta_m,
    }
    if args.from_best is not None:
        best_cfg = _load_best_config(args.from_best)
        print(f"Loaded tuned config from {args.from_best}: {best_cfg}")
        optimizer_kwargs.update(best_cfg)
    print(f"Using hyperparameters: {optimizer_kwargs}")

    predictor = load_predictor()
    crops = (args.crop,) if args.crop else CROPS

    t0 = time.perf_counter()
    for crop in crops:
        problem = GreenhouseProblem(crop=crop, predictor=predictor)
        print(f"\n=== {crop}  (dim={problem.dim}, "
              f"n_constraints={len(problem.constraints)}) ===")

        for seed in range(args.seeds):
            t_run = time.perf_counter()
            ga = GeneticAlgorithm(
                problem=problem,
                max_gens=args.max_gens,
                patience_gens=args.patience_gens,
                seed=seed,
                **optimizer_kwargs,
            )
            hist = ga.run()
            out_path = args.out_dir / f"{crop}_seed{seed}.json"
            hist.to_json(out_path)
            best = hist.best_feasible_so_far()
            elapsed = time.perf_counter() - t_run
            gens = len(hist.populations) if hist.populations is not None else 0
            print(f"  seed {seed:2d}: {gens:3d} gens ({len(hist.ys):4d} evals), "
                  f"best feasible y={best:6.3f}, "
                  f"converged={hist.converged}, {elapsed:5.1f}s "
                  f"→ {out_path.name}")

    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
