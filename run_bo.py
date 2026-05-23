"""Run Bayesian Optimization on all 4 crops, multiple seeds.

Saves one OptimizerHistory JSON per (crop, seed) under results/bo/.

Usage:
    python run_bo.py                              # textbook defaults
    python run_bo.py --from-best results/best_bo.json   # promoted-from-tuning
    python run_bo.py --crop Tomato --seeds 5
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from optimization import BayesianOptimizer, GreenhouseProblem
from surrogate.api import load_predictor


CROPS = ("Tomato", "Cucumber", "Lettuce", "Pepper")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10,
                   help="number of random seeds per crop (default: 10)")
    p.add_argument("--crop", type=str, default=None,
                   choices=CROPS, help="run only a single crop")
    p.add_argument("--max-evals", type=int, default=200,
                   help="per-seed evaluation budget (default: 200)")
    p.add_argument("--patience", type=int, default=30,
                   help="plateau patience in iterations (default: 30)")
    p.add_argument("--kernel", default="matern52", choices=["matern52", "rbf"])
    p.add_argument("--acquisition", default="cei",
                   choices=["cei", "ei", "ucb", "pi"])
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--from-best", type=Path, default=None,
                   help="path to best_bo.json from tune_bo.py; if present, "
                        "overrides --kernel/--acquisition/--n-init/etc.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results") / "bo")
    return p.parse_args()


def _load_best_config(path: Path) -> dict:
    with path.open("r") as f:
        payload = json.load(f)
    cfg = payload.get("best_combined_config", {})
    if not cfg:
        raise ValueError(f"{path} has empty best_combined_config")
    return cfg


def main() -> None:
    warnings.filterwarnings("ignore")  # silence sklearn GP convergence chatter
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build the kwarg set the optimizer will be constructed with.
    optimizer_kwargs = {
        "kernel": args.kernel,
        "acquisition": args.acquisition,
        "n_init": args.n_init,
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
            bo = BayesianOptimizer(
                problem=problem,
                max_evals=args.max_evals,
                patience=args.patience,
                seed=seed,
                **optimizer_kwargs,
            )
            hist = bo.run()
            out_path = args.out_dir / f"{crop}_seed{seed}.json"
            hist.to_json(out_path)
            best = hist.best_feasible_so_far()
            elapsed = time.perf_counter() - t_run
            print(f"  seed {seed:2d}: {len(hist.ys):3d} evals, "
                  f"best feasible y={best:6.3f}, "
                  f"converged={hist.converged}, {elapsed:5.1f}s "
                  f"→ {out_path.name}")

    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
