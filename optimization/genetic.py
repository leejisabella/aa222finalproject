"""Real-coded Genetic Algorithm with SBX + polynomial mutation.

Implementation follows Kochenderfer & Wheeler, *Algorithms for
Optimization* — Chapter 9 ("Population Methods"). Constraint handling
uses Deb's constrained-domination tournament:

    Deb (2000): "An efficient constraint handling method for
    genetic algorithms"

which orders individuals by:
  1. feasible ≻ infeasible
  2. among feasible: higher objective ≻ lower
  3. among infeasible: lower max-violation ≻ higher

Crossover and mutation are the standard Deb operators:
  - Simulated Binary Crossover (Deb & Agrawal 1995)
  - Polynomial Mutation (Deb & Goyal 1996)
Both respect the per-variable box bounds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from optimization.history import OptimizerHistory
from optimization.problem import GreenhouseProblem


# =============================================================================
# Variation operators
# =============================================================================


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    eta_c: float,
    pc: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (bound-respecting).

    Each variable independently undergoes SBX with probability pc;
    otherwise the child copies the parent value. eta_c is the
    distribution index — larger ↔ children closer to parents.

    Returns two children, each within [lo, hi].
    """
    dim = p1.shape[0]
    c1 = p1.copy()
    c2 = p2.copy()

    for i in range(dim):
        if rng.random() > pc:
            continue
        if abs(p1[i] - p2[i]) < 1e-14:
            continue  # parents identical → no crossover

        x1 = min(p1[i], p2[i])
        x2 = max(p1[i], p2[i])
        xl, xu = lo[i], hi[i]

        # Bound-respecting SBX (Deb's variant).
        u = rng.random()
        # ---- child 1 (closer to lower parent) ----
        beta = 1.0 + (2.0 * (x1 - xl) / max(x2 - x1, 1e-14))
        alpha = 2.0 - beta ** -(eta_c + 1.0)
        if u <= 1.0 / alpha:
            betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
        ch1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        # ---- child 2 (closer to upper parent) ----
        beta = 1.0 + (2.0 * (xu - x2) / max(x2 - x1, 1e-14))
        alpha = 2.0 - beta ** -(eta_c + 1.0)
        if u <= 1.0 / alpha:
            betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
        ch2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        # Clip + random swap to avoid systematic bias toward c1.
        ch1 = float(np.clip(ch1, xl, xu))
        ch2 = float(np.clip(ch2, xl, xu))
        if rng.random() < 0.5:
            c1[i], c2[i] = ch1, ch2
        else:
            c1[i], c2[i] = ch2, ch1

    return c1, c2


def polynomial_mutation(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    eta_m: float,
    pm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bound-respecting polynomial mutation. eta_m larger → smaller perturbations."""
    dim = x.shape[0]
    out = x.copy()
    for i in range(dim):
        if rng.random() > pm:
            continue
        xl, xu = lo[i], hi[i]
        if xu <= xl:
            continue
        delta1 = (out[i] - xl) / (xu - xl)
        delta2 = (xu - out[i]) / (xu - xl)
        u = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)
        if u < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
            deltaq = 1.0 - val ** mut_pow
        out[i] = out[i] + deltaq * (xu - xl)
        out[i] = float(np.clip(out[i], xl, xu))
    return out


# =============================================================================
# Deb's constrained-domination tournament
# =============================================================================


def deb_better(
    i: int, j: int,
    fitnesses: np.ndarray, max_viol: np.ndarray,
) -> int:
    """Return the index (i or j) that wins under Deb's feasibility tournament.

    fitnesses[k]  — higher is better (yield).
    max_viol[k]   — max constraint violation; 0 means feasible.
    """
    fi, fj = fitnesses[i], fitnesses[j]
    vi, vj = max_viol[i], max_viol[j]
    i_feas = vi <= 1e-9
    j_feas = vj <= 1e-9

    if i_feas and not j_feas:
        return i
    if j_feas and not i_feas:
        return j
    if i_feas and j_feas:
        return i if fi >= fj else j
    # both infeasible — prefer less violation
    return i if vi <= vj else j


def tournament_select(
    rng: np.random.Generator,
    pop_size: int,
    fitnesses: np.ndarray,
    max_viol: np.ndarray,
    k: int = 2,
) -> int:
    """Sample k contestants uniformly, return the index of the Deb-best."""
    idx = rng.integers(0, pop_size, size=k)
    winner = int(idx[0])
    for c in idx[1:]:
        winner = deb_better(winner, int(c), fitnesses, max_viol)
    return winner


# =============================================================================
# GeneticAlgorithm
# =============================================================================


@dataclass
class GeneticAlgorithm:
    problem: GreenhouseProblem
    pop_size: int = 50
    max_gens: int = 100
    pc: float = 0.9
    pm: Optional[float] = None     # None → 1/dim
    eta_c: float = 15.0
    eta_m: float = 20.0
    tournament_k: int = 2
    elitism: int = 1
    patience_gens: int = 10
    eps_rel: float = 1e-4
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.dim = self.problem.dim
        self.lo, self.hi = self.problem.bounds()
        self._rng = np.random.default_rng(self.seed)
        if self.pm is None:
            self.pm = 1.0 / self.dim

    # ------------------------------------------------------------------
    # Initial population: half feasible (where reachable), half random
    # ------------------------------------------------------------------

    def _initial_population(self) -> np.ndarray:
        n_feasible = self.pop_size // 2
        feas: list[np.ndarray] = []
        for _ in range(n_feasible):
            try:
                feas.append(self.problem.sample_random_feasible(rng=self._rng))
            except RuntimeError:
                break
        n_rand = self.pop_size - len(feas)
        rand = self.problem.sample_random_batch(n_rand, rng=self._rng)
        if feas:
            pop = np.vstack([np.asarray(feas), rand])
        else:
            pop = rand
        return pop[: self.pop_size]

    # ------------------------------------------------------------------
    # Evaluation: batch objective + constraints
    # ------------------------------------------------------------------

    def _evaluate(self, pop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (fitness, max_violation, feasibility_mask)."""
        fitness = self.problem.objective_batch(pop)
        G = self.problem.constraint_values_batch(pop)             # (n, n_c)
        max_viol = np.maximum(0.0, G).max(axis=1)                  # (n,)
        # Box-bound violation (rare since SBX/mutation clip, but be safe):
        box_low = np.maximum(0.0, self.lo - pop).max(axis=1)
        box_hi = np.maximum(0.0, pop - self.hi).max(axis=1)
        max_viol = np.maximum.reduce([max_viol, box_low, box_hi])
        feasible = max_viol <= 1e-9
        return fitness, max_viol, feasible

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> OptimizerHistory:
        t0 = time.perf_counter()
        history = OptimizerHistory(
            algorithm="GeneticAlgorithm",
            crop=self.problem.crop,
            seed=int(self.seed) if self.seed is not None else -1,
            config={
                "pop_size": self.pop_size,
                "max_gens": self.max_gens,
                "pc": self.pc,
                "pm": self.pm,
                "eta_c": self.eta_c,
                "eta_m": self.eta_m,
                "tournament_k": self.tournament_k,
                "elitism": self.elitism,
                "patience_gens": self.patience_gens,
                "eps_rel": self.eps_rel,
            },
            populations=[],
            fitnesses=[],
            feasibility_masks=[],
            parent_indices=[],
        )

        pop = self._initial_population()
        fitness, max_viol, feas_mask = self._evaluate(pop)

        # Log generation 0
        for k in range(self.pop_size):
            history.log_eval(
                x=pop[k], y=float(fitness[k]),
                g=np.array([max_viol[k]]),                # store max-violation per eval
                feasible=bool(feas_mask[k]),
                iter_index=0,
                wall_time_s=time.perf_counter() - t0,
            )
        history.populations.append(pop.copy())
        history.fitnesses.append(fitness.copy())
        history.feasibility_masks.append(feas_mask.copy())
        history.parent_indices.append(np.array([], dtype=int))  # no parents yet

        plateau_run = 0  # # of consecutive generations with no real improvement

        for gen in range(1, self.max_gens + 1):
            prev_best = history.best_feasible_so_far()

            # ---- Selection: produce a mating pool of pop_size parents ----
            parent_idx = np.array([
                tournament_select(self._rng, self.pop_size,
                                  fitness, max_viol, k=self.tournament_k)
                for _ in range(self.pop_size)
            ], dtype=int)
            parents = pop[parent_idx]

            # ---- Crossover + Mutation ----
            children = np.empty_like(pop)
            for i in range(0, self.pop_size, 2):
                p1 = parents[i]
                p2 = parents[i + 1] if i + 1 < self.pop_size else parents[0]
                c1, c2 = sbx_crossover(p1, p2, self.lo, self.hi,
                                       self.eta_c, self.pc, self._rng)
                c1 = polynomial_mutation(c1, self.lo, self.hi,
                                         self.eta_m, self.pm, self._rng)
                c2 = polynomial_mutation(c2, self.lo, self.hi,
                                         self.eta_m, self.pm, self._rng)
                children[i] = c1
                if i + 1 < self.pop_size:
                    children[i + 1] = c2

            # ---- Evaluate children ----
            child_fit, child_viol, child_feas = self._evaluate(children)

            # ---- Elitism: keep top-`elitism` from previous gen (by Deb-rank) ----
            if self.elitism > 0:
                rank_order = _deb_argsort(fitness, max_viol)
                elite_idx = rank_order[: self.elitism]
                elite_pop = pop[elite_idx]
                elite_fit = fitness[elite_idx]
                elite_viol = max_viol[elite_idx]
                elite_feas = feas_mask[elite_idx]

                # Replace the worst children with the elites
                child_rank = _deb_argsort(child_fit, child_viol)
                worst_idx = child_rank[-self.elitism:]
                children[worst_idx] = elite_pop
                child_fit[worst_idx] = elite_fit
                child_viol[worst_idx] = elite_viol
                child_feas[worst_idx] = elite_feas

            pop = children
            fitness = child_fit
            max_viol = child_viol
            feas_mask = child_feas

            # ---- Log generation `gen` ----
            for k in range(self.pop_size):
                history.log_eval(
                    x=pop[k], y=float(fitness[k]),
                    g=np.array([max_viol[k]]),
                    feasible=bool(feas_mask[k]),
                    iter_index=gen,
                    wall_time_s=time.perf_counter() - t0,
                )
            history.populations.append(pop.copy())
            history.fitnesses.append(fitness.copy())
            history.feasibility_masks.append(feas_mask.copy())
            history.parent_indices.append(parent_idx.copy())

            new_best = history.best_feasible_so_far()
            # Convergence — only count after we have feasible solutions
            if np.isfinite(prev_best) and np.isfinite(new_best):
                rel_improve = (new_best - prev_best) / max(abs(prev_best), 1e-12)
                if rel_improve < self.eps_rel:
                    plateau_run += 1
                else:
                    plateau_run = 0
            else:
                plateau_run = 0

            if plateau_run >= self.patience_gens:
                history.converged = True
                break

        # ---- Finalize ----
        best_x = history.best_feasible_x()
        if best_x is None:
            # No feasible eval ever; pick highest-yield infeasible point.
            best_idx = int(np.argmax(history.ys))
            best_x = history.xs[best_idx]
        history.final_x = best_x
        history.final_summary = self.problem.summary(best_x)
        return history


def _deb_argsort(fitness: np.ndarray, max_viol: np.ndarray) -> np.ndarray:
    """Argsort indices best→worst under Deb's constrained domination.

    Key: feasible first (sorted by -fitness), then infeasible (sorted by violation).
    """
    feasible = max_viol <= 1e-9
    feas_idx = np.where(feasible)[0]
    infeas_idx = np.where(~feasible)[0]
    feas_order = feas_idx[np.argsort(-fitness[feas_idx])]
    infeas_order = infeas_idx[np.argsort(max_viol[infeas_idx])]
    return np.concatenate([feas_order, infeas_order])
