"""Tests for the real-coded Genetic Algorithm.

Two scopes:
  - Operator-level tests (SBX bounds, mutation bounds, Deb tournament).
  - Integration on a 2-D toy constrained problem.
"""
from __future__ import annotations

import numpy as np
import pytest

from optimization.genetic import (
    GeneticAlgorithm,
    _deb_argsort,
    deb_better,
    polynomial_mutation,
    sbx_crossover,
    tournament_select,
)
from optimization.history import OptimizerHistory


# =============================================================================
# Operator-level tests
# =============================================================================


def test_sbx_children_respect_bounds():
    rng = np.random.default_rng(0)
    lo = np.array([0.0, -1.0])
    hi = np.array([1.0, 1.0])
    p1 = np.array([0.2, -0.5])
    p2 = np.array([0.8, 0.5])
    for _ in range(50):
        c1, c2 = sbx_crossover(p1, p2, lo, hi, eta_c=15.0, pc=1.0, rng=rng)
        assert np.all(c1 >= lo) and np.all(c1 <= hi)
        assert np.all(c2 >= lo) and np.all(c2 <= hi)


def test_sbx_with_pc_zero_returns_parents_unchanged():
    rng = np.random.default_rng(0)
    p1 = np.array([0.2, 0.3, 0.4])
    p2 = np.array([0.8, 0.7, 0.6])
    c1, c2 = sbx_crossover(
        p1, p2, np.zeros(3), np.ones(3), eta_c=15.0, pc=0.0, rng=rng,
    )
    np.testing.assert_array_equal(c1, p1)
    np.testing.assert_array_equal(c2, p2)


def test_polynomial_mutation_respects_bounds():
    rng = np.random.default_rng(1)
    lo = np.array([-1.0, 0.0, 5.0])
    hi = np.array([1.0, 10.0, 20.0])
    x = np.array([0.5, 5.0, 12.0])
    for _ in range(100):
        m = polynomial_mutation(x, lo, hi, eta_m=20.0, pm=1.0, rng=rng)
        assert np.all(m >= lo)
        assert np.all(m <= hi)


def test_polynomial_mutation_with_pm_zero_is_identity():
    rng = np.random.default_rng(0)
    x = np.array([0.3, 0.5, 0.7])
    m = polynomial_mutation(x, np.zeros(3), np.ones(3),
                            eta_m=20.0, pm=0.0, rng=rng)
    np.testing.assert_array_equal(m, x)


# ---- Deb tournament ----


def test_deb_feasible_beats_infeasible():
    fitness = np.array([10.0, 20.0])  # second has higher fitness
    max_viol = np.array([0.0, 1.5])   # but is infeasible
    # Index 0 should win because feasibility dominates
    assert deb_better(0, 1, fitness, max_viol) == 0
    assert deb_better(1, 0, fitness, max_viol) == 0  # symmetric


def test_deb_among_feasible_higher_fitness_wins():
    fitness = np.array([10.0, 20.0])
    max_viol = np.array([0.0, 0.0])
    assert deb_better(0, 1, fitness, max_viol) == 1


def test_deb_among_infeasible_lower_violation_wins():
    fitness = np.array([100.0, 50.0])      # higher fitness loses
    max_viol = np.array([5.0, 2.0])         # because more violated
    assert deb_better(0, 1, fitness, max_viol) == 1


def test_tournament_select_returns_valid_index():
    rng = np.random.default_rng(0)
    fit = np.array([1.0, 2.0, 3.0, 4.0])
    vio = np.array([0.0, 0.5, 0.0, 0.0])
    for _ in range(100):
        idx = tournament_select(rng, 4, fit, vio, k=2)
        assert 0 <= idx < 4


def test_deb_argsort_feasible_first():
    fitness = np.array([5.0, 10.0, 1.0, 7.0])
    max_viol = np.array([0.0, 2.0, 0.0, 0.0])  # idx 1 is infeasible
    order = _deb_argsort(fitness, max_viol)
    # Top should be the highest-fitness feasible (idx 3, fitness 7)
    assert order[0] == 3
    # Infeasible (idx 1) should be last
    assert order[-1] == 1


# =============================================================================
# Integration: 2-D constrained toy problem
# =============================================================================


class _ToyProblem:
    """Same toy used in test_bayesian: max -((x-0.7)² + (y-0.3)²) s.t. x+y ≤ 1.5."""
    crop = "Toy"
    dim = 2

    def __init__(self) -> None:
        self.lo = np.zeros(2)
        self.hi = np.ones(2)
        self.constraints = [None]

    def bounds(self):
        return self.lo.copy(), self.hi.copy()

    def clip_to_box(self, x):
        return np.minimum(np.maximum(x, self.lo), self.hi)

    @staticmethod
    def _y(x):
        return -((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2)

    @staticmethod
    def _g(x):
        return np.array([x[0] + x[1] - 1.5])

    def objective(self, x):
        return float(self._y(x))

    def objective_batch(self, X):
        return np.array([self._y(x) for x in X])

    def constraint_values(self, x):
        return self._g(x)

    def constraint_values_batch(self, X):
        return np.array([self._g(x) for x in X])

    def is_feasible(self, x, tol=1e-6):
        return bool((self._g(x) <= tol).all()) and bool(
            np.all(x >= self.lo - tol) and np.all(x <= self.hi + tol)
        )

    def sample_random(self, rng):
        return rng.uniform(self.lo, self.hi)

    def sample_random_batch(self, n, rng):
        return rng.uniform(self.lo, self.hi, size=(n, self.dim))

    def sample_random_feasible(self, max_tries=1000, rng=None, tol=1e-6):
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(max_tries):
            x = self.sample_random(rng)
            if self.is_feasible(x, tol=tol):
                return x
        raise RuntimeError("could not find feasible toy sample")

    def summary(self, x):
        return {"x": x.tolist(), "y": self.objective(x),
                "feasible": self.is_feasible(x)}


def test_ga_converges_on_toy_problem():
    problem = _ToyProblem()
    ga = GeneticAlgorithm(
        problem=problem,                # type: ignore[arg-type]
        pop_size=30,
        max_gens=60,
        patience_gens=15,
        seed=0,
    )
    hist = ga.run()
    assert isinstance(hist, OptimizerHistory)
    assert hist.final_x is not None
    # Optimum at (0.7, 0.3); GA with pop=30 × 60 gens should find ~optimum
    assert np.linalg.norm(hist.final_x - np.array([0.7, 0.3])) < 0.1


def test_ga_logs_per_generation_data():
    problem = _ToyProblem()
    ga = GeneticAlgorithm(
        problem=problem,                # type: ignore[arg-type]
        pop_size=20, max_gens=10, patience_gens=999, seed=0,
    )
    hist = ga.run()
    # populations include gen 0 + (max_gens) further = at least 2
    assert hist.populations is not None and len(hist.populations) >= 2
    assert hist.fitnesses is not None and len(hist.fitnesses) == len(hist.populations)
    assert hist.feasibility_masks is not None
    assert hist.parent_indices is not None


def test_ga_respects_constraint_on_toy():
    problem = _ToyProblem()
    ga = GeneticAlgorithm(
        problem=problem,                # type: ignore[arg-type]
        pop_size=20, max_gens=40, patience_gens=10, seed=1,
    )
    hist = ga.run()
    assert problem.is_feasible(hist.final_x)


def test_ga_elitism_preserves_best():
    """With elitism=1, best feasible fitness is monotone non-decreasing."""
    problem = _ToyProblem()
    ga = GeneticAlgorithm(
        problem=problem,                # type: ignore[arg-type]
        pop_size=20, max_gens=20, patience_gens=999, elitism=1, seed=2,
    )
    hist = ga.run()
    # best_feasible_y is recorded per-eval, but the relevant guarantee is
    # generation-level. Skip leading NaNs (no feasible found yet).
    bfy = np.asarray(hist.best_feasible_y, dtype=float)
    bfy = bfy[np.isfinite(bfy)]
    if len(bfy) > 0:
        assert np.all(np.diff(bfy) >= -1e-9), "best-feasible decreased over time"
