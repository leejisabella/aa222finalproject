"""Tests for the Bayesian Optimization implementation.

Two scopes:
  - Operator-level tests (EI/PI/UCB math, GP fit sanity) — fast, no surrogate.
  - Integration on a 2-D toy constrained problem — verifies convergence.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from optimization.bayesian import (
    BayesianOptimizer,
    _expected_improvement,
    _feasibility_probability,
    _make_kernel,
    _probability_of_improvement,
    _upper_confidence_bound,
)
from optimization.history import OptimizerHistory


# =============================================================================
# Operator-level tests
# =============================================================================


def test_ei_zero_at_incumbent_with_no_uncertainty():
    """If μ = f_best and σ = 0, EI = 0 (no improvement possible)."""
    mu = np.array([5.0])
    sigma = np.array([0.0])
    ei = _expected_improvement(mu, sigma, f_best=5.0, xi=0.0)
    assert ei[0] == pytest.approx(0.0, abs=1e-9)


def test_ei_positive_when_predicted_higher_than_incumbent():
    """μ > f_best with any σ → EI > 0."""
    mu = np.array([6.0])
    sigma = np.array([0.5])
    ei = _expected_improvement(mu, sigma, f_best=5.0, xi=0.0)
    assert ei[0] > 0


def test_ei_zero_with_xi_blocking_improvement():
    """xi penalty equal to the improvement → EI ≈ 0.5 · σ · φ(0). Strict zero only when σ=0."""
    mu = np.array([6.0])
    sigma = np.array([0.0])
    ei = _expected_improvement(mu, sigma, f_best=5.0, xi=1.0)  # improvement margin = 1.0
    # mu - f_best - xi = 0 and σ=0 → z=0/eps → improvement·Φ ≈ 0, σ·φ = 0
    assert ei[0] == pytest.approx(0.0, abs=1e-6)


def test_pi_strictly_between_zero_and_one():
    mu = np.linspace(-1, 6, 11)
    sigma = np.full_like(mu, 0.5)
    pi = _probability_of_improvement(mu, sigma, f_best=2.0)
    assert np.all(pi >= 0.0) and np.all(pi <= 1.0)


def test_ucb_increases_with_kappa():
    mu = np.array([1.0])
    sigma = np.array([0.5])
    low = _upper_confidence_bound(mu, sigma, kappa=0.1)
    high = _upper_confidence_bound(mu, sigma, kappa=5.0)
    assert high[0] > low[0]


def test_feasibility_probability_drops_at_infeasible_mean():
    """μ_g ≫ 0 (clear violation) → P(g ≤ 0) ≈ 0."""
    mu_c = np.array([[10.0, 0.0]])
    sig_c = np.array([[1.0, 1.0]])
    p = _feasibility_probability(mu_c, sig_c)
    assert p[0] < 0.01  # first constraint kills it


def test_feasibility_probability_high_at_feasible_mean():
    """μ_g ≪ 0 (well inside feasible) → P(g ≤ 0) ≈ 1."""
    mu_c = np.array([[-5.0, -5.0]])
    sig_c = np.array([[1.0, 1.0]])
    p = _feasibility_probability(mu_c, sig_c)
    assert p[0] > 0.99


def test_kernel_factory_supports_matern_and_rbf():
    k1 = _make_kernel("matern52", dim=3)
    k2 = _make_kernel("rbf", dim=3)
    assert k1 is not k2  # different objects
    with pytest.raises(ValueError):
        _make_kernel("unknown_kernel", dim=3)


def test_gp_fit_recovers_known_function():
    """GP on a smooth 1-D function should predict the training points accurately."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(20, 1))
    y = np.sin(2 * np.pi * X.ravel())
    gp = GaussianProcessRegressor(
        kernel=Matern(length_scale=0.2, nu=2.5),
        alpha=1e-6, n_restarts_optimizer=3, random_state=0,
    )
    gp.fit(X, y)
    pred = gp.predict(X)
    assert np.allclose(pred, y, atol=1e-3)  # near-interpolation


# =============================================================================
# Integration: 2-D constrained toy problem
# =============================================================================


class _ToyProblem:
    """Minimal interface compatible with BayesianOptimizer for testing.

    Maximize y = -((x[0] - 0.7)^2 + (x[1] - 0.3)^2)   ← optimum at (0.7, 0.3)
    Subject to:  g(x) = x[0] + x[1] - 1.5 ≤ 0          ← linear half-plane
    Box: x ∈ [0, 1]^2
    """
    crop = "Toy"
    dim = 2

    def __init__(self) -> None:
        self.lo = np.zeros(2)
        self.hi = np.ones(2)
        # Stub for OptimizerHistory expectations
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


def test_bo_converges_on_toy_problem():
    problem = _ToyProblem()
    bo = BayesianOptimizer(
        problem=problem,                # type: ignore[arg-type]
        kernel="matern52",
        acquisition="cei",
        n_init=8,
        max_evals=40,
        patience=15,
        seed=42,
    )
    hist = bo.run()
    assert isinstance(hist, OptimizerHistory)
    assert hist.final_x is not None
    # Optimum is at (0.7, 0.3) and is feasible (0.7+0.3=1.0 ≤ 1.5).
    # 40 evals on a 2-D problem should get us within 0.1 of the optimum.
    assert np.linalg.norm(hist.final_x - np.array([0.7, 0.3])) < 0.15


def test_bo_respects_constraint_on_toy():
    problem = _ToyProblem()
    bo = BayesianOptimizer(
        problem=problem,                # type: ignore[arg-type]
        acquisition="cei",
        n_init=8,
        max_evals=30,
        seed=0,
    )
    hist = bo.run()
    # Final reported solution should be feasible
    assert problem.is_feasible(hist.final_x)
