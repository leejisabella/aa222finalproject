"""Bayesian Optimization with constrained Expected Improvement.

Implementation follows Kochenderfer & Wheeler, *Algorithms for
Optimization* — Chapter 16 (Gaussian-process surrogate models) and
Chapter 17 (probabilistic surrogate optimization). Constraint handling
uses the Constrained Expected Improvement criterion of

    Gardner, Kusner, Xu, Weinberger, Cunningham (2014):
    "Bayesian Optimization with Inequality Constraints"

which multiplies the standard EI by the posterior probability that
every constraint is satisfied:

    cEI(x) = EI(x) · ∏_i  Φ((0 − μ_g_i(x)) / σ_g_i(x))

One GP is fit to the objective; one additional GP is fit per coupling
constraint. The acquisition function is maximized at each step by
multi-start L-BFGS-B in normalized [0,1]^d space.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from optimization.history import OptimizerHistory
from optimization.problem import GreenhouseProblem


# =============================================================================
# Kernel factory
# =============================================================================


def _make_kernel(kind: str, dim: int) -> object:
    """Build a fresh sklearn kernel — one anisotropic length scale per dim."""
    ls = np.ones(dim)
    if kind == "matern52":
        base = Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    elif kind == "rbf":
        base = RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e2))
    else:
        raise ValueError(f"unknown kernel {kind!r}; use 'matern52' or 'rbf'")
    # ConstantKernel × base + small WhiteKernel for numerical stability.
    return (
        ConstantKernel(1.0, (1e-3, 1e3)) * base
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    )


# =============================================================================
# Acquisition functions  (operate on standardized objective values)
# =============================================================================


def _expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.0
) -> np.ndarray:
    """EI assuming MAXIMIZATION in standardized space.

    Vectorized: mu, sigma shape (n,).
    """
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - f_best - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def _probability_of_improvement(
    mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.0
) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    return norm.cdf((mu - f_best - xi) / sigma)


def _upper_confidence_bound(
    mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.0
) -> np.ndarray:
    return mu + kappa * sigma


def _feasibility_probability(
    constraint_mu: np.ndarray, constraint_sigma: np.ndarray
) -> np.ndarray:
    """P(g_i(x) ≤ 0 for all i) under independent-GP assumption.

    constraint_mu, constraint_sigma: shape (n, n_constraints), in
    *standardized* per-constraint space. Caller must pass z = (0-μ)/σ in
    the same normalized space — we do that below.
    """
    # Convert (μ, σ) at the standardized 0-threshold to P(g ≤ 0).
    sigma = np.maximum(constraint_sigma, 1e-12)
    z = (0.0 - constraint_mu) / sigma  # threshold of 0 in raw constraint units
    cdf = norm.cdf(z)                  # P(g_i ≤ 0)
    return np.prod(cdf, axis=1)


# =============================================================================
# BayesianOptimizer
# =============================================================================


@dataclass
class BayesianOptimizer:
    problem: GreenhouseProblem
    kernel: str = "matern52"        # "matern52" | "rbf"
    acquisition: str = "cei"         # "cei" | "ucb" | "pi"
    kappa: float = 2.0               # UCB exploration weight
    xi: float = 0.01                 # EI/PI improvement margin (in std units)
    n_init: int = 20
    max_evals: int = 200
    patience: int = 30
    eps_rel: float = 1e-4
    n_acq_restarts: int = 10
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.dim = self.problem.dim
        self.lo, self.hi = self.problem.bounds()
        self.span = self.hi - self.lo
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # Normalization helpers — work in [0,1]^d to keep GP kernels balanced
    # ------------------------------------------------------------------

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.lo) / self.span

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.asarray(u, dtype=float) * self.span

    # ------------------------------------------------------------------
    # Initial design — Latin Hypercube + fallback feasible samples
    # ------------------------------------------------------------------

    def _initial_design(self) -> np.ndarray:
        lhs = qmc.LatinHypercube(d=self.dim, seed=int(self._rng.integers(2**31 - 1)))
        unit = lhs.random(n=self.n_init)
        X = self._from_unit(unit)
        # If <2 are feasible, top up via rejection sampling so the GP
        # has at least a couple of feasible anchors.
        feasibles = [self.problem.is_feasible(x) for x in X]
        if sum(feasibles) < 2:
            extra: list[np.ndarray] = []
            for _ in range(10):
                try:
                    extra.append(
                        self.problem.sample_random_feasible(rng=self._rng)
                    )
                except RuntimeError:
                    break
            if extra:
                X = np.vstack([X, np.asarray(extra)])
        return X

    # ------------------------------------------------------------------
    # GP fit + predict (with standardization)
    # ------------------------------------------------------------------

    def _fit_gp(self, U: np.ndarray, y: np.ndarray) -> Tuple[
        GaussianProcessRegressor, float, float
    ]:
        """Fit a GP on unit-cube inputs and standardized outputs.

        Returns (gp, y_mean, y_std). y is standardized internally because
        sklearn's GP wants outputs centered; we keep mean/std so we can
        un-standardize predictions later.
        """
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        if y_std < 1e-12:
            y_std = 1.0
        y_norm = (y - y_mean) / y_std
        gp = GaussianProcessRegressor(
            kernel=_make_kernel(self.kernel, self.dim),
            alpha=1e-6,                # objective surrogate is near-deterministic
            normalize_y=False,         # we handle it manually
            n_restarts_optimizer=5,
            random_state=int(self._rng.integers(2**31 - 1)),
        )
        gp.fit(U, y_norm)
        return gp, y_mean, y_std

    @staticmethod
    def _predict(gp: GaussianProcessRegressor, U: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        mu_norm, sigma_norm = gp.predict(U, return_std=True)
        return np.asarray(mu_norm), np.asarray(sigma_norm)

    # ------------------------------------------------------------------
    # Acquisition function — works in standardized objective space
    # ------------------------------------------------------------------

    def _acquisition(
        self,
        U: np.ndarray,
        gp_obj: GaussianProcessRegressor,
        y_mean: float,
        y_std: float,
        f_best: float,
        gp_constraints: list[GaussianProcessRegressor],
        c_means: list[float],
        c_stds: list[float],
    ) -> np.ndarray:
        """Score a batch of candidate points U (shape (n, d), in unit cube)."""
        if U.ndim == 1:
            U = U.reshape(1, -1)
        mu_n, sig_n = self._predict(gp_obj, U)
        # Convert standardized incumbent into the same scale
        f_best_n = (f_best - y_mean) / y_std

        if self.acquisition == "ucb":
            base = _upper_confidence_bound(mu_n, sig_n, kappa=self.kappa)
        elif self.acquisition == "pi":
            base = _probability_of_improvement(mu_n, sig_n, f_best_n, xi=self.xi)
        else:  # "cei" or "ei" baseline
            base = _expected_improvement(mu_n, sig_n, f_best_n, xi=self.xi)

        if self.acquisition != "cei" or not gp_constraints:
            return base

        # Constrained EI: multiply by P(g_i ≤ 0).
        # Each constraint GP was fit on STANDARDIZED constraint values.
        # The raw threshold is g_i ≤ 0; convert to standardized threshold.
        c_mu_list, c_sig_list = [], []
        for gp_c, c_mean, c_std in zip(gp_constraints, c_means, c_stds):
            mu_c, sig_c = self._predict(gp_c, U)
            # Un-standardize so threshold "0" makes sense
            mu_c_raw = mu_c * c_std + c_mean
            sig_c_raw = sig_c * c_std
            c_mu_list.append(mu_c_raw)
            c_sig_list.append(sig_c_raw)
        c_mu_arr = np.column_stack(c_mu_list)         # (n, n_c)
        c_sig_arr = np.column_stack(c_sig_list)
        feas_prob = _feasibility_probability(c_mu_arr, c_sig_arr)
        return base * feas_prob

    # ------------------------------------------------------------------
    # Inner-loop optimization of the acquisition function
    # ------------------------------------------------------------------

    def _maximize_acquisition(
        self,
        gp_obj: GaussianProcessRegressor,
        y_mean: float,
        y_std: float,
        f_best: float,
        gp_constraints: list[GaussianProcessRegressor],
        c_means: list[float],
        c_stds: list[float],
    ) -> np.ndarray:
        """Multi-start L-BFGS-B on −acquisition. Returns the next x to evaluate."""

        def neg_acq(u_flat: np.ndarray) -> float:
            return -float(self._acquisition(
                u_flat.reshape(1, -1),
                gp_obj, y_mean, y_std, f_best,
                gp_constraints, c_means, c_stds,
            )[0])

        bounds = [(0.0, 1.0)] * self.dim
        best_u, best_val = None, np.inf

        # Multi-start: random uniform draws in the unit cube.
        starts = self._rng.uniform(0.0, 1.0, size=(self.n_acq_restarts, self.dim))
        for u0 in starts:
            try:
                res = minimize(
                    neg_acq, u0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 100, "ftol": 1e-9},
                )
            except Exception:
                continue
            if res.fun < best_val:
                best_val = float(res.fun)
                best_u = np.clip(res.x, 0.0, 1.0)

        if best_u is None:
            # Fallback: dense random search.
            U_rand = self._rng.uniform(0.0, 1.0, size=(2000, self.dim))
            scores = self._acquisition(
                U_rand, gp_obj, y_mean, y_std, f_best,
                gp_constraints, c_means, c_stds,
            )
            best_u = U_rand[int(np.argmax(scores))]

        return best_u

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> OptimizerHistory:
        t0 = time.perf_counter()
        history = OptimizerHistory(
            algorithm="BayesianOptimization",
            crop=self.problem.crop,
            seed=int(self.seed) if self.seed is not None else -1,
            config={
                "kernel": self.kernel,
                "acquisition": self.acquisition,
                "kappa": self.kappa,
                "xi": self.xi,
                "n_init": self.n_init,
                "max_evals": self.max_evals,
                "patience": self.patience,
                "eps_rel": self.eps_rel,
                "n_acq_restarts": self.n_acq_restarts,
            },
        )

        # ---- 1. Initial design (LHS, with feasible fallback) ----
        X_init = self._initial_design()
        Y_init = self.problem.objective_batch(X_init)
        G_init = self.problem.constraint_values_batch(X_init)

        X_eval: list[np.ndarray] = [x.copy() for x in X_init]
        Y_eval: list[float] = [float(y) for y in Y_init]
        G_eval: list[np.ndarray] = [g.copy() for g in G_init]

        for i, (x, y, g) in enumerate(zip(X_eval, Y_eval, G_eval)):
            feas = bool((g <= 1e-6).all()) and bool(
                np.all(x >= self.lo - 1e-6) and np.all(x <= self.hi + 1e-6)
            )
            history.log_eval(
                x=x, y=y, g=g, feasible=feas,
                iter_index=i, wall_time_s=time.perf_counter() - t0,
            )

        # ---- 2. Sequential BO loop ----
        plateau_run = 0  # # of consecutive iterations w/o meaningful improvement
        while len(X_eval) < self.max_evals:
            U = self._to_unit(np.array(X_eval))

            # Fit objective GP
            y_arr = np.asarray(Y_eval, dtype=float)
            gp_obj, y_mean, y_std = self._fit_gp(U, y_arr)

            # Fit one GP per constraint (only for cEI)
            gp_constraints: list[GaussianProcessRegressor] = []
            c_means: list[float] = []
            c_stds: list[float] = []
            if self.acquisition == "cei":
                G_arr = np.asarray(G_eval, dtype=float)  # (n, n_constraints)
                for j in range(G_arr.shape[1]):
                    gp_c, c_m, c_s = self._fit_gp(U, G_arr[:, j])
                    gp_constraints.append(gp_c)
                    c_means.append(c_m)
                    c_stds.append(c_s)

            # Best feasible objective so far ("incumbent")
            best_feas = history.best_feasible_so_far()
            if not np.isfinite(best_feas):
                # No feasible point yet — use the best observed objective
                # as a stand-in target. cEI's feasibility-probability factor
                # handles the rest (drives search toward feasibility).
                best_feas = float(np.max(y_arr))

            # Optimize acquisition → next x (in unit cube)
            u_next = self._maximize_acquisition(
                gp_obj, y_mean, y_std, best_feas,
                gp_constraints, c_means, c_stds,
            )
            x_next = self._from_unit(u_next)
            x_next = self.problem.clip_to_box(x_next)

            # Evaluate the true objective + constraints
            y_next = self.problem.objective(x_next)
            g_next = self.problem.constraint_values(x_next)
            feas_next = self.problem.is_feasible(x_next)

            X_eval.append(x_next)
            Y_eval.append(float(y_next))
            G_eval.append(g_next)

            prev_best = history.best_feasible_so_far()
            history.log_eval(
                x=x_next, y=y_next, g=g_next, feasible=feas_next,
                iter_index=len(X_eval) - 1,
                wall_time_s=time.perf_counter() - t0,
            )
            new_best = history.best_feasible_so_far()

            # Convergence check — only count once we have feasible incumbents.
            if np.isfinite(prev_best) and np.isfinite(new_best):
                rel_improve = (new_best - prev_best) / max(abs(prev_best), 1e-12)
                if rel_improve < self.eps_rel:
                    plateau_run += 1
                else:
                    plateau_run = 0
            else:
                plateau_run = 0

            if plateau_run >= self.patience:
                history.converged = True
                break

        # ---- 3. Finalize ----
        best_x = history.best_feasible_x()
        if best_x is None:
            # No feasible eval ever; report the highest-yield infeasible point.
            best_idx = int(np.argmax(history.ys))
            best_x = history.xs[best_idx]
        history.final_x = best_x
        history.final_summary = self.problem.summary(best_x)
        return history
