"""Optimizer helper functions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    """Return num / den unless den is numerically zero."""
    if abs(den) <= 1e-15:
        return default
    return num / den


def local_lipschitz(problem: Any, x_new: np.ndarray, x_old: np.ndarray, F_new: np.ndarray | None = None, F_old: np.ndarray | None = None) -> float:
    """Compute ||F(x_new)-F(x_old)||_* / ||x_new-x_old||."""
    if F_new is None:
        F_new = problem.operator(x_new)
    if F_old is None:
        F_old = problem.operator(x_old)
    dx = x_new - x_old
    den = problem.primal_norm(dx)
    if den <= 1e-15:
        return 0.0
    return problem.dual_norm(F_new - F_old) / den


def estimate_initial_step(problem: Any, x0: np.ndarray, requested: float | None = None, shrink: float = 0.5) -> float:
    """Heuristic safe initial step for proximal VI methods.

    The routine tries one full proximal step and shrinks until

        a <= 1 / (sqrt(2) * L_local),

    mirroring the ADUCA initialization condition.  It is deliberately simple and
    deterministic; users can override it through CLI hyperparameters.
    """
    if requested is not None and requested > 0:
        a = float(requested)
    else:
        # Unit-free starting point based on cost scale at x0.  If costs are huge,
        # start smaller; if costs are tiny, the backtracking loop below will still
        # keep the step safe.
        F0 = problem.operator(x0)
        scale = max(problem.dual_norm(F0), 1.0)
        a = 1.0 / scale

    F0 = problem.operator(x0)
    for _ in range(60):
        x1 = problem.prox_full(x0, F0, a)
        F1 = problem.operator(x1)
        L = local_lipschitz(problem, x1, x0, F1, F0)
        if L <= 1e-15 or a <= 1.0 / (math.sqrt(2.0) * L):
            return max(a, 1e-15)
        a *= shrink
    return max(a, 1e-15)


def estimate_coder_lhat(problem: Any, x0: np.ndarray, base_step: float | None = None) -> float:
    """Estimate CODER's cyclic block Lipschitz constant at the initial point.

    CODER's theorem uses a block-cyclic Lipschitz constant that is expensive to
    know a priori.  This deterministic estimate computes one small cyclic move
    and measures ||F(x)-p|| / ||x-x0||, where p is the partial cyclic operator.
    It is adequate as a tuning baseline; CODER-LineSearch can adapt if this is
    too low.
    """
    F0 = problem.operator(x0)
    if base_step is None or base_step <= 0:
        base_step = estimate_initial_step(problem, x0)
    x_probe = problem.prox_full(x0, F0, base_step)
    p_probe = problem.delayed_cyclic_operator(x_probe, x0)
    num = problem.dual_norm(problem.operator(x_probe) - p_probe)
    den = problem.primal_norm(x_probe - x0)
    if den <= 1e-15 or num <= 1e-15:
        # Fallback to ordinary local Lipschitz estimate.
        L = local_lipschitz(problem, x_probe, x0)
        return max(L, 1.0)
    return max(num / den, 1e-12)
