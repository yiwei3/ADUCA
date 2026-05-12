"""Golden Ratio Algorithm (GRAAL) style full-vector adaptive VI method."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import HistoryLogger, OptimizerResult
from .utils import estimate_initial_step, local_lipschitz


def run_graal(
    problem: Any,
    x0: np.ndarray,
    num_iterations: int,
    a0: float | None = None,
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0,
    growth: float = 1.15,
    lipschitz_coeff: float = 0.45,
    log_every: int = 1,
) -> OptimizerResult:
    """Run a practical adaptive GRAAL variant.

    The update is

        xbar_k = ((phi - 1) x_k + xbar_{k-1}) / phi,
        x_{k+1} = prox_g(xbar_k - a_k F(x_k)),

    with a local-Lipschitz step update

        a_{k+1} = min(growth * a_k, lipschitz_coeff / L_k).

    This captures the key baseline property emphasized in the ADUCA draft: a
    full-vector method adaptive to local smoothness and requiring no line search.
    """
    if phi <= 1.0:
        raise ValueError("phi must be > 1.")
    x = problem.project_feasible(x0)
    xbar = x.copy()
    a = estimate_initial_step(problem, x, requested=a0)
    F_x = problem.operator(x)

    logger = HistoryLogger("graal", problem, log_every=log_every)
    logger.add_operator_evals(1)
    logger.maybe_log(0, x, {"stepsize": a, "local_L": 0.0}, force=True)

    for k in range(1, num_iterations + 1):
        xbar_new = ((phi - 1.0) * x + xbar) / phi
        x_next = problem.prox_full(xbar_new, F_x, a)
        F_next = problem.operator(x_next)
        logger.add_operator_evals(1)

        L = local_lipschitz(problem, x_next, x, F_next, F_x)
        if L > 1e-15:
            a_next = min(float(growth) * a, float(lipschitz_coeff) / L)
        else:
            a_next = float(growth) * a
        a_next = max(a_next, 1e-15)

        xbar, x, F_x, a = xbar_new, x_next, F_next, a_next
        logger.maybe_log(k, x, {"stepsize": a, "local_L": L})

    logger.maybe_log(num_iterations, x, {"stepsize": a, "local_L": 0.0}, force=True)
    return OptimizerResult(
        method="graal",
        x_final=x,
        history=logger.dataframe(),
        config={"a0": a0, "phi": phi, "growth": growth, "lipschitz_coeff": lipschitz_coeff},
    )
