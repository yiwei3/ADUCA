"""CODER and CODER-LineSearch baselines.

The fixed-step implementation follows Algorithm 3.1 of Song--Diakonikolas
(Cyclic cOordinate Dual avEraging with extRapolation).  The line-search variant
follows their parameter-free Appendix A idea: double the cyclic Lipschitz guess
until ||F(x_k)-p_k|| <= Lhat_k ||x_k-x_{k-1}||.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import HistoryLogger, OptimizerResult
from .utils import estimate_coder_lhat


def _coder_cycle(problem: Any, x0_anchor: np.ndarray, x_prev: np.ndarray, F_prev: np.ndarray, p_prev: np.ndarray, z_prev: np.ndarray, a_prev: float, a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One CODER cyclic dual-averaging cycle.

    Returns ``(x_new, p_curr, z_new)``.  ``p_curr`` is the cyclic partial operator
    assembled from the new prefix and previous suffix.
    """
    x_new = x_prev.copy()
    z_new = z_prev.copy()
    p_curr = np.zeros_like(x_prev)
    ratio = 0.0 if a <= 0 else a_prev / a
    flows = problem.link_flows(x_new)

    for b, sl in enumerate(problem.block_slices):
        # p_b = F_b(x_new^1, ..., x_new^{b-1}, x_prev^b, ..., x_prev^m)
        p_b = problem.block_operator_from_flows(flows, x_new[sl], b)
        # q_b = p_b + (a_{k-1}/a_k)(F_b(x_{k-1}) - p_{k-1,b})
        q_b = p_b + ratio * (F_prev[sl] - p_prev[sl])
        z_new[sl] = z_prev[sl] + a * q_b
        # For indicator-simplex g, prox_{A_k g}(x0 - z_k) is projection.  The
        # helper below also handles non-identity Lambda if selected.
        next_block = problem.prox_block(x0_anchor[sl], z_new[sl], step=1.0, block_id=b)
        flows = problem.apply_block_delta_to_flows(flows, b, next_block - x_new[sl])
        x_new[sl] = next_block
        p_curr[sl] = p_b
    return x_new, p_curr, z_new


def run_coder(
    problem: Any,
    x0: np.ndarray,
    num_iterations: int,
    lhat: float | None = None,
    gamma: float = 0.0,
    log_every: int = 1,
) -> OptimizerResult:
    """Run fixed-Lipschitz CODER."""
    x_anchor = problem.project_feasible(x0)
    x_prev = x_anchor.copy()
    F_prev = problem.operator(x_prev)
    p_prev = F_prev.copy()
    z_prev = np.zeros_like(x_prev)
    A_prev = 0.0
    a_prev = 0.0

    if lhat is None or lhat <= 0:
        lhat = estimate_coder_lhat(problem, x_anchor)
    lhat = float(max(lhat, 1e-12))

    logger = HistoryLogger("coder", problem, log_every=log_every)
    logger.add_operator_evals(1)
    logger.maybe_log(0, x_prev, {"lhat": lhat, "stepsize": 0.0, "A": A_prev}, force=True)

    weighted_sum = np.zeros_like(x_prev)
    for k in range(1, num_iterations + 1):
        a = (1.0 + float(gamma) * A_prev) / (2.0 * lhat)
        A = A_prev + a
        x_new, p_curr, z_new = _coder_cycle(problem, x_anchor, x_prev, F_prev, p_prev, z_prev, a_prev, a)
        logger.add_operator_evals(problem.num_blocks)
        F_new = problem.operator(x_new)
        logger.add_operator_evals(1)

        weighted_sum += a * x_new
        x_avg = weighted_sum / max(A, 1e-15)

        x_prev, F_prev, p_prev, z_prev = x_new, F_new, p_curr, z_new
        A_prev, a_prev = A, a
        logger.maybe_log(k, x_avg, {"lhat": lhat, "stepsize": a, "A": A})

    final = weighted_sum / max(A_prev, 1e-15) if A_prev > 0 else x_prev
    logger.maybe_log(num_iterations, final, {"lhat": lhat, "stepsize": a_prev, "A": A_prev}, force=True)
    return OptimizerResult(method="coder", x_final=final, history=logger.dataframe(), config={"lhat": lhat, "gamma": gamma})


def run_coder_linesearch(
    problem: Any,
    x0: np.ndarray,
    num_iterations: int,
    lhat0: float | None = None,
    gamma: float = 0.0,
    log_every: int = 1,
    max_backtracks: int = 40,
) -> OptimizerResult:
    """Run parameter-free CODER with per-cycle line search."""
    x_anchor = problem.project_feasible(x0)
    x_prev = x_anchor.copy()
    F_prev = problem.operator(x_prev)
    p_prev = F_prev.copy()
    z_prev = np.zeros_like(x_prev)
    A_prev = 0.0
    a_prev = 0.0

    if lhat0 is None or lhat0 <= 0:
        lhat_prev = max(estimate_coder_lhat(problem, x_anchor), 1e-12)
    else:
        lhat_prev = float(max(lhat0, 1e-12))

    logger = HistoryLogger("coder_linesearch", problem, log_every=log_every)
    logger.add_operator_evals(1)
    logger.maybe_log(0, x_prev, {"lhat": lhat_prev, "stepsize": 0.0, "A": A_prev, "backtracks": 0}, force=True)

    weighted_sum = np.zeros_like(x_prev)
    for k in range(1, num_iterations + 1):
        # Algorithm A.1 starts the next epoch by halving the previous estimate
        # and then doubling until the condition succeeds.
        lhat_trial = max(lhat_prev / 2.0, 1e-12)
        accepted = False
        backtracks = 0
        for bt in range(max_backtracks + 1):
            lhat_trial *= 2.0
            a = (1.0 + float(gamma) * A_prev) / (2.0 * lhat_trial)
            A = A_prev + a
            x_new, p_curr, z_new = _coder_cycle(problem, x_anchor, x_prev, F_prev, p_prev, z_prev, a_prev, a)
            logger.add_operator_evals(problem.num_blocks)
            F_new = problem.operator(x_new)
            logger.add_operator_evals(1)

            lhs = problem.dual_norm(F_new - p_curr)
            rhs = lhat_trial * problem.primal_norm(x_new - x_prev)
            if lhs <= rhs + 1e-12:
                accepted = True
                backtracks = bt
                break
        if not accepted:
            raise RuntimeError("CODER-LineSearch failed to satisfy the Lipschitz condition.")

        weighted_sum += a * x_new
        x_avg = weighted_sum / max(A, 1e-15)
        x_prev, F_prev, p_prev, z_prev = x_new, F_new, p_curr, z_new
        A_prev, a_prev, lhat_prev = A, a, lhat_trial
        logger.maybe_log(k, x_avg, {"lhat": lhat_prev, "stepsize": a, "A": A, "backtracks": backtracks})

    final = weighted_sum / max(A_prev, 1e-15) if A_prev > 0 else x_prev
    logger.maybe_log(num_iterations, final, {"lhat": lhat_prev, "stepsize": a_prev, "A": A_prev, "backtracks": 0}, force=True)
    return OptimizerResult(
        method="coder_linesearch",
        x_final=final,
        history=logger.dataframe(),
        config={"lhat0": lhat0, "gamma": gamma, "final_lhat": lhat_prev},
    )
