"""Proximal Cyclic Coordinate Method (PCCM) baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import HistoryLogger, OptimizerResult
from .utils import estimate_initial_step


def run_pccm(
    problem: Any,
    x0: np.ndarray,
    num_iterations: int,
    stepsize: float | None = None,
    log_every: int = 1,
) -> OptimizerResult:
    """Run a straightforward cyclic proximal coordinate method.

    Each cycle visits OD blocks in order.  For block b, it evaluates F_b at the
    current partially updated vector and performs a proximal step over the OD
    simplex.  This baseline is intentionally simple and mirrors the "natural"
    cyclic method discussed as PCCM in the ADUCA draft.
    """
    x = problem.project_feasible(x0)
    a = estimate_initial_step(problem, x, requested=stepsize) if stepsize is None or stepsize <= 0 else float(stepsize)

    logger = HistoryLogger("pccm", problem, log_every=log_every)
    logger.maybe_log(0, x, {"stepsize": a}, force=True)

    for k in range(1, num_iterations + 1):
        partial = x.copy()
        flows = problem.link_flows(partial)
        for b, sl in enumerate(problem.block_slices):
            fb = problem.block_operator_from_flows(flows, partial[sl], b)
            logger.add_operator_evals(1)
            next_block = problem.prox_block(partial[sl], fb, a, b)
            flows = problem.apply_block_delta_to_flows(flows, b, next_block - partial[sl])
            partial[sl] = next_block
        x = partial
        logger.maybe_log(k, x, {"stepsize": a})

    logger.maybe_log(num_iterations, x, {"stepsize": a}, force=True)
    return OptimizerResult(method="pccm", x_final=x, history=logger.dataframe(), config={"stepsize": a})
