"""Shared optimizer result containers and logging utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class OptimizerResult:
    """Return object for all optimizers."""

    method: str
    x_final: np.ndarray
    history: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)


class HistoryLogger:
    """Lightweight metric logger shared by all optimizers.

    The logger records both abstract progress counters and wall-clock time.
    ``logical_passes`` counts full cycles for cyclic methods or iterations for
    full-vector methods.  ``operator_evals`` counts calls to the problem's full
    operator made by the optimizer implementation.  Because the reference block
    operator currently slices the full operator, operator_evals can be larger
    than logical passes for cyclic methods.
    """

    def __init__(self, method: str, problem: Any, log_every: int = 1) -> None:
        self.method = method
        self.problem = problem
        self.log_every = max(int(log_every), 1)
        self.rows: list[dict[str, Any]] = []
        self.t0 = perf_counter()
        self.operator_evals = 0

    def add_operator_evals(self, n: int = 1) -> None:
        self.operator_evals += int(n)

    def maybe_log(self, iteration: int, x: np.ndarray, extra: dict[str, Any] | None = None, force: bool = False) -> None:
        if not force and iteration % self.log_every != 0:
            return
        # Avoid duplicate final rows when the last iteration already matched
        # log_every and the caller also requests a forced final log.
        if force and self.rows and int(self.rows[-1].get("iteration", -1)) == int(iteration):
            return
        metrics = self.problem.evaluate(x)
        row: dict[str, Any] = {
            "method": self.method,
            "iteration": int(iteration),
            "logical_passes": float(iteration),
            "operator_evals": int(self.operator_evals),
            "time_sec": float(perf_counter() - self.t0),
        }
        row.update(metrics)
        if extra:
            row.update(extra)
        self.rows.append(row)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
