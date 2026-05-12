"""Projection and proximal operators for simplex-constrained path flows.

The traffic equilibrium model is represented in *path-flow* variables.  For each
origin--destination (OD) pair r, the feasible set is a scaled simplex

    X_r = {x_r >= 0 : 1^T x_r = demand_r}.

All optimizers in this package repeatedly solve proximal problems of the form

    argmin_{x_r in X_r}  <a * c_r, x_r> + 1/2 ||x_r - center_r||^2_Lambda,

where Lambda is diagonal and positive.  This file keeps those prox/projection
routines in one place so the problem model and algorithms remain clean.
"""

from __future__ import annotations

import numpy as np


def project_simplex_euclidean(y: np.ndarray, radius: float) -> np.ndarray:
    """Project ``y`` onto {x >= 0, sum(x) = radius} in Euclidean norm.

    This is the O(n log n) sorting algorithm of Duchi et al.  It is numerically
    stable for the small/medium OD block sizes produced by k-shortest paths.

    Parameters
    ----------
    y:
        Vector to project.
    radius:
        Nonnegative simplex radius.  In this experiment it is the OD demand.

    Returns
    -------
    np.ndarray
        The projected vector.
    """
    y = np.asarray(y, dtype=float)
    if radius < 0:
        raise ValueError("Simplex radius must be nonnegative.")
    if y.size == 0:
        return y.copy()
    if radius == 0:
        return np.zeros_like(y)

    # Sort descending and find the unique threshold theta such that
    # sum(max(y_i - theta, 0)) = radius.
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - radius
    ind = np.arange(1, y.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        # This only occurs in extreme floating-point corner cases.  The uniform
        # point is feasible and harmless as a fallback.
        return np.full_like(y, radius / y.size)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(y - theta, 0.0)


def project_simplex_weighted(y: np.ndarray, radius: float, weights: np.ndarray) -> np.ndarray:
    """Project ``y`` onto a scaled simplex under a diagonal weighted norm.

    Computes

        argmin_x 1/2 * sum_i weights_i * (x_i - y_i)^2
        s.t.     x_i >= 0, sum_i x_i = radius.

    KKT conditions give

        x_i(tau) = max(y_i - tau / weights_i, 0),

    and the scalar tau is chosen by bisection so that sum_i x_i(tau) = radius.
    This routine is used when ADUCA is run with a nontrivial diagonal Lambda.
    """
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if y.shape != weights.shape:
        raise ValueError("y and weights must have the same shape.")
    if np.any(weights <= 0):
        raise ValueError("All projection weights must be strictly positive.")
    if radius < 0:
        raise ValueError("Simplex radius must be nonnegative.")
    if y.size == 0:
        return y.copy()
    if radius == 0:
        return np.zeros_like(y)

    # If all weights are effectively equal, use the faster exact Euclidean path.
    if np.allclose(weights, weights[0]):
        return project_simplex_euclidean(y, radius)

    # tau_low makes every coordinate active and sum >= radius; tau_high makes
    # every coordinate zero and sum <= radius.  The function is monotone.
    tau_low = np.min(weights * (y - radius))
    tau_high = np.max(weights * y)

    # Expand the bracket defensively for poorly scaled data.
    def total(tau: float) -> float:
        return float(np.maximum(y - tau / weights, 0.0).sum())

    while total(tau_low) < radius:
        tau_low *= 2.0
    while total(tau_high) > radius:
        tau_high *= 2.0

    for _ in range(100):
        tau_mid = 0.5 * (tau_low + tau_high)
        if total(tau_mid) > radius:
            tau_low = tau_mid
        else:
            tau_high = tau_mid

    x = np.maximum(y - tau_high / weights, 0.0)
    # Final mass correction removes tiny bisection error without changing the
    # point meaningfully.  Assign to active coordinates when possible.
    err = radius - float(x.sum())
    if abs(err) > 1e-10:
        active = x > 1e-12
        if np.any(active):
            x[active] += err / active.sum()
            x = np.maximum(x, 0.0)
        # Re-normalize if clipping changed the sum again.
        s = x.sum()
        if s > 0:
            x *= radius / s
        else:
            x[:] = radius / x.size
    return x


def prox_scaled_simplex_block(
    center: np.ndarray,
    direction: np.ndarray,
    step: float,
    radius: float,
    lambda_diag: np.ndarray | None = None,
) -> np.ndarray:
    """Prox step over one OD simplex block.

    Solves

        argmin_{x >= 0, 1^T x = radius}
            step * <direction, x> + 1/2 ||x - center||^2_Lambda.

    If ``lambda_diag`` is omitted, Lambda is the identity and the problem reduces
    to Euclidean projection of ``center - step * direction``.  If Lambda is
    supplied, the linear term is scaled by Lambda^{-1} before weighted
    projection.
    """
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if center.shape != direction.shape:
        raise ValueError("center and direction must have the same shape.")
    if step < 0:
        raise ValueError("step must be nonnegative.")

    if lambda_diag is None:
        shifted = center - step * direction
        return project_simplex_euclidean(shifted, radius)

    lambda_diag = np.asarray(lambda_diag, dtype=float)
    shifted = center - step * direction / lambda_diag
    return project_simplex_weighted(shifted, radius, lambda_diag)
