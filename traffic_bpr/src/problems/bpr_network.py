"""Path-flow traffic equilibrium VI with nonlinear BPR congestion costs.

The model implemented here is a finite-dimensional path-flow version of the
static user-equilibrium traffic assignment problem.  For each OD pair r, path
flows x_r live on a simplex with total mass equal to OD demand D_r.  Link flows
are f = A x, where A is the link-path incidence matrix.  Link travel times use
the Bureau of Public Roads (BPR) volume-delay form

    t_e(f_e) = t0_e * (1 + alpha_e * (f_e / capacity_e) ** beta_e).

The VI operator is the vector of path costs

    F(x) = A^T t(Ax) + path_regularization * x.

The optional path regularization is normally zero; setting it positive makes the
path-flow operator strictly monotone in directions invisible to link flows, which
can be useful for controlled strongly monotone stress tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from .simplex import prox_scaled_simplex_block


@dataclass(frozen=True)
class LinkData:
    """Directed network link attributes.

    All arrays have length ``num_links`` and are ordered consistently with the
    link IDs used in generated paths.
    """

    tail: np.ndarray
    head: np.ndarray
    capacity: np.ndarray
    free_flow_time: np.ndarray
    alpha: np.ndarray
    power: np.ndarray
    length: np.ndarray | None = None
    raw_link_ids: np.ndarray | None = None


@dataclass(frozen=True)
class ODData:
    """Origin--destination demand table."""

    origin: np.ndarray
    destination: np.ndarray
    demand: np.ndarray


@dataclass(frozen=True)
class PathData:
    """Generated path set used by the VI model.

    ``paths`` is a list of paths, each represented as a list of link indices.
    ``block_slices[r]`` gives the coordinates corresponding to OD pair r.
    """

    paths: list[list[int]]
    block_slices: list[slice]
    block_demands: np.ndarray
    block_od_pairs: list[tuple[int, int]]
    path_free_flow_time: np.ndarray


class BPRTrafficProblem:
    """Nonlinear monotone VI for path-flow traffic assignment.

    Parameters
    ----------
    links:
        Directed link attributes.
    od:
        OD demand table used only for metadata.  The feasible set is determined
        by ``path_data.block_demands``.
    path_data:
        Generated paths and block structure.
    path_regularization:
        Optional coefficient added as ``path_regularization * x`` to the
        operator.  Leave at zero for the classical traffic assignment VI.
    lambda_diag:
        Positive diagonal Lambda used by adaptive methods and weighted prox
        steps.  If ``None``, the identity is used.
    """

    def __init__(
        self,
        links: LinkData,
        od: ODData,
        path_data: PathData,
        path_regularization: float = 0.0,
        lambda_diag: np.ndarray | None = None,
    ) -> None:
        self.links = links
        self.od = od
        self.path_data = path_data
        self.path_regularization = float(path_regularization)

        self.num_links = int(len(links.tail))
        self.num_paths = int(len(path_data.paths))
        self.num_blocks = int(len(path_data.block_slices))
        if self.num_paths == 0 or self.num_blocks == 0:
            raise ValueError("The path set is empty. Increase k_paths or check OD connectivity.")

        self.block_slices = path_data.block_slices
        self.block_demands = np.asarray(path_data.block_demands, dtype=float)
        self.block_od_pairs = list(path_data.block_od_pairs)

        self.A = self._build_link_path_incidence(path_data.paths)
        self.block_A = [self.A[:, sl].tocsr() for sl in self.block_slices]

        if lambda_diag is None:
            self.lambda_diag = np.ones(self.num_paths, dtype=float)
        else:
            lambda_diag = np.asarray(lambda_diag, dtype=float)
            if lambda_diag.shape != (self.num_paths,):
                raise ValueError("lambda_diag must have length equal to the number of paths.")
            if np.any(lambda_diag <= 0):
                raise ValueError("lambda_diag entries must be strictly positive.")
            self.lambda_diag = lambda_diag

        # Precompute safe versions of link parameters.  Zero capacities make BPR
        # undefined, so we clip to a tiny positive number and report this through
        # metrics rather than failing late inside the optimizer.
        self.capacity = np.maximum(np.asarray(links.capacity, dtype=float), 1e-12)
        self.t0 = np.maximum(np.asarray(links.free_flow_time, dtype=float), 1e-12)
        self.alpha = np.maximum(np.asarray(links.alpha, dtype=float), 0.0)
        self.power = np.maximum(np.asarray(links.power, dtype=float), 1.0)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_link_path_incidence(self, paths: Sequence[Sequence[int]]) -> sparse.csr_matrix:
        """Build sparse link-path incidence matrix A with A[e, p] = 1."""
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for p_idx, path in enumerate(paths):
            # Repeated links inside a simple path should not occur, but using +=1
            # is mathematically harmless if a user supplies a walk.
            for e_idx in path:
                rows.append(int(e_idx))
                cols.append(p_idx)
                data.append(1.0)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.num_links, self.num_paths))

    # ------------------------------------------------------------------
    # Norms and prox operators
    # ------------------------------------------------------------------
    def primal_norm(self, x: np.ndarray) -> float:
        """Return ||x||_Lambda."""
        return float(np.sqrt(np.dot(self.lambda_diag, np.asarray(x) ** 2)))

    def dual_norm(self, y: np.ndarray) -> float:
        """Return ||y||_{Lambda^{-1}}."""
        return float(np.sqrt(np.dot(np.asarray(y) ** 2, 1.0 / self.lambda_diag)))

    def prox_block(self, center_block: np.ndarray, direction_block: np.ndarray, step: float, block_id: int) -> np.ndarray:
        """Prox step for one OD block."""
        sl = self.block_slices[block_id]
        return prox_scaled_simplex_block(
            center=center_block,
            direction=direction_block,
            step=step,
            radius=float(self.block_demands[block_id]),
            lambda_diag=self.lambda_diag[sl],
        )

    def prox_full(self, center: np.ndarray, direction: np.ndarray, step: float) -> np.ndarray:
        """Block-separable prox over the full product of OD simplexes."""
        out = np.asarray(center, dtype=float).copy()
        direction = np.asarray(direction, dtype=float)
        for b, sl in enumerate(self.block_slices):
            out[sl] = self.prox_block(out[sl], direction[sl], step, b)
        return out

    def project_feasible(self, x: np.ndarray) -> np.ndarray:
        """Project an arbitrary vector onto the product of OD simplexes."""
        zeros = np.zeros_like(x, dtype=float)
        return self.prox_full(center=x, direction=zeros, step=0.0)

    # ------------------------------------------------------------------
    # Traffic operator and metrics
    # ------------------------------------------------------------------
    def link_flows(self, x: np.ndarray) -> np.ndarray:
        """Compute link flows f = A x."""
        return np.asarray(self.A @ np.asarray(x, dtype=float)).ravel()

    def link_costs_from_flows(self, flows: np.ndarray) -> np.ndarray:
        """Compute BPR link travel times from link flows."""
        flows = np.maximum(np.asarray(flows, dtype=float), 0.0)
        ratio = flows / self.capacity
        return self.t0 * (1.0 + self.alpha * np.power(ratio, self.power))

    def operator(self, x: np.ndarray) -> np.ndarray:
        """Return path costs F(x) = A^T t(Ax) + lambda_reg x."""
        x = np.asarray(x, dtype=float)
        flows = self.link_flows(x)
        return self.operator_from_flows(x, flows)

    def operator_from_flows(self, x: np.ndarray, flows: np.ndarray) -> np.ndarray:
        """Return the full path-cost operator when link flows are already known."""
        x = np.asarray(x, dtype=float)
        link_costs = self.link_costs_from_flows(flows)
        path_costs = np.asarray(self.A.T @ link_costs).ravel()
        if self.path_regularization > 0:
            path_costs = path_costs + self.path_regularization * x
        return path_costs

    def block_operator_from_flows(self, flows: np.ndarray, x_block: np.ndarray, block_id: int) -> np.ndarray:
        """Return one block of F(x) when the current link flows Ax are known."""
        link_costs = self.link_costs_from_flows(flows)
        block_costs = np.asarray(self.block_A[block_id].T @ link_costs).ravel()
        if self.path_regularization > 0:
            block_costs = block_costs + self.path_regularization * np.asarray(x_block, dtype=float)
        return block_costs

    def apply_block_delta_to_flows(self, flows: np.ndarray, block_id: int, delta_block: np.ndarray) -> np.ndarray:
        """Update link flows after changing one path-flow block."""
        if np.any(delta_block):
            flows = np.asarray(flows, dtype=float).copy()
            flows += np.asarray(self.block_A[block_id] @ np.asarray(delta_block, dtype=float)).ravel()
        return flows

    def block_operator(self, x: np.ndarray, block_id: int) -> np.ndarray:
        """Return one block of the operator.

        This reference implementation computes the full operator and slices it.
        That keeps the optimizer code simple and reliable.  If you later need
        large-scale speed, this is the best location to replace with an
        incremental flow/cache implementation.
        """
        sl = self.block_slices[block_id]
        return self.operator(x)[sl]

    def delayed_cyclic_operator(self, prefix_source: np.ndarray, suffix_source: np.ndarray) -> np.ndarray:
        """Assemble the delayed cyclic operator used by ADUCA/CODER.

        For block i this returns

            F_i(prefix_source^1, ..., prefix_source^{i-1},
                suffix_source^i, ..., suffix_source^m).

        This exactly matches the partial-update operator in the ADUCA draft and
        the p_k operator in CODER.
        """
        partial = np.asarray(suffix_source, dtype=float).copy()
        flows = self.link_flows(partial)
        out = np.zeros(self.num_paths, dtype=float)
        for b, sl in enumerate(self.block_slices):
            out[sl] = self.block_operator_from_flows(flows, partial[sl], b)
            delta = np.asarray(prefix_source[sl], dtype=float) - partial[sl]
            flows = self.apply_block_delta_to_flows(flows, b, delta)
            partial[sl] = prefix_source[sl]
        return out

    def initial_flow(self, mode: str = "uniform", seed: int | None = None) -> np.ndarray:
        """Construct a feasible starting point.

        ``uniform`` splits each OD demand equally across generated paths.
        ``shortest`` puts all mass on the first generated path for each OD; path
        generation sorts by free-flow time, so the first path is shortest under
        free-flow costs.
        ``random_simplex`` draws one uniform random point from each OD simplex.
        """
        x = np.zeros(self.num_paths, dtype=float)
        rng = np.random.default_rng(seed) if mode == "random_simplex" else None
        for b, sl in enumerate(self.block_slices):
            n = sl.stop - sl.start
            D = float(self.block_demands[b])
            if n <= 0:
                continue
            if mode == "shortest":
                x[sl.start] = D
            elif mode == "uniform":
                x[sl] = D / n
            elif mode == "random_simplex":
                assert rng is not None
                x[sl] = D * rng.dirichlet(np.ones(n, dtype=float))
            else:
                raise ValueError(f"Unknown init mode: {mode}")
        return x

    def beckmann_potential(self, x: np.ndarray) -> float:
        """Return the Beckmann potential for separable BPR costs.

        The user-equilibrium link-flow solution minimizes this potential over
        feasible path flows when path_regularization is zero.  For nonzero path
        regularization we add 0.5 * reg * ||x||^2.
        """
        f = self.link_flows(x)
        # Integral_0^f t0 * [1 + alpha * (s/c)^power] ds
        integral = self.t0 * (
            f + self.alpha * np.power(f, self.power + 1.0) / ((self.power + 1.0) * np.power(self.capacity, self.power))
        )
        val = float(np.sum(integral))
        if self.path_regularization > 0:
            val += 0.5 * self.path_regularization * float(np.dot(x, x))
        return val

    def wardrop_gap(self, x: np.ndarray, costs: np.ndarray | None = None) -> tuple[float, float, float]:
        """Compute the path-set Wardrop VI gap and related quantities.

        For each OD block r,

            gap_r = <c_r, x_r> - D_r * min_p c_{r,p}.

        The returned ``relative_gap`` is gap / total_travel_cost.  It is the
        standard traffic-assignment relative gap restricted to the generated path
        set.
        """
        x = np.asarray(x, dtype=float)
        if costs is None:
            costs = self.operator(x)
        total_gap = 0.0
        total_cost = 0.0
        max_excess_used = 0.0
        for b, sl in enumerate(self.block_slices):
            xb = x[sl]
            cb = costs[sl]
            D = float(self.block_demands[b])
            if xb.size == 0 or D <= 0:
                continue
            min_cost = float(np.min(cb))
            total_gap += float(np.dot(cb, xb) - D * min_cost)
            total_cost += float(np.dot(cb, xb))
            used = xb > max(1e-10, 1e-8 * D)
            if np.any(used):
                max_excess_used = max(max_excess_used, float(np.max(cb[used] - min_cost)))
        rel = total_gap / max(abs(total_cost), 1e-12)
        return total_gap, rel, max_excess_used

    def feasibility_error(self, x: np.ndarray) -> tuple[float, float]:
        """Return (max mass error, most negative coordinate magnitude)."""
        max_mass_error = 0.0
        min_coord = float(np.min(x)) if x.size else 0.0
        for b, sl in enumerate(self.block_slices):
            max_mass_error = max(max_mass_error, abs(float(np.sum(x[sl])) - float(self.block_demands[b])))
        return max_mass_error, max(0.0, -min_coord)

    def evaluate(self, x: np.ndarray) -> dict[str, float]:
        """Compute all scalar metrics logged by experiments."""
        x = np.asarray(x, dtype=float)
        costs = self.operator(x)
        flows = self.link_flows(x)
        gap, rel_gap, max_excess = self.wardrop_gap(x, costs)
        mass_err, nonneg_err = self.feasibility_error(x)
        return {
            "wardrop_gap": float(gap),
            "relative_gap": float(rel_gap),
            "max_used_path_excess_cost": float(max_excess),
            "beckmann_potential": float(self.beckmann_potential(x)),
            "total_travel_cost": float(np.dot(costs, x)),
            "max_link_flow": float(np.max(flows)) if flows.size else 0.0,
            "mean_link_flow": float(np.mean(flows)) if flows.size else 0.0,
            "max_path_cost": float(np.max(costs)) if costs.size else 0.0,
            "min_path_cost": float(np.min(costs)) if costs.size else 0.0,
            "feasibility_mass_error": float(mass_err),
            "feasibility_nonneg_error": float(nonneg_err),
        }

    def summary(self) -> dict[str, float | int]:
        """Return problem dimensions and basic scaling information."""
        return {
            "num_links": self.num_links,
            "num_paths": self.num_paths,
            "num_blocks": self.num_blocks,
            "total_demand": float(np.sum(self.block_demands)),
            "min_block_demand": float(np.min(self.block_demands)),
            "max_block_demand": float(np.max(self.block_demands)),
            "path_regularization": float(self.path_regularization),
            "lambda_min": float(np.min(self.lambda_diag)),
            "lambda_max": float(np.max(self.lambda_diag)),
        }
