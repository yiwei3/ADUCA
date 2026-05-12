r"""ADUCA (torch) for the linearly constrained Huber (LC-Huber) saddle operator.

Problem
-------
We solve the saddle-point / VI formulation associated with

    minimize_u h_δ(u)  s.t.  A u = b,

via the Lagrangian L(u,v) = h_δ(u) + <Au - b, v> and the monotone operator

    F(u,v) = (∇h_δ(u) + A^T v,  b - A u).

This file implements the ADUCA cyclic block method (torch/GPU-capable) using
the same step-size constants as the repository's existing ADUCA implementations.

Notes
-----
* Here g ≡ 0, so the proximal map is the identity, and the block updates reduce
  to (diagonally) preconditioned forward steps.
* Blocks are constructed so that they never cross the u/v boundary, which
  allows inexpensive block-slice operator evaluation.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Tuple

import torch

from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.algorithms.utils.helper import construct_uv_block_slices, compute_opt_measure
from src.algorithms.utils.preconditioner import normalizers_torch, recip_normalizers_torch
from src.algorithms.utils.results import Results, logresult
from src.problems.GMVI_func import GMVIProblem
from src.problems.lc_huber_torch_oracle import LCHuberTorchOracle


def aduca_torch(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    """Run ADUCA on an LC-Huber instance.

    Parameters
    ----------
    problem:
        GMVIProblem wrapping an LCHuberOprFunc and (typically) a ZeroGFunc.
    exit_criterion:
        Stopping criteria.
    parameters:
        Required keys: beta, gamma, rho.
        Optional keys:
            - block_size, block_size_u, block_size_v
            - device (e.g. 'cuda:0'), dtype ('float32' or 'float64')
            - preconditioner ('identity' or 'diag_lipschitz')
    u_0:
        Optional initial x0 (numpy or torch) of length n_u + n_v.

    Returns
    -------
    (results, x_final_numpy)
    """

    # ------------------------------------------------------------------
    # Torch config
    # ------------------------------------------------------------------
    device_str = parameters.get("device", "cpu")
    device = torch.device(str(device_str))

    dtype_str = str(parameters.get("dtype", "float64"))
    if dtype_str == "float32":
        dtype = torch.float32
    elif dtype_str == "float64":
        dtype = torch.float64
    else:
        raise ValueError("parameters['dtype'] must be one of {'float32','float64'}")

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])

    if not (beta > 0.0 and rho > 0.0 and 0.0 < gamma < 1.0):
        raise ValueError("Require beta>0, rho>0, and gamma in (0,1).")

    # ------------------------------------------------------------------
    # Constants from the ADUCA paper (same as the other folders)
    # ------------------------------------------------------------------
    rho_0 = min(rho, beta * (1.0 + beta) * (1.0 - gamma))
    eta = math.sqrt((gamma * (1.0 + beta)) / (1.0 + beta**2))
    tau = (3.0 * rho_0**2 * (1.0 + rho * beta)) / (
        2.0 * (rho * beta) ** 2 + 3.0 * rho_0**2 * (1.0 + rho * beta)
    )
    C = (
        eta
        / (2.0 * math.sqrt(beta))
        * (math.sqrt(tau) * rho * beta)
        / (math.sqrt(3.0) * math.sqrt(1.0 + rho * beta))
    )
    C_hat = eta / (2.0 * math.sqrt(beta)) * math.sqrt((1.0 - tau) * rho * beta) / math.sqrt(2.0)

    logging.info("rho_0 = %s", rho_0)
    logging.info("C = %s", C)
    logging.info("C_hat = %s", C_hat)

    # ------------------------------------------------------------------
    # Problem structure
    # ------------------------------------------------------------------
    operator = problem.operator_func
    g = problem.g_func
    n_u = int(operator.n_u)
    n_v = int(operator.n_v)
    n = int(operator.n)

    # Block partition: keep u and v blocks separate.
    block_size_u = int(parameters.get("block_size_u", parameters.get("block_size", 1)))
    block_size_v = int(parameters.get("block_size_v", parameters.get("block_size", 1)))
    blocks, block_types = construct_uv_block_slices(n_u, n_v, block_size_u, block_size_v)
    m = len(blocks)
    logging.info("m = %s blocks (u blocks=%s, v blocks=%s)", m, block_types.count("u"), block_types.count("v"))

    oracle = LCHuberTorchOracle(operator.A, operator.b, delta=operator.delta, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Diagonal preconditioner Λ^{-1} (called "normalizers" in this repo)
    # ------------------------------------------------------------------
    prec_mode = str(parameters.get("preconditioner", "diag_lipschitz"))
    normalizers = normalizers_torch(oracle.A, device=device, dtype=dtype, mode=prec_mode)
    normalizers_recip = recip_normalizers_torch(normalizers)

    use_weighted_prox = (prec_mode != 'identity')

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    # Requested change: default initial point is the all-ones vector.
    if u_0 is None:
        x0 = torch.ones(n, dtype=dtype, device=device)
    else:
        x0 = torch.tensor(u_0, dtype=dtype, device=device)
        if x0.ndim != 1 or x0.numel() != n:
            raise ValueError(f"u_0 must be a 1D tensor/array of length {n}")

    def inner_weighted_sq(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.dot(x, w * x)

    # State and operator at x0
    Au0, ATv0, r2_0 = oracle.compute_state(x0)
    F0 = oracle.func_map_with_state(x0, Au0, ATv0, r2_0)

    results = Results()
    opt_kind = str(parameters.get("opt_measure", "prox_residual"))

    opt_measure = compute_opt_measure(
        opt_kind,
        x=x0,
        F_x=F0,
        g=g,
        oracle=oracle,
        Au=Au0,
        r2=r2_0,
    )
    logresult(results, 0, 0.0, opt_measure)

    # ------------------------------------------------------------------
    # Local backtracking initialization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_F1_and_Ftilde1(x0_: torch.Tensor, x1_: torch.Tensor):
        x_tmp = x0_.clone()
        Au_tmp = Au0.clone()
        ATv_tmp = ATv0.clone()
        r2_tmp = r2_0.clone()

        F_tilde1_ = torch.empty_like(x0_)

        for sl, typ in zip(blocks, block_types):
            F_tilde1_[sl] = oracle.func_map_slice_with_state(x_tmp, Au_tmp, ATv_tmp, r2_tmp, sl)

            old_block = x_tmp[sl]
            new_block = x1_[sl]

            if typ == "u":
                dq = new_block - old_block
                r2_tmp = r2_tmp + torch.dot(new_block, new_block) - torch.dot(old_block, old_block)
                Au_tmp.add_(oracle.A[:, sl] @ dq)
            else:
                dv = new_block - old_block
                i0 = int(sl.start) - n_u
                i1 = int(sl.stop) - n_u
                ATv_tmp.add_(oracle.A[i0:i1, :].T @ dv)

            x_tmp[sl] = new_block

        F1_ = oracle.func_map_with_state(x1_, Au_tmp, ATv_tmp, r2_tmp)
        return F1_, F_tilde1_, Au_tmp, ATv_tmp, r2_tmp

    @torch.no_grad()
    def weighted_L_from_points(xa: torch.Tensor, xb: torch.Tensor, Fa: torch.Tensor, Fb: torch.Tensor) -> float:
        dx = xb - xa
        denom = float(inner_weighted_sq(dx, normalizers_recip).item())
        if denom <= 1e-24:
            return 0.0
        dF = Fb - Fa
        num = float(inner_weighted_sq(dF, normalizers).item())
        return math.sqrt(max(num, 0.0) / denom)

    with torch.no_grad():
        a_max = float(parameters.get("a_max", parameters.get("a0", 1.0)))
        if not math.isfinite(a_max) or a_max <= 0.0:
            a_max = 1.0
        i_bt = 0
        while True:
            a0 = a_max * (0.5 ** i_bt)
            x1_bt_forward = x0 - a0 * normalizers * F0
            x1_bt = g.prox_opr_torch(x1_bt_forward, tau=a0, weights=normalizers_recip if use_weighted_prox else None)

            Au_bt, ATv_bt, r2_bt = oracle.compute_state(x1_bt)
            F1_bt = oracle.func_map_with_state(x1_bt, Au_bt, ATv_bt, r2_bt)
            L1_bt = weighted_L_from_points(x0, x1_bt, F0, F1_bt)

            if L1_bt == 0.0 or a0 * L1_bt <= 1.0:
                break
            i_bt += 1
            if a0 <= 1e-300:
                break

        x1 = x1_bt
        F1, F_tilde1, Au1, ATv1, r2_1 = compute_F1_and_Ftilde1(x0, x1)

        logging.info(
            "Local backtracking init: a_max=%.4e, L1=%.4e, backtracks=%d, a0=%.4e",
            a_max,
            L1_bt,
            i_bt,
            a0,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    a_prev = a0
    a = a0

    x = x1.clone()
    x_prev = x0.clone()

    v_prev = x0.clone()
    v = torch.empty_like(x)

    F_prev = F0
    F = F1

    F_tilde_prev = F0.clone()
    F_tilde = F_tilde1.clone()

    Au = Au1.clone()
    ATv = ATv1.clone()
    r2 = r2_1.clone()

    k = 0
    exit_flag = False
    start_time = time.time()

    def aduca_stepsize(
        x_cur: torch.Tensor,
        x_old: torch.Tensor,
        a_cur: float,
        a_old: float,
        F_cur: torch.Tensor,
        F_old: torch.Tensor,
        F_tilde_cur: torch.Tensor,
    ) -> Tuple[float, float, float]:
        step_1 = rho_0 * a_cur

        dx_ = x_cur - x_old
        denom = float(inner_weighted_sq(dx_, normalizers_recip).item())
        denom = max(denom, 1e-24)

        dF_ = F_cur - F_old
        num = float(inner_weighted_sq(dF_, normalizers).item())
        L_k = math.sqrt(max(num, 0.0) / denom)
        step_2 = math.inf if L_k == 0.0 else (C / L_k) * math.sqrt(a_cur / a_old)

        dF_hat = F_cur - F_tilde_cur
        num_hat = float(inner_weighted_sq(dF_hat, normalizers).item())
        L_hat_k = math.sqrt(max(num_hat, 0.0) / denom)
        step_3 = math.inf if L_hat_k == 0.0 else (C_hat / L_hat_k) * math.sqrt(a_cur / a_old)

        return min(step_1, step_2, step_3), L_k, L_hat_k

    with torch.no_grad():
        while not exit_flag:
            step, L_k, L_hat_k = aduca_stepsize(x, x_prev, a, a_prev, F, F_prev, F_tilde)
            a_prev, a = a, step  # now a_prev = a_k, a = a_{k+1}

            for sl, typ in zip(blocks, block_types):
                # Save old block for state update and for constructing x_prev = x_k for next epoch.
                x_prev[sl] = x[sl]

                # Extrapolation point
                v_block = (1.0 - beta) * x[sl] + beta * v_prev[sl]
                v[sl] = v_block

                # Current-cycle delayed operator block: \tilde F_{k+1}^i
                F_partial_block = oracle.func_map_slice_with_state(x, Au, ATv, r2, sl)

                # Previous-cycle delayed operator block: \tilde F_k^i
                old_tilde_block = F_tilde[sl]

                # -------------------- FIXED BUG HERE --------------------
                # Correct ADUCA bar operator:
                #   \bar F_{k+1}^i = \tilde F_{k+1}^i + (a_k/a_{k+1}) (F_k^i - \tilde F_k^i)
                # Here: F is F_k, a_prev is a_k, a is a_{k+1}.
                F_bar_block = F_partial_block + (a_prev / a) * (F[sl] - old_tilde_block)
                # --------------------------------------------------------

                # Shift delayed operator history
                F_tilde_prev[sl] = old_tilde_block
                F_tilde[sl] = F_partial_block

                # Prox step (g ≡ 0 → identity)
                x_forward = v_block - a * normalizers[sl] * F_bar_block
                x_new = g.prox_block_torch(
                    x_forward,
                    block_type=typ,
                    tau=a,
                    weights_block=normalizers_recip[sl] if use_weighted_prox else None,
                )
                x[sl] = x_new

                # Update cached state in-place
                r2 = oracle.update_state_after_block_update_(x, Au, ATv, r2, sl, x_prev[sl], x_new)

            F_new = oracle.func_map_with_state(x, Au, ATv, r2)

            F_prev, F = F, F_new
            v_prev, v = v, v_prev

            k += 1

            if k % exit_criterion.loggingfreq == 0:
                elapsed = time.time() - start_time
                opt_measure = compute_opt_measure(
                    opt_kind,
                    x=x,
                    F_x=F,
                    g=g,
                    oracle=oracle,
                    Au=Au,
                    r2=r2,
                )
                logging.info(
                    "elapsed_time: %s, iteration: %s, opt_measure: %s, L: %s, L_hat: %s",
                    elapsed,
                    k,
                    opt_measure,
                    L_k,
                    L_hat_k,
                )
                logresult(results, k, elapsed, opt_measure, L=L_k, L_hat=L_hat_k)
                exit_flag = CheckExitCondition(exit_criterion, k, elapsed, opt_measure)

    return results, x.detach().cpu().numpy()
