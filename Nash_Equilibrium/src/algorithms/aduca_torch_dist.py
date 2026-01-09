import os
import math
import time
import logging
import socket
from typing import Optional, Tuple

import numpy as np

from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range
from src.algorithms.aduca import aduca as _aduca_numpy_reference


def _find_free_port(default: int = 29500) -> int:
    """Pick a free TCP port for single-process dist runs."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1] or default


def _as_int_blocksize(value, default: int) -> int:
    """
    Robustly coerce a block size that may come from argparse (possibly inf / None)
    into a sane positive integer.
    """
    if value is None:
        return int(default)
    try:
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            return int(default)
        v = int(value)
        return v if v > 0 else int(default)
    except Exception:
        return int(default)


def _split_range(n: int, world_size: int, rank: int) -> Tuple[int, int]:
    """
    Contiguous partition of [0, n) across ranks.
    """
    n = int(n)
    world_size = int(world_size)
    rank = int(rank)
    base = n // world_size
    rem = n % world_size
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end


def _aduca_torch_distributed_nash(problem: GMVIProblem,
                                 exit_criterion: ExitCriterion,
                                 parameters,
                                 u_0: Optional[np.ndarray] = None):
    """
    Multi-process ADUCA implementation for the Nash Equilibrium problem.

    This is a distributed GPU extension of aduca.py:
    - The update order and block logic match the NumPy implementation.
    - q (dimension n) is sharded across ranks.
    - Global scalars (Q, norms) are synchronized with all_reduce.

    This implementation avoids per-block full-vector updates by keeping df(q)
    and global scalars (p, dp) up to date, and materializing F(q) only when
    needed (once per iteration).
    """
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        raise RuntimeError(
            "PyTorch (with torch.distributed) is required for parameters['backend'] == 'torch_dist'."
        ) from exc

    # ----------------------------
    # Distributed init
    # ----------------------------
    if not dist.is_initialized():
        dist_backend = parameters.get("dist_backend", "nccl")
        if dist_backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("Requested NCCL backend but torch.cuda.is_available() is False.")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(_find_free_port())
        dist.init_process_group(backend=dist_backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    # ----------------------------
    # Problem constants
    # ----------------------------
    n = int(problem.operator_func.n)

    beta_alg = float(parameters["beta"])
    gamma_alg = float(parameters["gamma"])
    rho = float(parameters["rho"])
    mu = float(parameters.get("mu", 0.0))
    strong_convexity = bool(parameters.get("strong_convexity", parameters.get("strong-convexity", False)))

    op_gamma = float(problem.operator_func.gamma)
    min_total_quantity = float(getattr(problem.operator_func, "_min_total_quantity", 1e-12))

    rho_0 = min(rho, beta_alg * (1 + beta_alg) * (1 - gamma_alg))
    eta = ((gamma_alg * (1 + beta_alg)) / (1 + beta_alg ** 2)) ** 0.5
    tau = (3 * rho_0 ** 2 * (1 + rho * beta_alg) /
           (2 * (rho * beta_alg) ** 2 + 3 * rho_0 ** 2 * (1 + rho * beta_alg)))
    C = eta / (2 * beta_alg ** 0.5) * (tau ** 0.5 * rho * beta_alg) / (3 ** 0.5 * (1 + rho * beta_alg) ** 0.5)
    C_hat = eta / (2 * beta_alg ** 0.5) * ((1 - tau) * rho * beta_alg) ** 0.5 / 2 ** 0.5

    if rank == 0:
        logging.info(f"[torch_dist] world_size = {world_size}, device = {device}")
        logging.info(f"rho = {rho_0}")
        logging.info(f"C = {C}")
        logging.info(f"C_hat = {C_hat}")

    block_size = _as_int_blocksize(parameters.get("block_size", 1), default=1)
    blocks = construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    if rank == 0:
        logging.info(f"m = {m}")

    # ----------------------------
    # Shard data by coordinates
    # ----------------------------
    start_idx, end_idx = _split_range(n, world_size, rank)
    n_local = end_idx - start_idx
    if n_local <= 0:
        raise RuntimeError(f"Rank {rank} got empty shard. Check n={n}, world_size={world_size}.")

    c_full = np.asarray(problem.operator_func.c)
    L_full = np.asarray(problem.operator_func.L)
    beta_op_full = np.asarray(problem.operator_func.beta)

    if c_full.ndim == 0:
        c_full = np.full(n, float(c_full))
    if L_full.ndim == 0:
        L_full = np.full(n, float(L_full))
    if beta_op_full.ndim == 0:
        beta_op_full = np.full(n, float(beta_op_full))

    dtype_param = str(parameters.get("dtype", "float32")).lower()
    if dtype_param in ("float64", "fp64", "double"):
        vec_dtype = torch.float64
    else:
        vec_dtype = torch.float32

    reduce_dtype = vec_dtype
    reduce_dtype_param = str(parameters.get("reduce_dtype", "")).lower()
    if reduce_dtype_param in ("float32", "fp32"):
        reduce_dtype = torch.float32
    elif reduce_dtype_param in ("float64", "fp64", "double"):
        reduce_dtype = torch.float64

    c_local = torch.tensor(c_full[start_idx:end_idx], device=device, dtype=vec_dtype)
    L_local = torch.tensor(L_full[start_idx:end_idx], device=device, dtype=vec_dtype)
    beta_op_local = torch.tensor(beta_op_full[start_idx:end_idx], device=device, dtype=vec_dtype)

    inv_beta_local = 1.0 / beta_op_local
    L_pow_local = torch.pow(L_local, inv_beta_local)
    normalizer = torch.pow(1.0 / L_local, inv_beta_local) * (1.0 / beta_alg)
    normalizer_recip = torch.where(normalizer != 0, 1.0 / normalizer, torch.zeros_like(normalizer))

    p_const = float(5000.0 ** (1.0 / op_gamma))
    dp_const = - (1.0 / op_gamma) * p_const
    p_power = -1.0 / op_gamma
    dp_power = -1.0 / op_gamma - 1.0
    p_const_t = torch.tensor(p_const, device=device, dtype=vec_dtype)
    dp_const_t = torch.tensor(dp_const, device=device, dtype=vec_dtype)
    min_total_quantity_t = torch.tensor(min_total_quantity, device=device, dtype=vec_dtype)

    local_block_slices = []
    for block in blocks:
        local_start = max(block.start, start_idx)
        local_stop = min(block.stop, end_idx)
        if local_start >= local_stop:
            local_block_slices.append(None)
        else:
            local_block_slices.append(slice(local_start - start_idx, local_stop - start_idx))

    reduce_buf3 = torch.empty(3, device=device, dtype=reduce_dtype)
    residual_buf = torch.empty(n_local, device=device, dtype=vec_dtype)
    F_diff = torch.empty(n_local, device=device, dtype=vec_dtype)
    F_tilde_diff = torch.empty(n_local, device=device, dtype=vec_dtype)
    u_diff = torch.empty(n_local, device=device, dtype=vec_dtype)
    q_clamped = torch.empty(n_local, device=device, dtype=vec_dtype)

    delta_q_t = torch.zeros(1, device=device, dtype=reduce_dtype)

    def compute_Q(u_local_vec):
        q_sum = torch.sum(u_local_vec, dtype=reduce_dtype)
        if world_size > 1:
            dist.all_reduce(q_sum, op=dist.ReduceOp.SUM)
        return q_sum

    def compute_p_dp(Q_reduce):
        Q_vec = Q_reduce.to(dtype=vec_dtype)
        Q_safe = torch.maximum(Q_vec, min_total_quantity_t)
        Q_pow = torch.pow(Q_safe, p_power)
        p_val = p_const_t * Q_pow
        dp_val = dp_const_t * Q_pow / Q_safe
        return p_val, dp_val

    def update_df_full(q_local_vec, df_out):
        torch.clamp(q_local_vec, min=0.0, out=q_clamped)
        torch.pow(q_clamped, inv_beta_local, out=df_out)
        df_out.mul_(L_pow_local).add_(c_local)

    def update_df_block(q_local_vec, df_out, block_slice):
        q_block = q_local_vec[block_slice]
        q_block_clamped = q_clamped[block_slice]
        torch.clamp(q_block, min=0.0, out=q_block_clamped)
        df_block = df_out[block_slice]
        torch.pow(q_block_clamped, inv_beta_local[block_slice], out=df_block)
        df_block.mul_(L_pow_local[block_slice]).add_(c_local[block_slice])

    def write_full_F(out, df_vec, q_vec, p_val, dp_val):
        out.copy_(df_vec)
        out.add_(q_vec, alpha=-dp_val)
        out.add_(-p_val)

    def compute_weighted_norms(F_diff_vec, F_tilde_diff_vec, u_diff_vec):
        reduce_buf3[0] = torch.sum(normalizer * (F_diff_vec * F_diff_vec), dtype=reduce_dtype)
        reduce_buf3[1] = torch.sum(normalizer * (F_tilde_diff_vec * F_tilde_diff_vec), dtype=reduce_dtype)
        reduce_buf3[2] = torch.sum(normalizer_recip * (u_diff_vec * u_diff_vec), dtype=reduce_dtype)
        if world_size > 1:
            dist.all_reduce(reduce_buf3, op=dist.ReduceOp.SUM)
        norm_F = torch.sqrt(reduce_buf3[0]).item()
        norm_F_tilde = torch.sqrt(reduce_buf3[1]).item()
        norm_u = torch.sqrt(reduce_buf3[2]).item()
        return norm_F, norm_F_tilde, norm_u

    def compute_L_values(u_vec, u_prev_vec, F_vec, F_prev_vec, F_tilde_vec):
        u_diff.copy_(u_vec).sub_(u_prev_vec)
        F_diff.copy_(F_vec).sub_(F_prev_vec)
        F_tilde_diff.copy_(F_vec).sub_(F_tilde_vec)
        reduce_buf3[0] = torch.sum(normalizer * (F_diff * F_diff), dtype=reduce_dtype)
        reduce_buf3[1] = torch.sum(normalizer_recip * (u_diff * u_diff), dtype=reduce_dtype)
        reduce_buf3[2] = torch.sum(normalizer * (F_tilde_diff * F_tilde_diff), dtype=reduce_dtype)
        if world_size > 1:
            dist.all_reduce(reduce_buf3, op=dist.ReduceOp.SUM)
        L_val = torch.sqrt(reduce_buf3[0] / reduce_buf3[1]).item()
        L_hat_val = torch.sqrt(reduce_buf3[2] / reduce_buf3[1]).item()
        return L_val, L_hat_val

    def compute_residual_from_F(q_local_vec, F_local_vec):
        torch.minimum(q_local_vec, F_local_vec, out=residual_buf)
        norm_sq = torch.sum(residual_buf * residual_buf, dtype=reduce_dtype)
        if world_size > 1:
            dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM)
        return torch.sqrt(norm_sq).item()

    # ----------------------------
    # Initialize u0, df(u0), F0
    # ----------------------------
    if u_0 is None:
        u0 = torch.ones(n_local, device=device, dtype=vec_dtype)
    else:
        u_0 = np.asarray(u_0, dtype=np.float64)
        if u_0.shape[0] != n:
            raise ValueError(f"u_0 must have shape ({n},), got {u_0.shape}.")
        u0 = torch.tensor(u_0[start_idx:end_idx], device=device, dtype=vec_dtype)

    df_local = torch.empty_like(u0)
    update_df_full(u0, df_local)

    Q0 = compute_Q(u0)
    p0, dp0 = compute_p_dp(Q0)

    F_prev = torch.empty_like(u0)
    write_full_F(F_prev, df_local, u0, p0, dp0)
    F_tilde_prev = F_prev.clone()

    # Results (rank 0 only)
    results = Results() if rank == 0 else None
    start_time = time.time()

    init_measure = compute_residual_from_F(u0, F_prev)
    if rank == 0:
        logresult(results, 1, 0.0, init_measure)

    # Buffers for line search and main loop
    u = torch.empty_like(u0)
    u1 = torch.empty_like(u0)
    F_curr = torch.empty_like(u0)
    F_tilde = torch.empty_like(u0)
    F_bar = torch.empty_like(u0)
    F_store_block = torch.empty_like(u0)

    def run_line_search(a0):
        u.copy_(u0)
        update_df_full(u0, df_local)
        F_tilde.copy_(F_prev)

        Q = Q0.clone()
        p_val = p0
        dp_val = dp0

        u1.copy_(F_prev).mul_(normalizer).mul_(a0).neg_().add_(u0).clamp_(min=0.0)

        for block_slice in local_block_slices:
            if block_slice is not None:
                F_block = F_store_block[block_slice]
                F_block.copy_(df_local[block_slice])
                F_block.add_(u[block_slice], alpha=-dp_val)
                F_block.add_(-p_val)

                F_tilde[block_slice].copy_(F_block)

                delta_q_t[0] = torch.sum(u1[block_slice] - u[block_slice], dtype=reduce_dtype)
                u[block_slice].copy_(u1[block_slice])
            else:
                delta_q_t[0] = 0.0

            if world_size > 1:
                dist.all_reduce(delta_q_t, op=dist.ReduceOp.SUM)
            Q = Q + delta_q_t[0]

            p_val, dp_val = compute_p_dp(Q)

            if block_slice is not None:
                update_df_block(u, df_local, block_slice)

        write_full_F(F_curr, df_local, u, p_val, dp_val)

        F_diff.copy_(F_curr).sub_(F_prev)
        F_tilde_diff.copy_(F_curr).sub_(F_tilde)
        u_diff.copy_(u1).sub_(u0)
        norm_F, norm_F_tilde, norm_u = compute_weighted_norms(F_diff, F_tilde_diff, u_diff)
        return norm_F, norm_F_tilde, norm_u, Q, p_val, dp_val

    # ----------------------------
    # Line-search for the first step size a0 (full-block update)
    # ----------------------------
    alpha_ls = math.sqrt(2.0)
    i_ls = -1
    a0 = 1.0

    while True:
        i_ls += 1
        a0 = alpha_ls ** (-i_ls)
        norm_F, norm_F_tilde, norm_u, Q, p_val, dp_val = run_line_search(a0)
        if (a0 * norm_F_tilde <= C_hat * norm_u) and (a0 * norm_F <= C * norm_u):
            break

    while True:
        if (a0 * norm_F_tilde >= (C_hat / alpha_ls) * norm_u) or (a0 * norm_F >= (C / alpha_ls) * norm_u):
            break
        a0 = a0 * alpha_ls
        norm_F, norm_F_tilde, norm_u, Q, p_val, dp_val = run_line_search(a0)

    # Initialize states at k=1 (matching the NumPy code after line-search)
    a_prev = float(a0)
    a_curr = float(a0)
    A_accum = 0.0

    u_prev = u0.clone()
    v_prev = u_prev.clone()
    u_hat = torch.zeros_like(u0)

    F_tilde_prev = F_prev.clone()
    # F_curr and F_tilde are already set by run_line_search

    # Iteration counters
    k = 0
    exit_flag = False
    exit_t = torch.zeros(1, device=device, dtype=torch.int32)

    # ----------------------------
    # Main loop
    # ----------------------------
    while not exit_flag:
        L, L_hat = compute_L_values(u, u_prev, F_curr, F_prev, F_tilde)

        step_1 = rho_0 * a_curr
        if L == 0:
            step_2 = float("inf")
        else:
            step_2 = C / L * math.sqrt(a_curr / a_prev)
        if L_hat == 0:
            step_3 = float("inf")
        else:
            step_3 = C_hat / L_hat * math.sqrt(a_curr / a_prev)

        step = min(step_1, step_2, step_3)

        a_prev = a_curr
        a_curr = float(step)
        A_accum += a_curr

        if strong_convexity:
            omega_k = (1.0 + rho * beta_alg * mu * a_curr) / (1.0 + mu * a_curr)
            ratio_bar = (a_prev * omega_k) / a_curr
        else:
            ratio_bar = a_prev / a_curr

        for block_slice in local_block_slices:
            if block_slice is not None:
                F_block = F_store_block[block_slice]
                F_block.copy_(df_local[block_slice])
                F_block.add_(u[block_slice], alpha=-dp_val)
                F_block.add_(-p_val)

                F_bar_block = F_bar[block_slice]
                F_bar_block.copy_(F_tilde[block_slice])
                F_bar_block.add_(F_prev[block_slice] - F_tilde_prev[block_slice], alpha=ratio_bar)

                v_block = u[block_slice] * (1.0 - beta_alg) + v_prev[block_slice] * beta_alg

                u_prev[block_slice].copy_(u[block_slice])
                u[block_slice].copy_(v_block)
                u[block_slice].add_(F_bar_block * normalizer[block_slice], alpha=-a_curr)
                u[block_slice].clamp_(min=0.0)

                v_prev[block_slice].copy_(v_block)

                F_tilde_prev[block_slice].copy_(F_tilde[block_slice])
                F_tilde[block_slice].copy_(F_block)

                delta_q_t[0] = torch.sum(u[block_slice] - u_prev[block_slice], dtype=reduce_dtype)
            else:
                delta_q_t[0] = 0.0

            if world_size > 1:
                dist.all_reduce(delta_q_t, op=dist.ReduceOp.SUM)
            Q = Q + delta_q_t[0]

            p_val, dp_val = compute_p_dp(Q)

            if block_slice is not None:
                update_df_block(u, df_local, block_slice)

        F_prev.copy_(F_curr)
        write_full_F(F_curr, df_local, u, p_val, dp_val)

        if A_accum != 0.0:
            u_hat.mul_((A_accum - a_curr) / A_accum).add_(u_prev, alpha=a_curr / A_accum)

        k += m

        if k % (m * exit_criterion.loggingfreq) == 0:
            elapsed = time.time() - start_time
            measure = compute_residual_from_F(u, F_curr)

            if rank == 0:
                logging.info(f"[torch_dist] elapsed_time: {elapsed:.4f}, iteration: {k}, opt_measure: {measure}")
                logresult(results, k, elapsed, measure, L=L, L_hat=L_hat)

            exit_t.zero_()
            if rank == 0:
                exit_t[0] = 1 if CheckExitCondition(exit_criterion, k, elapsed, measure) else 0
            if world_size > 1:
                dist.broadcast(exit_t, src=0)
            exit_flag = bool(int(exit_t.item()))

    # ----------------------------
    # Gather full solution u to rank 0
    # ----------------------------
    if world_size == 1:
        if rank == 0:
            return results, u.cpu().numpy()
        return Results(), np.zeros((n,), dtype=np.float32)

    sizes = torch.tensor([n_local], device=device, dtype=torch.int64)
    sizes_list = [torch.empty_like(sizes) for _ in range(world_size)]
    dist.all_gather(sizes_list, sizes)
    sizes_cpu = [int(s.item()) for s in sizes_list]
    max_n_local = max(sizes_cpu)

    u_pad = torch.zeros(max_n_local, device=device, dtype=vec_dtype)
    u_pad[:n_local] = u
    u_gather = [torch.empty_like(u_pad) for _ in range(world_size)]
    dist.all_gather(u_gather, u_pad)

    if rank == 0:
        u_full = torch.cat([u_gather[r][:sizes_cpu[r]].cpu() for r in range(world_size)], dim=0).numpy()
        return results, u_full

    return Results(), np.zeros((n,), dtype=np.float32)


def aduca_distributed(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    """
    Unified entry point.

    - Default: original NumPy implementation (single-process).
    - Set parameters["backend"] = "torch_dist" (or parameters["torch_distributed"]=True) to enable
      PyTorch Distributed (multi-process).
    """
    backend = str(parameters.get("backend", "numpy")).lower()
    if parameters.get("torch_distributed", True) or backend in ("torch_dist", "torch_distributed", "ddp", "pytorch_dist"):
        logging.info("Using PyTorch Distributed backend for ADUCA Nash Equilibrium.")
        return _aduca_torch_distributed_nash(problem, exit_criterion, parameters, u_0=u_0)
    return _aduca_numpy_reference(problem, exit_criterion, parameters, u_0=u_0)
