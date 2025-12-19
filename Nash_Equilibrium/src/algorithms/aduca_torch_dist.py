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

    Design:
    - q (dimension n) is sharded across ranks.
    - The operator depends on the global quantity Q = sum(q), computed via all_reduce.
    - Updates are performed in parallel across all coordinates (Jacobi-style).
    - Iteration accounting uses block_size to match the sequential code's k += m.
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
        # Provide sane defaults for single-process runs so env:// works without explicit env vars.
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

    # Reduce log spam from non-zero ranks
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    # ----------------------------
    # Problem constants
    # ----------------------------
    n = int(problem.operator_func.n)

    beta_alg = float(parameters["beta"])
    gamma_alg = float(parameters["gamma"])
    rho = float(parameters["rho"])
    eps = float(parameters.get("eps", 1e-12))

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
    m = (n + block_size - 1) // block_size

    # ----------------------------
    # Shard data by coordinates
    # ----------------------------
    start_idx, end_idx = _split_range(n, world_size, rank)
    n_local = end_idx - start_idx
    if n_local <= 0:
        raise RuntimeError(f"Rank {rank} got empty shard. Check n={n}, world_size={world_size}.")

    c_full = np.asarray(problem.operator_func.c, dtype=np.float32)
    L_full = np.asarray(problem.operator_func.L, dtype=np.float32)
    beta_op_full = np.asarray(problem.operator_func.beta, dtype=np.float32)

    if c_full.ndim == 0:
        c_full = np.full(n, float(c_full), dtype=np.float32)
    if L_full.ndim == 0:
        L_full = np.full(n, float(L_full), dtype=np.float32)
    if beta_op_full.ndim == 0:
        beta_op_full = np.full(n, float(beta_op_full), dtype=np.float32)

    vec_dtype = torch.float32 if str(parameters.get("dtype", "float32")).lower() in ("float32", "fp32") else torch.float64

    c_local = torch.tensor(c_full[start_idx:end_idx], device=device, dtype=vec_dtype)
    L_local = torch.tensor(L_full[start_idx:end_idx], device=device, dtype=vec_dtype)
    beta_op_local = torch.tensor(beta_op_full[start_idx:end_idx], device=device, dtype=vec_dtype)
    inv_beta_local = 1.0 / beta_op_local

    normalizer = torch.pow(1.0 / L_local, inv_beta_local) * (1.0 / beta_alg)
    normalizer_recip = torch.where(normalizer != 0, 1.0 / normalizer, torch.zeros_like(normalizer))

    p_const = float(5000.0 ** (1.0 / op_gamma))
    dp_const = - (1.0 / op_gamma) * p_const
    p_const_t = torch.tensor(p_const, device=device, dtype=vec_dtype)
    dp_const_t = torch.tensor(dp_const, device=device, dtype=vec_dtype)
    p_power = -1.0 / op_gamma
    dp_power = -1.0 / op_gamma - 1.0

    # ----------------------------
    # Helpers
    # ----------------------------
    def compute_global_Q(q_local_vec):
        Q_local = torch.sum(q_local_vec.to(torch.float64))
        dist.all_reduce(Q_local, op=dist.ReduceOp.SUM)
        return Q_local

    def compute_p_dp(Q_global):
        Q_safe = torch.clamp(Q_global, min=min_total_quantity).to(dtype=vec_dtype)
        p = p_const_t * torch.pow(Q_safe, p_power)
        dp = dp_const_t * torch.pow(Q_safe, dp_power)
        return p, dp

    def compute_F_local(q_local_vec):
        Q_global = compute_global_Q(q_local_vec)
        p_val, dp_val = compute_p_dp(Q_global)
        q_clamped = torch.clamp(q_local_vec, min=0.0)
        df_local = c_local + torch.pow(L_local * q_clamped, inv_beta_local)
        F_local = df_local - p_val - q_local_vec * dp_val
        return F_local

    def global_weighted_F_norm_sq(vec_local):
        local = torch.sum((normalizer * (vec_local ** 2)).to(torch.float64))
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local

    def global_weighted_u_norm_sq(vec_local):
        local = torch.sum((normalizer_recip * (vec_local ** 2)).to(torch.float64))
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local

    def compute_weighted_inner_products(u_diff, F_diff, Ftilde_diff=None):
        den = torch.sum((normalizer_recip * (u_diff ** 2)).to(torch.float64))
        num = torch.sum((normalizer * (F_diff ** 2)).to(torch.float64))
        dist.all_reduce(den, op=dist.ReduceOp.SUM)
        dist.all_reduce(num, op=dist.ReduceOp.SUM)

        if Ftilde_diff is None:
            return num, den, None

        num_hat = torch.sum((normalizer * (Ftilde_diff ** 2)).to(torch.float64))
        dist.all_reduce(num_hat, op=dist.ReduceOp.SUM)
        return num, den, num_hat

    def compute_residual_from_F(q_local_vec, F_local_vec):
        prox = torch.clamp(q_local_vec - F_local_vec, min=0.0)
        r_local = q_local_vec - prox
        norm_sq = torch.sum((r_local ** 2).to(torch.float64))
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM)
        return torch.sqrt(torch.clamp(norm_sq, min=0.0)).item()

    # ----------------------------
    # Initialize u0, F0
    # ----------------------------
    if u_0 is None:
        u0 = torch.ones(n_local, device=device, dtype=vec_dtype)
    else:
        u_0 = np.asarray(u_0, dtype=np.float32)
        if u_0.shape[0] != n:
            raise ValueError(f"u_0 must have shape ({n},), got {u_0.shape}.")
        u0 = torch.tensor(u_0[start_idx:end_idx], device=device, dtype=vec_dtype)

    F_0 = compute_F_local(u0)

    # ----------------------------
    # Line-search for the first step size a0 (full-block update)
    # ----------------------------
    alpha_ls = math.sqrt(2.0)
    i_ls = -1
    a0 = 1.0

    while True:
        i_ls += 1
        a0 = alpha_ls ** (-i_ls)
        u1 = torch.clamp(u0 - a0 * normalizer * F_0, min=0.0)
        F_1 = compute_F_local(u1)

        norm_F_sq = global_weighted_F_norm_sq(F_1 - F_0)
        norm_u_sq = global_weighted_u_norm_sq(u1 - u0)

        norm_F = torch.sqrt(torch.clamp(norm_F_sq, min=0.0)).item()
        norm_u = torch.sqrt(torch.clamp(norm_u_sq, min=0.0)).item()

        if (a0 * norm_F <= C * norm_u) and (a0 * norm_F <= C_hat * norm_u):
            break

    while True:
        if (a0 * norm_F >= (C / alpha_ls) * norm_u) or (a0 * norm_F >= (C_hat / alpha_ls) * norm_u):
            break
        a0 = a0 * alpha_ls
        u1 = torch.clamp(u0 - a0 * normalizer * F_0, min=0.0)
        F_1 = compute_F_local(u1)

        norm_F_sq = global_weighted_F_norm_sq(F_1 - F_0)
        norm_u_sq = global_weighted_u_norm_sq(u1 - u0)

        norm_F = torch.sqrt(torch.clamp(norm_F_sq, min=0.0)).item()
        norm_u = torch.sqrt(torch.clamp(norm_u_sq, min=0.0)).item()

    # Initialize states at k=1 (matching the NumPy code after line-search)
    u = u1.clone()
    u_prev = u0.clone()
    v_prev = u_prev.clone()

    F = F_1.clone()
    F_prev = F_0.clone()
    tilde = F_prev.clone()
    tilde_prev = F_prev.clone()

    a_prev = float(a0)
    a_curr = float(a0)
    A_accum = 0.0

    # Results (rank 0 only)
    results = Results() if rank == 0 else None
    start_time = time.time()

    init_measure = compute_residual_from_F(u, F)
    if rank == 0:
        logresult(results, 1, 0.0, init_measure)

    # Iteration counters
    k = 0
    exit_flag = False

    # ----------------------------
    # Main loop
    # ----------------------------
    while not exit_flag:
        # Step size selection (distributed)
        u_diff = u - u_prev
        F_diff = F - F_prev
        Ftilde_diff = F - tilde

        num, den, num_hat = compute_weighted_inner_products(u_diff, F_diff, Ftilde_diff)
        den_val = float(den.item())
        num_val = float(num.item())
        num_hat_val = float(num_hat.item()) if num_hat is not None else 0.0

        if den_val <= eps:
            L = 0.0
            L_hat = 0.0
        else:
            L = math.sqrt(max(num_val, 0.0) / max(den_val, eps))
            L_hat = math.sqrt(max(num_hat_val, 0.0) / max(den_val, eps))

        ratio = math.sqrt(max(a_curr, eps) / max(a_prev, eps))
        step_1 = rho_0 * a_curr
        step_2 = float("inf") if L <= 0.0 else (C / L) * ratio
        step_3 = float("inf") if L_hat <= 0.0 else (C_hat / L_hat) * ratio
        step = min(step_1, step_2, step_3)
        if not math.isfinite(step):
            step = step_1

        # Broadcast step to avoid drift
        step_t = torch.tensor([step], device=device, dtype=torch.float64)
        dist.broadcast(step_t, src=0)
        step = float(step_t.item())

        # Update step sizes and accumulator
        a_old = a_curr
        a_prev = a_curr
        a_curr = step
        A_accum += a_curr

        # F_bar = tilde + (a_old/a_curr) * (F_prev - tilde_prev)
        ratio_bar = a_old / max(a_curr, eps)
        F_bar = tilde + ratio_bar * (F_prev - tilde_prev)

        # v = (1-beta) u + beta v_prev
        v = (1.0 - beta_alg) * u + beta_alg * v_prev

        # Save previous iterate
        u_prev = u.clone()

        # Prox step
        u_new = torch.clamp(v - a_curr * normalizer * F_bar, min=0.0)

        # Update v_prev
        v_prev = v

        # Compute new operator values F(u_{k+1})
        F_new = compute_F_local(u_new)

        # Shift stored operator values for next iteration
        tilde_prev = tilde
        tilde = F
        F_prev = F
        F = F_new

        # Update iterate
        u = u_new

        # Increment iteration counters
        k += m

        # Logging and exit checks
        if k % (m * exit_criterion.loggingfreq) == 0:
            elapsed = time.time() - start_time
            measure = compute_residual_from_F(u, F)

            if rank == 0:
                logging.info(f"[torch_dist] elapsed_time: {elapsed:.4f}, iteration: {k}, opt_measure: {measure}")
                logresult(results, k, elapsed, measure, L=L, L_hat=L_hat)

            # Exit decision on rank 0, broadcast to others
            exit_t = torch.tensor([0], device=device, dtype=torch.int32)
            if rank == 0:
                exit_t[0] = 1 if CheckExitCondition(exit_criterion, k, elapsed, measure) else 0
            dist.broadcast(exit_t, src=0)
            exit_flag = bool(int(exit_t.item()))

    # ----------------------------
    # Gather full solution u to rank 0
    # ----------------------------
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
