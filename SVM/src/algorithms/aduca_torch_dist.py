
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


def _prox_elastic_net_torch(z, tau, lambda1: float, lambda2: float):
    """
    Elastic-net prox on x-part:
        prox_{tau*(lambda1*||.||_1 + (lambda2/2)||.||^2)}(z)
      = (1/(1+tau*lambda2)) * sign(z) * max(|z| - tau*lambda1, 0)
    tau is a tensor (same shape as z).
    """
    import torch

    p1 = tau * float(lambda1)
    p2 = 1.0 / (1.0 + tau * float(lambda2))
    return p2 * torch.sign(z) * torch.clamp(torch.abs(z) - p1, min=0.0)


def _aduca_torch_distributed_svm(problem: GMVIProblem,
                                exit_criterion: ExitCriterion,
                                parameters,
                                u_0: Optional[np.ndarray] = None):
    """
    Multi-GPU / multi-process ADUCA implementation for the SVM-ElasticNet saddle-point
    formulation implemented in SVMElasticOprFunc + SVMElasticGFunc.

    Design:
    - x (dimension d) is replicated on every rank.
    - y (dimension n) and rows of A are sharded across ranks.
    - F_x = (A^T (b ⊙ y))/n is computed via local A_i^T(...) + all_reduce(sum).
    - F_y = (1 - b ⊙ (A x))/n is computed locally (row-sharded).
    - This directly exploits the delay structure you pointed out:
        * For x-blocks, F depends only on y (previous iterate is enough).
        * For y-blocks, F depends only on x (current iterate is enough).
      so we update all blocks "Jacobi-style" in parallel (same iterate, same epoch).

    Notes:
    - Launch with torchrun, e.g.:
        torchrun --nproc_per_node=8 run_algos.py --algo ADUCA ...
    - Only rank 0 returns populated Results + full u; other ranks return dummy outputs.
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
    d = int(problem.operator_func.d)
    n = int(problem.operator_func.n)

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    mu = float(parameters.get("mu", 0.0))
    eps = float(parameters.get("eps", 1e-12))
    strong_convexity = bool(parameters.get("strong_convexity", parameters.get("strong-convexity", False)))

    lambda1 = float(getattr(problem.g_func, "lambda1", 0.0))
    lambda2 = float(getattr(problem.g_func, "lambda2", 0.0))

    # Algorithm constants (same as your current aduca.py)
    rho_0 = min(rho, beta * (1 + beta) * (1 - gamma))
    eta = ((gamma * (1 + beta)) / (1 + beta ** 2)) ** 0.5
    tau_const = (3 * rho_0 ** 2 * (1 + rho * beta) / (2 * (rho * beta) ** 2 + 3 * rho_0 ** 2 * (1 + rho * beta)))
    C = eta / (2 * beta ** 0.5) * (tau_const ** 0.5 * rho * beta) / (3 ** 0.5 * (1 + rho * beta) ** 0.5)
    C_hat = eta / (2 * beta ** 0.5) * ((1 - tau_const) * rho * beta) ** 0.5 / 2 ** 0.5

    if rank == 0:
        logging.info(f"[torch_dist] world_size = {world_size}, device = {device}")
        logging.info(f"rho = {rho_0}")
        logging.info(f"C = {C}")
        logging.info(f"C_hat = {C_hat}")

    # Iteration accounting to match the original code (k += m per full block sweep)
    block_size = _as_int_blocksize(parameters.get("block_size", 1), default=1)
    block_size_2 = _as_int_blocksize(parameters.get("block_size_2", n), default=n)
    m_1 = (d + block_size - 1) // block_size
    m_2 = (n + block_size_2 - 1) // block_size_2
    m = int(m_1 + m_2)

    # ----------------------------
    # Shard data by rows
    # ----------------------------
    start_row, end_row = _split_range(n, world_size, rank)
    n_local = end_row - start_row
    if n_local <= 0:
        raise RuntimeError(f"Rank {rank} got empty shard. Check n={n}, world_size={world_size}.")

    # We rely on the existing SVMElasticOprFunc storage
    A_csr = getattr(problem.operator_func, "A_sparse", None)
    if A_csr is None:
        raise RuntimeError("problem.operator_func must expose A_sparse (scipy.sparse.csr_matrix).")

    b_full = getattr(problem.operator_func, "b", None)
    if b_full is None:
        raise RuntimeError("problem.operator_func must expose b (labels).")

    # Local slices (CPU)
    A_local_csr = A_csr[start_row:end_row]
    b_local_np = np.asarray(b_full[start_row:end_row], dtype=np.float32)

    # Decide matmul mode
    # - Sparse is the default (matches the current codebase).
    # - For (near-)dense libsvm datasets, you may set parameters['use_dense']=True.
    use_dense = bool(parameters.get("use_dense", False))
    dense_threshold = float(parameters.get("dense_threshold", 0.25))
    if not use_dense:
        try:
            density = float(A_local_csr.nnz) / float(max(1, n_local * d))
            if density >= dense_threshold:
                # If A is effectively dense, dense GEMV may be faster on GPU
                use_dense = True
                if rank == 0:
                    logging.info(f"[torch_dist] Auto-selected dense matmul (density ~ {density:.3f}).")
        except Exception:
            pass

    vec_dtype = torch.float32 if str(parameters.get("dtype", "float32")).lower() in ("float32", "fp32") else torch.float64

    # Convert b_local to device
    b_local = torch.tensor(b_local_np, device=device, dtype=vec_dtype)

    # Build local A and A^T on device
    def scipy_csr_to_torch_csr(mat, shape, device_, dtype_):
        # mat must be CSR
        import torch
        mat = mat.tocsr()
        crow = torch.tensor(mat.indptr, device=device_, dtype=torch.int64)
        col = torch.tensor(mat.indices, device=device_, dtype=torch.int64)
        val = torch.tensor(mat.data, device=device_, dtype=dtype_)
        return torch.sparse_csr_tensor(crow, col, val, size=shape, device=device_, dtype=dtype_)

    if use_dense:
        A_local_dense = torch.tensor(A_local_csr.toarray(), device=device, dtype=vec_dtype)
        A_local_T_dense = None  # use A_local_dense.t()
        A_local = A_local_dense
        A_local_T = None
        def matvec_A(x_vec):
            return A_local_dense.matmul(x_vec)
        def matvec_AT(v_vec):
            return A_local_dense.t().matmul(v_vec)
    else:
        # Sparse CSR on device
        A_local = scipy_csr_to_torch_csr(A_local_csr, (n_local, d), device, vec_dtype)
        A_local_T_csr = A_local_csr.transpose().tocsr()  # (d, n_local) CSR on CPU
        A_local_T = scipy_csr_to_torch_csr(A_local_T_csr, (d, n_local), device, vec_dtype)

        def matvec_A(x_vec):
            # (n_local, d) @ (d,) -> (n_local,)
            return torch.sparse.mm(A_local, x_vec.unsqueeze(1)).squeeze(1)

        def matvec_AT(v_vec):
            # (d, n_local) @ (n_local,) -> (d,)
            return torch.sparse.mm(A_local_T, v_vec.unsqueeze(1)).squeeze(1)

    # ----------------------------
    # Normalizers (diagonal preconditioner)
    #   x: 1 / ||A[:,j]||_2  (global column norms)  [b^2 = 1]
    #   y: 1 / ||A[i,:]||_2  (local row norms)
    # ----------------------------
    t0_init = time.time()

    # y-normalizers (row norms): local only
    if use_dense:
        row_sq = (A_local_dense.float() ** 2).sum(dim=1).cpu().numpy()
    else:
        # use SciPy for a cheap one-time row-sum on CPU
        row_sq = np.asarray(A_local_csr.power(2).sum(axis=1)).reshape(-1)
    row_norm = np.sqrt(np.maximum(row_sq, 0.0)).astype(np.float32)
    normalizer_y_np = np.where(row_norm != 0.0, 1.0 / row_norm, 1.0).astype(np.float32)
    normalizer_y = torch.tensor(normalizer_y_np, device=device, dtype=vec_dtype)
    normalizer_recip_y = torch.where(normalizer_y != 0, 1.0 / normalizer_y, torch.zeros_like(normalizer_y))

    # x-normalizers (column norms): distributed reduction of squared sums
    if use_dense:
        col_sq_local_np = (A_local_dense.float() ** 2).sum(dim=0).cpu().numpy()
    else:
        col_sq_local_np = np.asarray(A_local_csr.power(2).sum(axis=0)).reshape(-1)

    col_sq_local = torch.tensor(col_sq_local_np, device=device, dtype=torch.float64)
    dist.all_reduce(col_sq_local, op=dist.ReduceOp.SUM)
    col_norm = torch.sqrt(torch.clamp(col_sq_local, min=0.0)).to(dtype=vec_dtype)
    normalizer_x = torch.where(col_norm != 0.0, 1.0 / col_norm, torch.ones_like(col_norm))
    normalizer_recip_x = torch.where(normalizer_x != 0.0, 1.0 / normalizer_x, torch.zeros_like(normalizer_x))

    mu = mu / torch.min(normalizer_x.min(), normalizer_y.min()).item()
    logging.info(f"[torch_dist] Adjusted mu = {mu:.6e}")

    if rank == 0:
        logging.info(f"[torch_dist] Initialization time = {time.time() - t0_init:.4f} seconds")

    # ----------------------------
    # Helper: compute operator parts on distributed shards
    # ----------------------------
    def compute_F_parts(x_vec, y_local_vec):
        """
        Returns:
          F_x (d,) replicated (all_reduce)
          F_y_local (n_local,) shard-local
        """
        # F_y local
        Ax_local = matvec_A(x_vec)
        F_y_local = (1.0 - b_local * Ax_local) / float(n)

        # F_x global via all_reduce over local contributions
        by_local = b_local * y_local_vec
        g_local = matvec_AT(by_local)  # shape (d,)
        dist.all_reduce(g_local, op=dist.ReduceOp.SUM)
        F_x = g_local / float(n)
        return F_x, F_y_local

    def compute_weighted_inner_products(u_diff_x, u_diff_y,
                                        F_diff_x, F_diff_y,
                                        Ftilde_diff_x=None, Ftilde_diff_y=None):
        """
        Computes global (across ranks) numerator/denominator for L and L_hat with the same
        weighting as the NumPy implementation:
          num = <F_diff, normalizer * F_diff>
          den = <u_diff, normalizer_recip * u_diff>
        If Ftilde_diff_* are given, also returns num_hat = <Ftilde_diff, normalizer * Ftilde_diff>.
        """
        # Use float64 for stable reductions
        den_x = torch.sum((normalizer_recip_x * (u_diff_x ** 2)).to(torch.float64))
        den_y = torch.sum((normalizer_recip_y * (u_diff_y ** 2)).to(torch.float64))
        local_den = den_x / float(world_size) + den_y
        dist.all_reduce(local_den, op=dist.ReduceOp.SUM)

        num_x = torch.sum((normalizer_x * (F_diff_x ** 2)).to(torch.float64))
        num_y = torch.sum((normalizer_y * (F_diff_y ** 2)).to(torch.float64))
        local_num = num_x / float(world_size) + num_y
        dist.all_reduce(local_num, op=dist.ReduceOp.SUM)

        if Ftilde_diff_x is None or Ftilde_diff_y is None:
            return local_num, local_den, None

        num_hat_x = torch.sum((normalizer_x * (Ftilde_diff_x ** 2)).to(torch.float64))
        num_hat_y = torch.sum((normalizer_y * (Ftilde_diff_y ** 2)).to(torch.float64))
        local_num_hat = num_hat_x / float(world_size) + num_hat_y
        dist.all_reduce(local_num_hat, op=dist.ReduceOp.SUM)
        return local_num, local_den, local_num_hat

    # ----------------------------
    # Initialize u0, v0
    # ----------------------------
    if u_0 is None:
        x0 = torch.zeros(d, device=device, dtype=vec_dtype)
        y0 = torch.zeros(n_local, device=device, dtype=vec_dtype)
    else:
        u_0 = np.asarray(u_0)
        if u_0.shape[0] != d + n:
            raise ValueError(f"u_0 must have shape ({d+n},), got {u_0.shape}.")
        x0 = torch.tensor(u_0[:d], device=device, dtype=vec_dtype)
        y0_full = torch.tensor(u_0[d:], device=device, dtype=vec_dtype)
        y0 = y0_full[start_row:end_row].contiguous()

    # u, u_prev (like u and u_ in the NumPy code)
    x = x0.clone()
    y = y0.clone()
    x_prev = x0.clone()
    y_prev = y0.clone()

    # v_prev (v_ in NumPy code) starts at u0
    vx_prev = x0.clone()
    vy_prev = y0.clone()

    # Initial operator values
    F_x0, F_y0 = compute_F_parts(x0, y0)
    # tilde^0 = F(u0) in your NumPy code
    tilde_x0 = F_x0.clone()
    tilde_y0 = F_y0.clone()

    # ----------------------------
    # Line-search for the first step size a0 (same logic as NumPy reference)
    # ----------------------------
    alpha_ls = 2.0
    i_ls = 0
    a0 = 1.0

    def global_weighted_F_norm_sq(x_part, y_part):
        # x_part is replicated: divide by world_size before reduction
        num_x = torch.sum((normalizer_x * (x_part ** 2)).to(torch.float64))
        num_y = torch.sum((normalizer_y * (y_part ** 2)).to(torch.float64))
        local = num_x / float(world_size) + num_y
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local

    def global_weighted_u_norm_sq(x_part, y_part):
        # x_part is replicated: divide by world_size before reduction
        den_x = torch.sum((normalizer_recip_x * (x_part ** 2)).to(torch.float64))
        den_y = torch.sum((normalizer_recip_y * (y_part ** 2)).to(torch.float64))
        local = den_x / float(world_size) + den_y
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local

    # Initial prox and Lipschitz estimates
    z_x = x0 - a0 * (normalizer_x * F_x0)
    tau_x = a0 * normalizer_x
    x1 = _prox_elastic_net_torch(z_x, tau_x, lambda1, lambda2)

    z_y = y0 - a0 * (normalizer_y * F_y0)
    y1 = torch.clamp(z_y, min=-1.0, max=0.0)

    F_x1, F_y1 = compute_F_parts(x1, y1)
    tilde_x1 = F_x0
    tilde_y1 = F_y1

    norm_F_sq = global_weighted_F_norm_sq(F_x1 - F_x0, F_y1 - F_y0)
    norm_Ftilde_sq = global_weighted_F_norm_sq(F_x1 - tilde_x1, F_y1 - tilde_y1)
    norm_u_sq = global_weighted_u_norm_sq(x1 - x0, y1 - y0)

    norm_F = torch.sqrt(torch.clamp(norm_F_sq, min=0.0)).item()
    norm_Ftilde = torch.sqrt(torch.clamp(norm_Ftilde_sq, min=0.0)).item()
    norm_u = torch.sqrt(torch.clamp(norm_u_sq, min=0.0)).item()

    L_1 = norm_F / norm_u if norm_u != 0 else float("inf")
    L_hat_1 = norm_Ftilde / norm_u if norm_u != 0 else float("inf")
    a0 = min(C / L_1 if L_1 else float("inf"), C_hat / L_hat_1 if L_hat_1 else float("inf"))

    while True:
        a0 = alpha_ls ** (-i_ls)

        z_x = x0 - a0 * (normalizer_x * F_x0)
        tau_x = a0 * normalizer_x
        x1 = _prox_elastic_net_torch(z_x, tau_x, lambda1, lambda2)

        z_y = y0 - a0 * (normalizer_y * F_y0)
        y1 = torch.clamp(z_y, min=-1.0, max=0.0)

        F_x1, F_y1 = compute_F_parts(x1, y1)
        tilde_x1 = F_x0
        tilde_y1 = F_y1

        norm_F_sq = global_weighted_F_norm_sq(F_x1 - F_x0, F_y1 - F_y0)
        norm_u_sq = global_weighted_u_norm_sq(x1 - x0, y1 - y0)

        norm_F = torch.sqrt(torch.clamp(norm_F_sq, min=0.0)).item()
        norm_u = torch.sqrt(torch.clamp(norm_u_sq, min=0.0)).item()

        if (2 ** 0.5 * a0 * norm_F <= norm_u):
            break
        i_ls += 1

    x_init, y_init = x1, y1
    F_x_init, F_y_init = F_x1, F_y1
    tilde_x_init, tilde_y_init = tilde_x1, tilde_y1

    a_0 = a0

    # Initialize states at k=1 (matching the NumPy code after line-search)
    x = x_init.clone()
    y = y_init.clone()
    x_prev = x0.clone()
    y_prev = y0.clone()

    vx_prev = x0.clone()
    vy_prev = y0.clone()

    # F^1 and F^0
    F_x = F_x_init.clone()
    F_y = F_y_init.clone()
    F_x_prev = F_x0.clone()
    F_y_prev = F_y0.clone()

    # tilde^1 and tilde^0
    tilde_x = tilde_x_init.clone()
    tilde_y = tilde_y_init.clone()
    tilde_x_prev = tilde_x0.clone()
    tilde_y_prev = tilde_y0.clone()

    # Step sizes (a and a_ in NumPy code)
    a_prev = float(a0)  # corresponds to a_ (previous)
    a_curr = float(a0)  # corresponds to a (current)

    A_accum = 0.0  # matches "A" accumulator in NumPy code

    # Results (rank 0 only)
    results = Results() if rank == 0 else None
    start_time = time.time()

    # Initial objective value logging (iteration = 1, time = 0)
    # Compute objective in distributed manner (hinge via all_reduce; reg locally)
    def compute_objective(x_vec):
        Ax_local = matvec_A(x_vec)
        hinge_local = torch.clamp(1.0 - b_local * Ax_local, min=0.0).to(torch.float64).sum()
        dist.all_reduce(hinge_local, op=dist.ReduceOp.SUM)
        hinge = hinge_local.item() / float(n)

        # Regularizer on x (replicated)
        reg = float(lambda1) * torch.abs(x_vec).to(torch.float64).sum().item()
        reg += 0.5 * float(lambda2) * (x_vec.to(torch.float64) ** 2).sum().item()
        return hinge + reg

    init_measure = compute_objective(x)
    if rank == 0:
        logresult(results, 1, 0.0, init_measure)

    # Iteration counters
    k = 0  # "block queries"
    exit_flag = False

    # ----------------------------
    # Main loop
    # ----------------------------
    while not exit_flag:
        # Step size selection (distributed)
        u_diff_x = x - x_prev
        u_diff_y = y - y_prev

        F_diff_x = F_x - F_x_prev
        F_diff_y = F_y - F_y_prev

        Ftilde_diff_x = F_x - tilde_x
        Ftilde_diff_y = F_y - tilde_y

        num, den, num_hat = compute_weighted_inner_products(
            u_diff_x, u_diff_y, F_diff_x, F_diff_y, Ftilde_diff_x, Ftilde_diff_y
        )

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

        # Broadcast step to avoid drift
        step_t = torch.tensor([step], device=device, dtype=torch.float64)
        dist.broadcast(step_t, src=0)
        step = float(step_t.item())

        # Update step sizes and accumulator
        a_old = a_curr
        a_prev = a_curr
        a_curr = step
        A_accum += a_curr

        # F_bar = tilde + ratio_bar * (F_prev - tilde_prev)
        if strong_convexity:
            omega_k = (1.0 + rho * beta * mu * a_curr) / (1.0 + mu * a_curr)
            ratio_bar = (a_old * omega_k) / max(a_curr, eps)
        else:
            ratio_bar = a_old / max(a_curr, eps)
        Fbar_x = tilde_x + ratio_bar * (F_x_prev - tilde_x_prev)
        Fbar_y = tilde_y + ratio_bar * (F_y_prev - tilde_y_prev)

        # v = (1-beta) u + beta v_prev
        vx = (1.0 - beta) * x + beta * vx_prev
        vy = (1.0 - beta) * y + beta * vy_prev

        # Save previous iterate
        x_prev = x.clone()
        y_prev = y.clone()

        # Prox step (diagonal scaling)
        z_x = vx - a_curr * (normalizer_x * Fbar_x)
        tau_x = a_curr * normalizer_x
        x_new = _prox_elastic_net_torch(z_x, tau_x, lambda1, lambda2)

        z_y = vy - a_curr * (normalizer_y * Fbar_y)
        y_new = torch.clamp(z_y, min=-1.0, max=0.0)

        # Update v_prev
        vx_prev = vx
        vy_prev = vy

        # Compute new operator values F(u_{k+1})
        F_x_new, F_y_new = compute_F_parts(x_new, y_new)

        # Build new tilde:
        #   tilde_{k+1,x} = F_x(y_k) = current F_x
        #   tilde_{k+1,y} = F_y(x_{k+1}) = F_y_new (since depends only on x_new)
        tilde_x_new = F_x.clone()
        tilde_y_new = F_y_new.clone()

        # Shift stored operator values for next iteration
        F_x_prev = F_x
        F_y_prev = F_y
        F_x = F_x_new
        F_y = F_y_new

        tilde_x_prev = tilde_x
        tilde_y_prev = tilde_y
        tilde_x = tilde_x_new
        tilde_y = tilde_y_new

        # Update iterate
        x = x_new
        y = y_new

        # Increment iteration counters
        k += m

        # Logging and exit checks
        if k % (m * exit_criterion.loggingfreq) == 0:
            elapsed = time.time() - start_time
            measure = compute_objective(x)

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
    # Gather full solution u = [x; y] to rank 0
    # ----------------------------
    # Gather y shards with padding (NCCL-friendly)
    sizes = torch.tensor([n_local], device=device, dtype=torch.int64)
    sizes_list = [torch.empty_like(sizes) for _ in range(world_size)]
    dist.all_gather(sizes_list, sizes)
    sizes_cpu = [int(s.item()) for s in sizes_list]
    max_n_local = max(sizes_cpu)

    y_pad = torch.zeros(max_n_local, device=device, dtype=vec_dtype)
    y_pad[:n_local] = y

    y_gather = [torch.empty_like(y_pad) for _ in range(world_size)]
    dist.all_gather(y_gather, y_pad)

    if rank == 0:
        y_full = torch.cat([y_gather[r][:sizes_cpu[r]].cpu() for r in range(world_size)], dim=0).numpy()
        x_full = x.cpu().numpy()
        u_full = np.concatenate([x_full, y_full], axis=0)
        return results, u_full

    # Non-zero ranks return dummy outputs (caller should not serialize them)
    return Results(), np.zeros((d + n,), dtype=np.float32)


def _aduca_numpy_reference(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    """
    This is the original NumPy implementation you provided (kept for backwards compatibility).
    """
    # Init of ADUCA.
    d = problem.operator_func.d
    n = problem.operator_func.n
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    rho = parameters["rho"]
    eps = parameters.get("eps", 1e-12)

    rho_0 = min(rho, beta * (1 + beta) * (1 - gamma))
    eta = ((gamma * (1 + beta)) / (1 + beta ** 2)) ** 0.5
    tau = (3 * rho_0 ** 2 * (1 + rho * beta) / (2 * (rho * beta) ** 2 + 3 * rho_0 ** 2 * (1 + rho * beta)))
    C = eta / (2 * beta ** 0.5) * (tau ** 0.5 * rho * beta) / (3 ** 0.5 * (1 + rho * beta) ** 0.5)
    C_hat = eta / (2 * beta ** 0.5) * ((1 - tau) * rho * beta) ** 0.5 / 2 ** 0.5
    logging.info(f"rho = {rho_0}")
    logging.info(f"C = {C}")
    logging.info(f"C_hat = {C_hat}")

    # Scale the blocks with respect to different variables (x and y).
    block_size = parameters['block_size']
    blocks_1 = construct_block_range(begin=0, end=d, block_size=block_size)
    block_size_2 = parameters['block_size_2']
    blocks_2 = construct_block_range(begin=d, end=d + n, block_size=block_size_2)
    blocks = blocks_1 + blocks_2
    m_1 = len(blocks_1)
    m_2 = len(blocks_2)
    m = len(blocks)
    logging.info(f"m_1 = {m_1}")
    logging.info(f"m_2 = {m_2}")
    logging.info(f"m = {m}")

    # normalizers
    time_start_initialization = time.time()
    A_matrix = problem.operator_func.A
    A_matrix_T = A_matrix.T
    b = problem.operator_func.b

    normalizers_1 = []
    for block in blocks_1:
        size = block.stop - block.start
        normalizer = np.zeros(shape=size)
        for i in range(block.start, block.stop):
            norm = np.linalg.norm(b * A_matrix_T[i])
            if norm != 0:
                normalizer[i - block.start] = 1 / norm
            else:
                normalizer[i - block.start] = 1
        normalizers_1.append(normalizer)

    normalizers_2 = []
    max_norm = 0

    for block in blocks_2:
        size = block.stop - block.start
        normalizer = np.zeros(size)
        for i in range(block.start, block.stop):
            norm = np.linalg.norm(b[i - d] * A_matrix[i - d])
            if norm > max_norm:
                max_norm = norm
            if norm != 0:
                normalizer[i - block.start] = 1 / norm
            else:
                normalizer[i - block.start] = 1
        normalizers_2.append(normalizer)

    normalizers = normalizers_1 + normalizers_2
    normalizers = np.concatenate(normalizers, axis=0)
    normalizers_recip = np.where(normalizers != 0, 1 / normalizers, 0)

    time_end_initialization = time.time()
    logging.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

    a = 0
    a_ = 0
    A = 0

    if u_0 is None:
        u_0 = np.zeros(problem.d)
    u_ = np.copy(u_0)
    u_hat = np.zeros(problem.d)
    v = np.zeros(problem.d)
    v_ = np.zeros(problem.d)

    F = np.zeros(problem.d)
    F_ = np.zeros(problem.d)
    F_tilde = np.zeros(problem.d)
    F_tilde_ = np.zeros(problem.d)
    F_bar = np.copy(F_tilde)

    k = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(u_)
    logresult(results, 1, 0.0, init_opt_measure)

    u = np.copy(u_0)
    u_ = np.copy(u_0)

    F_0 = problem.operator_func.func_map(u_0)
    F_tilde_0 = np.copy(F_0)
    F_tilde_1 = np.copy(F_tilde_0)

    def aduca_stepsize(normalizer, normalizer_recip, u, u_, a, a_, F, F_, F_tilde):
        step_1 = rho_0 * a
        u_diff = np.copy(u - u_)
        F_diff = np.copy(F - F_)
        L_k = np.sqrt(np.inner(F_diff, (normalizer * F_diff)) / (np.inner(u_diff, (normalizer_recip * u_diff))))
        if L_k == 0:
            step_2 = np.inf
        else:
            step_2 = C / L_k * (a / a_) ** 0.5
        F_tilde_diff = np.copy(F - F_tilde)
        L_hat_k = np.sqrt(
            np.inner(F_tilde_diff, (normalizer * F_tilde_diff)) / (np.inner(u_diff, (normalizer_recip * u_diff))))
        if L_hat_k == 0:
            step_3 = np.inf
        else:
            step_3 = (C_hat / L_hat_k) * (a / a_) ** 0.5
        step = min(step_1, step_2, step_3)
        return step, L_k, L_hat_k

    # line-search for the first step (matching NumPy implementation)
    alpha = 2
    i = 0
    a_0 = 1

    F_store = np.copy(F_0)
    u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0, a_0 * normalizers[:d], d)
    for block in blocks:
        F_tilde_1[block] = F_store[block]
        F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
    F_1 = np.copy(F_store)
    F_diff = F_1 - F_0
    F_tilde_diff = F_1 - F_tilde_1
    u_diff = u_1 - u_0
    norm_F = np.sqrt(np.inner(F_diff, normalizers * F_diff))
    norm_F_tilde = np.sqrt(np.inner(F_tilde_diff, normalizers * F_tilde_diff))
    norm_u = np.sqrt(np.inner(u_diff, normalizers_recip * u_diff))

    L_1 = norm_F / norm_u
    L_hat_1 = norm_F_tilde / norm_u

    a_0 = min(C / L_1, C_hat / L_hat_1)

    while True:
        F_store = np.copy(F_0)
        a_0 = alpha ** (-i)

        u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0, a_0 * normalizers[:d], d)

        for block in blocks:
            F_tilde_1[block] = F_store[block]
            F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)

        F_1 = np.copy(F_store)
        F_diff = F_1 - F_0
        F_tilde_diff = F_1 - F_tilde_1
        u_diff = u_1 - u_0
        norm_F = np.sqrt(np.inner(F_diff, normalizers * F_diff))
        norm_F_tilde = np.sqrt(np.inner(F_tilde_diff, normalizers * F_tilde_diff))
        norm_u = np.sqrt(np.inner(u_diff, normalizers_recip * u_diff))
        if (2 ** 0.5 * a_0 * norm_F <= norm_u):
            break
        i += 1

    a_ = a_0
    a = a_0
    A = 0

    u = np.copy(u_1)
    u_ = np.copy(u_0)
    v_ = np.copy(u_)
    u_hat = A * u_

    F = np.copy(F_1)
    F_ = np.copy(F_0)
    F_tilde = np.copy(F_tilde_1)
    F_tilde_ = np.copy(F_tilde_0)
    F_bar = np.zeros(problem.d)

    while not exit_flag:
        step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_, F, F_, F_tilde)
        a_ = a
        a = step
        A += a

        for index, block in enumerate(blocks, start=0):
            F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
            v[block] = (1 - beta) * u[block] + beta * v_[block]
            u_[block] = u[block]
            if block.stop <= d:
                u[block] = problem.g_func.prox_opr_block(block, v[block] - a * normalizers_1[index] * F_bar[block], a * normalizers_1[index])
            else:
                u[block] = problem.g_func.prox_opr_block(block, v[block] - a * normalizers_2[index - m_1] * F_bar[block], a)
            F_tilde_[block] = F_tilde[block]
            F_tilde[block] = F_store[block]
            F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block)

        np.copyto(F_, F)
        F = np.copy(F_store)
        np.copyto(v_, v)

        u_hat = ((A - a) * u_hat / A) + (a * u_ / A)

        k += m

        if k % (m * exit_criterion.loggingfreq) == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(u)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
            logresult(results, k, elapsed_time, opt_measure, L=L, L_hat=L_hat)
            exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)
            if exit_flag:
                break

    return results, u


def aduca_distributed(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    """
    Unified entry point.

    - Default: original NumPy implementation (single-process).
    - Set parameters["backend"] = "torch_dist" (or parameters["torch_distributed"]=True) to enable
      PyTorch Distributed (multi-process, multi-GPU).
    """
    backend = str(parameters.get("backend", "numpy")).lower()
    if parameters.get("torch_distributed", True) or backend in ("torch_dist", "torch_distributed", "ddp", "pytorch_dist"):
        logging.info("Using PyTorch Distributed backend for ADUCA SVM.")
        return _aduca_torch_distributed_svm(problem, exit_criterion, parameters, u_0=u_0)
    return _aduca_numpy_reference(problem, exit_criterion, parameters, u_0=u_0)
