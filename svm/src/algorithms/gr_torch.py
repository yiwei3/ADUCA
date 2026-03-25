import math
import time
import logging
import numpy as np

from src.algorithms.utils.results import Results, logresult
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.algorithms.utils.helper import construct_block_range


def _as_int_blocksize(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            return int(default)
        v = int(value)
        return v if v > 0 else int(default)
    except Exception:
        return int(default)


def _coerce_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("1", "true", "yes", "y", "on"):
            return True
        if val in ("0", "false", "no", "n", "off", ""):
            return False
    return bool(value)


def _select_device(parameters):
    import torch

    device = parameters.get("device") or parameters.get("torch_device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    return torch.device(device)


def _select_dtype(parameters):
    import torch

    dtype_str = str(parameters.get("dtype") or parameters.get("torch_dtype") or "float32").lower()
    if dtype_str in ("float64", "fp64", "double"):
        return torch.float64
    return torch.float32


def _scipy_csr_to_torch_csr(mat, shape, device, dtype):
    import torch

    mat = mat.tocsr()
    crow = torch.tensor(mat.indptr, device=device, dtype=torch.int64)
    col = torch.tensor(mat.indices, device=device, dtype=torch.int64)
    val = torch.tensor(mat.data, device=device, dtype=dtype)
    return torch.sparse_csr_tensor(crow, col, val, size=shape, device=device, dtype=dtype)


def _build_matvecs(problem: GMVIProblem, device, dtype, parameters):
    import torch

    d = int(problem.operator_func.d)
    n = int(problem.operator_func.n)
    A_sparse = getattr(problem.operator_func, "A_sparse", None)
    A_dense = getattr(problem.operator_func, "A", None)

    use_dense = _coerce_bool(parameters.get("use_dense", False))
    dense_threshold = float(parameters.get("dense_threshold", 0.25))
    if not use_dense and A_sparse is not None:
        try:
            density = float(A_sparse.nnz) / float(max(1, n * d))
            if density >= dense_threshold:
                use_dense = True
        except Exception:
            pass

    if use_dense:
        if A_dense is None and A_sparse is not None:
            A_dense = A_sparse.toarray()
        if A_dense is None:
            raise RuntimeError("No matrix data available for dense matvec.")
        A_t = torch.tensor(A_dense, device=device, dtype=dtype)

        def matvec_A(x_vec):
            return A_t.matmul(x_vec)

        def matvec_AT(v_vec):
            return A_t.t().matmul(v_vec)

        return matvec_A, matvec_AT, use_dense

    if A_sparse is None:
        from scipy.sparse import csr_matrix
        if A_dense is None:
            raise RuntimeError("No matrix data available for sparse matvec.")
        A_sparse = csr_matrix(A_dense)

    A_csr = A_sparse.tocsr()
    A_t = _scipy_csr_to_torch_csr(A_csr, (n, d), device, dtype)
    A_T_t = _scipy_csr_to_torch_csr(A_csr.transpose().tocsr(), (d, n), device, dtype)

    def matvec_A(x_vec):
        return torch.sparse.mm(A_t, x_vec.unsqueeze(1)).squeeze(1)

    def matvec_AT(v_vec):
        return torch.sparse.mm(A_T_t, v_vec.unsqueeze(1)).squeeze(1)

    return matvec_A, matvec_AT, use_dense


def _compute_normalizers(problem: GMVIProblem, use_dense: bool):
    A_sparse = getattr(problem.operator_func, "A_sparse", None)
    A_dense = getattr(problem.operator_func, "A", None)
    b = np.asarray(problem.operator_func.b, dtype=np.float64)
    b_sq = b * b

    if use_dense:
        if A_dense is None and A_sparse is not None:
            A_dense = A_sparse.toarray()
        if A_dense is None:
            raise RuntimeError("No matrix data available for normalizers.")
        A_scaled = A_dense * b[:, None]
        col_sq = np.sum(A_scaled ** 2, axis=0)
        row_sq = np.sum(A_scaled ** 2, axis=1)
    else:
        if A_sparse is None:
            from scipy.sparse import csr_matrix
            if A_dense is None:
                raise RuntimeError("No matrix data available for normalizers.")
            A_sparse = csr_matrix(A_dense)
        A_sq = A_sparse.power(2)
        if not np.allclose(b_sq, 1.0):
            A_sq = A_sq.multiply(b_sq[:, None])
        row_sq = np.asarray(A_sq.sum(axis=1)).reshape(-1)
        col_sq = np.asarray(A_sq.sum(axis=0)).reshape(-1)

    col_norm = np.sqrt(np.maximum(col_sq, 0.0))
    row_norm = np.sqrt(np.maximum(row_sq, 0.0))
    normalizer_x = np.where(col_norm != 0.0, 1.0 / col_norm, 1.0)
    normalizer_y = np.where(row_norm != 0.0, 1.0 / row_norm, 1.0)
    return normalizer_x, normalizer_y


def _prox_elastic_net_torch(z, tau, lambda1: float, lambda2: float):
    import torch

    p1 = tau * float(lambda1)
    p2 = 1.0 / (1.0 + tau * float(lambda2))
    return p2 * torch.sign(z) * torch.clamp(torch.abs(z) - p1, min=0.0)


def gr_torch(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, x_0=None):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to run GR_TORCH.") from exc

    d = int(problem.operator_func.d)
    n = int(problem.operator_func.n)
    block_size = _as_int_blocksize(parameters.get("block_size", 1), default=1)
    block_size_2 = _as_int_blocksize(parameters.get("block_size_2", n), default=n)
    blocks_1 = construct_block_range(begin=0, end=d, block_size=block_size)
    blocks_2 = construct_block_range(begin=d, end=d + n, block_size=block_size_2)
    m_1 = len(blocks_1)
    m_2 = len(blocks_2)
    m = m_1 + m_2
    logging.info(f"m_1 = {m_1}")
    logging.info(f"m_2 = {m_2}")
    logging.info(f"m = {m}")

    beta = float(parameters["beta"])
    rho = beta + beta ** 2

    device = _select_device(parameters)
    dtype = _select_dtype(parameters)
    matvec_A, matvec_AT, _ = _build_matvecs(problem, device, dtype, parameters)
    b = torch.tensor(problem.operator_func.b, device=device, dtype=dtype)

    if x_0 is None:
        x_0_t = torch.zeros(d + n, device=device, dtype=dtype)
    else:
        x_0_t = torch.tensor(np.asarray(x_0), device=device, dtype=dtype)

    x_1 = torch.full((d + n,), -0.0001, device=device, dtype=dtype)

    x = x_1.clone()
    x_ = x_0_t.clone()
    v = x_1.clone()
    v_ = x_1.clone()

    a = 1.0
    a_ = 1.0
    A = 1.0
    x_hat = a * x

    def compute_F(x_vec):
        x_part = x_vec[:d]
        y_part = x_vec[d:]
        Fx = matvec_AT(b * y_part) / float(n)
        Fy = (1.0 - b * matvec_A(x_part)) / float(n)
        return torch.cat((Fx, Fy))

    def gr_stepsize(a_val, a_prev, x_val, x_prev, F_val, F_prev):
        step_1 = rho * a_val
        F_norm = torch.norm(F_val - F_prev).item()
        if F_norm == 0.0:
            return step_1, 0.0
        x_norm = torch.norm(x_val - x_prev).item()
        if x_norm == 0.0:
            L = float("inf")
        else:
            L = F_norm / x_norm
        step_2 = 0.0 if not math.isfinite(L) else 1.0 / ((4 * beta ** 2 * a_prev) * L ** 2)
        step = min(step_1, step_2)
        return step, L

    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0_t.detach().cpu().numpy())
    logresult(results, 0, 0.0, init_opt_measure)

    lambda1 = float(problem.g_func.lambda1)
    lambda2 = float(problem.g_func.lambda2)

    with torch.no_grad():
        F = compute_F(x)
        F_ = compute_F(x_)

        while not exit_flag:
            step, L = gr_stepsize(a, a_, x, x_, F, F_)
            a_ = a
            a = step
            A += a

            v = (1 - beta) * x + beta * v_
            v_ = v.clone()
            x_ = x.clone()

            z = v - a * F
            x_new_x = _prox_elastic_net_torch(z[:d], a, lambda1, lambda2)
            x_new_y = torch.clamp(z[d:], min=-1.0, max=0.0)
            x = torch.cat((x_new_x, x_new_y))

            F_ = F.clone()
            F = compute_F(x)

            x_hat = (A - a) / A * x_hat + a / A * x

            iteration += 1
            if iteration % exit_criterion.loggingfreq == 0:
                elapsed_time = time.time() - start_time
                opt_measure = problem.func_value(x.detach().cpu().numpy())
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
                logresult(results, iteration, elapsed_time, opt_measure, L=L)
                exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x.detach().cpu().numpy()


def gr_torch_normalized(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, x_0=None):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to run GR_TORCH_normalized.") from exc

    d = int(problem.operator_func.d)
    n = int(problem.operator_func.n)
    block_size = _as_int_blocksize(parameters.get("block_size", 1), default=1)
    block_size_2 = _as_int_blocksize(parameters.get("block_size_2", n), default=n)
    blocks_1 = construct_block_range(begin=0, end=d, block_size=block_size)
    blocks_2 = construct_block_range(begin=d, end=d + n, block_size=block_size_2)
    m_1 = len(blocks_1)
    m_2 = len(blocks_2)
    m = m_1 + m_2
    logging.info(f"m_1 = {m_1}")
    logging.info(f"m_2 = {m_2}")
    logging.info(f"m = {m}")

    beta = float(parameters["beta"])
    rho = beta + beta ** 2

    device = _select_device(parameters)
    dtype = _select_dtype(parameters)
    matvec_A, matvec_AT, use_dense = _build_matvecs(problem, device, dtype, parameters)
    b = torch.tensor(problem.operator_func.b, device=device, dtype=dtype)

    normalizer_x_np, normalizer_y_np = _compute_normalizers(problem, use_dense)
    normalizer_x = torch.tensor(normalizer_x_np, device=device, dtype=dtype)
    normalizer_y = torch.tensor(normalizer_y_np, device=device, dtype=dtype)

    if x_0 is None:
        x_0_t = torch.zeros(d + n, device=device, dtype=dtype)
    else:
        x_0_t = torch.tensor(np.asarray(x_0), device=device, dtype=dtype)

    x_1 = torch.full((d + n,), -0.0001, device=device, dtype=dtype)

    x = x_1.clone()
    x_ = x_0_t.clone()
    v = x_1.clone()
    v_ = x_1.clone()

    a = 1.0
    a_ = 1.0
    A = 1.0
    x_hat = a * x

    def compute_F(x_vec):
        x_part = x_vec[:d]
        y_part = x_vec[d:]
        Fx = matvec_AT(b * y_part) / float(n)
        Fy = (1.0 - b * matvec_A(x_part)) / float(n)
        return torch.cat((Fx, Fy))

    def gr_stepsize(a_val, a_prev, x_val, x_prev, F_val, F_prev):
        step_1 = rho * a_val
        F_norm = torch.norm(F_val - F_prev).item()
        if F_norm == 0.0:
            return step_1, 0.0
        x_norm = torch.norm(x_val - x_prev).item()
        if x_norm == 0.0:
            L = float("inf")
        else:
            L = F_norm / x_norm
        step_2 = 0.0 if not math.isfinite(L) else 1.0 / ((4 * beta ** 2 * a_prev) * L ** 2)
        step = min(step_1, step_2)
        return step, L

    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0_t.detach().cpu().numpy())
    logresult(results, 0, 0.0, init_opt_measure)

    lambda1 = float(problem.g_func.lambda1)
    lambda2 = float(problem.g_func.lambda2)

    with torch.no_grad():
        F = compute_F(x)
        F_ = compute_F(x_)

        while not exit_flag:
            step, L = gr_stepsize(a, a_, x, x_, F, F_)
            a_ = a
            a = step
            A += a

            v = (1 - beta) * x + beta * v_
            v_ = v.clone()
            x_prev = x.clone()
            x_ = x.clone()

            F_prev = F.clone()

            z_x = v[:d] - a * normalizer_x * F_prev[:d]
            tau_x = a * normalizer_x
            x_new_x = _prox_elastic_net_torch(z_x, tau_x, lambda1, lambda2)

            z_y = v[d:] - a * normalizer_y * F_prev[d:]
            x_new_y = torch.clamp(z_y, min=-1.0, max=0.0)
            x = torch.cat((x_new_x, x_new_y))

            F = compute_F(x)
            F_ = F_prev

            x_hat = (A - a) / A * x_hat + a / A * x

            iteration += 1
            if iteration % exit_criterion.loggingfreq == 0:
                elapsed_time = time.time() - start_time
                opt_measure = problem.func_value(x.detach().cpu().numpy())
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
                logresult(results, iteration, elapsed_time, opt_measure, L=L)
                exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x.detach().cpu().numpy()
