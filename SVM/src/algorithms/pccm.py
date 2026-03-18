import logging
import time

import numpy as np

from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.algorithms.utils.helper import construct_block_range
from src.algorithms.utils.results import Results, logresult
from src.problems.GMVI_func import GMVIProblem


def _compute_normalizers(problem: GMVIProblem, blocks_1, blocks_2):
    """Build per-block normalizers matching ADUCA scaling."""
    A_matrix = problem.operator_func.A
    A_matrix_T = A_matrix.T
    b = problem.operator_func.b
    d = problem.operator_func.d

    normalizers_1 = []
    for block in blocks_1:
        size = block.stop - block.start
        normalizer = np.zeros(shape=size)
        for i in range(block.start, block.stop):
            norm = np.linalg.norm(b * A_matrix_T[i])
            normalizer[i - block.start] = 1 / norm if norm != 0 else 1
        normalizers_1.append(normalizer)

    normalizers_2 = []
    for block in blocks_2:
        size = block.stop - block.start
        normalizer = np.zeros(size)
        for i in range(block.start, block.stop):
            norm = np.linalg.norm(b[i - d] * A_matrix[i - d])
            normalizer[i - block.start] = 1 / norm if norm != 0 else 1
        normalizers_2.append(normalizer)

    return normalizers_1, normalizers_2


def _is_svm_elastic_bipartite(problem: GMVIProblem) -> bool:
    """See coder._is_svm_elastic_bipartite for motivation."""
    F = getattr(problem, "operator_func", None)
    if F is None:
        return False

    required_attrs = ("d", "n", "A_sparse", "b")
    if not all(hasattr(F, a) for a in required_attrs):
        return False

    if not callable(getattr(F, "func_map_block_update", None)):
        return False

    return True


def _concat_normalizers(norm_list) -> np.ndarray:
    if not norm_list:
        return np.array([], dtype=float)
    return np.concatenate(norm_list, axis=0)


def pccm(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    """PCCM for GMVI, with an SVM-specialized two-phase update."""

    d = problem.operator_func.d
    n = problem.operator_func.n
    L = parameters["L"]
    mu = parameters["mu"]

    block_size = parameters["block_size"]
    blocks_1 = construct_block_range(begin=0, end=d, block_size=block_size)
    block_size_2 = parameters["block_size_2"]
    blocks_2 = construct_block_range(begin=d, end=d + n, block_size=block_size_2)
    blocks = blocks_1 + blocks_2

    m_1 = len(blocks_1)
    m_2 = len(blocks_2)
    m = len(blocks)
    logging.info(f"m_1 = {m_1}")
    logging.info(f"m_2 = {m_2}")
    logging.info(f"m = {m}")

    use_two_phase = bool(parameters.get("svm_two_phase", True)) and _is_svm_elastic_bipartite(problem)
    x_slice = slice(0, d)
    y_slice = slice(d, d + n)

    a, A = 0.0, 0.0
    x0 = np.zeros(problem.d, dtype=float) if x0 is None else x0

    x = x0.copy()
    x_prev = x0.copy()

    x_tilde_sum = np.zeros(problem.d)

    p = problem.operator_func.func_map(x0)
    p_prev = p.copy()
    F_store = p.copy()

    z = np.zeros(problem.d)
    z_prev = np.zeros(problem.d)
    q = np.zeros(problem.d)

    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()

    init_optmeasure = problem.func_value(x0)
    logresult(results, 0, 0.0, init_optmeasure)

    while not exitflag:
        np.copyto(x_prev, x)
        np.copyto(p_prev, p)
        np.copyto(z_prev, z)

        A_prev = A
        a_prev = a
        a = (1 + mu * A_prev) / (2 * L)
        A = A_prev + a

        if use_two_phase:
            # Phase 1: x
            p[x_slice] = F_store[x_slice]
            q[x_slice] = p[x_slice]
            z[x_slice] = z_prev[x_slice] + a * q[x_slice]
            x[x_slice] = problem.g_func.prox_opr_block(x_slice, x0[x_slice] - z[x_slice], A)
            F_store = problem.operator_func.func_map_block_update(
                F_store, x[x_slice], x_prev[x_slice], x_slice
            )

            # Phase 2: y
            p[y_slice] = F_store[y_slice]
            q[y_slice] = p[y_slice]
            z[y_slice] = z_prev[y_slice] + a * q[y_slice]
            x[y_slice] = problem.g_func.prox_opr_block(y_slice, x0[y_slice] - z[y_slice], A)
            F_store = problem.operator_func.func_map_block_update(
                F_store, x[y_slice], x_prev[y_slice], y_slice
            )

        else:
            for block in blocks:
                p_prev[block] = p[block]
                p[block] = F_store[block]
                q[block] = p[block]
                z[block] = z_prev[block] + a * q[block]
                x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A)
                F_store = problem.operator_func.func_map_block_update(F_store, x[block], x_prev[block], block)

        x_tilde_sum += a * x
        iteration += 1

        if iteration % exitcriterion.loggingfreq == 0:
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x)
            logging.info(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    return results, x


def pccm_normalized(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    """PCCM variant with the same per-block preconditioners used in ADUCA.

    This implementation also supports the SVM two-phase update.
    """

    d = problem.operator_func.d
    n = problem.operator_func.n
    L = parameters["L"]
    mu = parameters["mu"]

    block_size = parameters["block_size"]
    blocks_1 = construct_block_range(begin=0, end=d, block_size=block_size)
    block_size_2 = parameters["block_size_2"]
    blocks_2 = construct_block_range(begin=d, end=d + n, block_size=block_size_2)
    blocks = blocks_1 + blocks_2

    m_1 = len(blocks_1)
    m_2 = len(blocks_2)
    m = len(blocks)
    logging.info(f"m_1 = {m_1}")
    logging.info(f"m_2 = {m_2}")
    logging.info(f"m = {m}")

    normalizers_1, normalizers_2 = _compute_normalizers(problem, blocks_1, blocks_2)
    norm_x = _concat_normalizers(normalizers_1)  # length d
    norm_y = _concat_normalizers(normalizers_2)  # length n

    use_two_phase = bool(parameters.get("svm_two_phase", True)) and _is_svm_elastic_bipartite(problem)
    x_slice = slice(0, d)
    y_slice = slice(d, d + n)

    a, A = 0.0, 0.0
    x0 = np.zeros(problem.d, dtype=float) if x0 is None else x0

    x = x0.copy()
    x_prev = x0.copy()

    x_tilde_sum = np.zeros(problem.d)

    p = problem.operator_func.func_map(x0)
    p_prev = p.copy()
    F_store = p.copy()

    z = np.zeros(problem.d)
    z_prev = np.zeros(problem.d)
    q = np.zeros(problem.d)

    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()

    init_optmeasure = problem.func_value(x0)
    logresult(results, 0, 0.0, init_optmeasure)

    while not exitflag:
        np.copyto(x_prev, x)
        np.copyto(p_prev, p)
        np.copyto(z_prev, z)

        A_prev = A
        a_prev = a
        a = (1 + mu * A_prev) / (2 * L)
        A = A_prev + a

        if use_two_phase:
            # Phase 1: x
            p[x_slice] = F_store[x_slice]
            q[x_slice] = p[x_slice]
            z[x_slice] = z_prev[x_slice] + a * norm_x * q[x_slice]
            x[x_slice] = problem.g_func.prox_opr_block(x_slice, x0[x_slice] - z[x_slice], A * norm_x)
            F_store = problem.operator_func.func_map_block_update(
                F_store, x[x_slice], x_prev[x_slice], x_slice
            )

            # Phase 2: y
            p[y_slice] = F_store[y_slice]
            q[y_slice] = p[y_slice]
            z[y_slice] = z_prev[y_slice] + a * norm_y * q[y_slice]
            x[y_slice] = problem.g_func.prox_opr_block(y_slice, x0[y_slice] - z[y_slice], A * norm_y)
            F_store = problem.operator_func.func_map_block_update(
                F_store, x[y_slice], x_prev[y_slice], y_slice
            )

        else:
            for idx, block in enumerate(blocks):
                p_prev[block] = p[block]
                p[block] = F_store[block]
                q[block] = p[block]

                if idx < m_1:
                    norm_vec = normalizers_1[idx]
                    z[block] = z_prev[block] + a * norm_vec * q[block]
                    x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A * norm_vec)
                else:
                    norm_vec = normalizers_2[idx - m_1]
                    z[block] = z_prev[block] + a * norm_vec * q[block]
                    x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A * norm_vec)

                F_store = problem.operator_func.func_map_block_update(F_store, x[block], x_prev[block], block)

        x_tilde_sum += a * x
        iteration += 1

        if iteration % exitcriterion.loggingfreq == 0:
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x)
            logging.info(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    return results, x
