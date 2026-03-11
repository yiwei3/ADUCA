import logging
import time
from typing import Tuple

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
    """Return True if the problem looks like the SVMElastic structure.

    We use this to enable a *two-phase* (x-then-y) update that is mathematically
    equivalent to the original per-block Gauss–Seidel loop for this operator,
    because:

    - F_x depends only on y
    - F_y depends only on x

    Therefore all x-block updates within an epoch commute, and all y-block
    updates within an epoch commute.
    """
    F = getattr(problem, "operator_func", None)
    if F is None:
        return False

    # Heuristic structural checks: these are present in SVMElasticOprFunc.
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


def coder(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    """CODER for GMVI, with an SVM-specialized two-phase update.

    For the SVMElastic operator, the block-wise loop can be rearranged into:
      1) update all x-blocks in parallel (Jacobi within x)
      2) refresh F_y
      3) update all y-blocks in parallel (Jacobi within y)
      4) refresh F_x

    This yields the *same iterate* as the original block loop but typically runs
    faster because it reduces Python-level overhead and performs only two large
    operator updates per epoch.
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

    use_two_phase = bool(parameters.get("svm_two_phase", True)) and _is_svm_elastic_bipartite(problem)
    x_slice = slice(0, d)
    y_slice = slice(d, d + n)

    a, A = 0.0, 0.0
    x0 = np.zeros(problem.d) if x0 is None else x0

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

        # Operator snapshot used in the extrapolation term.
        F_x_prev = F_store.copy()

        if use_two_phase:
            # -------------------------
            # Phase 1: update x (0:d)
            # -------------------------
            p[x_slice] = F_store[x_slice]
            q[x_slice] = p[x_slice] + (a_prev / a) * (F_x_prev[x_slice] - p_prev[x_slice])
            z[x_slice] = z_prev[x_slice] + a * q[x_slice]
            x[x_slice] = problem.g_func.prox_opr_block(x_slice, x0[x_slice] - z[x_slice], A)
            F_store = problem.operator_func.func_map_block_update(
                F_store, x[x_slice], x_prev[x_slice], x_slice
            )

            # -------------------------
            # Phase 2: update y (d:d+n)
            # -------------------------
            p[y_slice] = F_store[y_slice]
            q[y_slice] = p[y_slice] + (a_prev / a) * (F_x_prev[y_slice] - p_prev[y_slice])
            z[y_slice] = z_prev[y_slice] + a * q[y_slice]
            x[y_slice] = problem.g_func.prox_opr_block(y_slice, x0[y_slice] - z[y_slice], A)
            F_store = problem.operator_func.func_map_block_update(
                F_store, x[y_slice], x_prev[y_slice], y_slice
            )

        else:
            # Generic Gauss–Seidel block update.
            for block in blocks:
                p_prev[block] = p[block]
                p[block] = F_store[block]
                q[block] = p[block] + (a_prev / a) * (F_x_prev[block] - p_prev[block])
                z[block] = z_prev[block] + a * q[block]
                x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A)
                F_store = problem.operator_func.func_map_block_update(F_store, x[block], x_prev[block], block)

        x_tilde_sum += a * x
        x_tilde = x_tilde_sum / A
        iteration += 1

        if iteration % exitcriterion.loggingfreq == 0:
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x)
            logging.info(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    return results, x


def coder_normalized(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    """CODER variant with the same per-block preconditioners used in ADUCA.

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
    x0 = np.zeros(problem.d) if x0 is None else x0

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

        F_x_prev = F_store.copy()

        if use_two_phase:
            # Phase 1: x
            p[x_slice] = F_store[x_slice]
            q[x_slice] = p[x_slice] + (a_prev / a) * (F_x_prev[x_slice] - p_prev[x_slice])
            z[x_slice] = z_prev[x_slice] + a * norm_x * q[x_slice]
            x[x_slice] = problem.g_func.prox_opr_block(x_slice, x0[x_slice] - z[x_slice], A * norm_x)
            F_store = problem.operator_func.func_map_block_update(
                F_store, x[x_slice], x_prev[x_slice], x_slice
            )

            # Phase 2: y
            p[y_slice] = F_store[y_slice]
            q[y_slice] = p[y_slice] + (a_prev / a) * (F_x_prev[y_slice] - p_prev[y_slice])
            z[y_slice] = z_prev[y_slice] + a * norm_y * q[y_slice]
            x[y_slice] = problem.g_func.prox_opr_block(y_slice, x0[y_slice] - z[y_slice], A * norm_y)
            F_store = problem.operator_func.func_map_block_update(
                F_store, x[y_slice], x_prev[y_slice], y_slice
            )

        else:
            for idx, block in enumerate(blocks):
                p_prev[block] = p[block]
                p[block] = F_store[block]
                q[block] = p[block] + (a_prev / a) * (F_x_prev[block] - p_prev[block])

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


def coder_linesearch(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    """CODER with backtracking line-search.

    This includes the same SVM two-phase acceleration used in `coder()`.
    """

    d = problem.operator_func.d
    n = problem.operator_func.n
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

    L = 1e-07
    L_ = 1e-07

    use_two_phase = bool(parameters.get("svm_two_phase", True)) and _is_svm_elastic_bipartite(problem)
    x_slice = slice(0, d)
    y_slice = slice(d, d + n)

    a, A = 0.0, 0.0

    x0 = np.zeros(problem.d) if x0 is None else x0
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

        F_x_prev = F_store.copy()
        A_prev = A
        a_prev = a
        L_ = L

        # Step 5
        L = L_ / 2

        # Step 6: line-search
        while True:
            # Step 7
            L = 2 * L

            temp_x = x.copy()
            temp_p = p.copy()
            temp_p_prev = p_prev.copy()
            temp_F_store = F_store.copy()

            # Step 8
            a = (1 + mu * A_prev) / (2 * L)
            A = A_prev + a

            # Step 9
            if use_two_phase:
                # Phase 1: x
                temp_p_prev[x_slice] = temp_p[x_slice]
                temp_p[x_slice] = temp_F_store[x_slice]
                q[x_slice] = temp_p[x_slice] + (a_prev / a) * (F_x_prev[x_slice] - temp_p_prev[x_slice])
                z[x_slice] = z_prev[x_slice] + a * q[x_slice]
                temp_x[x_slice] = problem.g_func.prox_opr_block(x_slice, x0[x_slice] - z[x_slice], A)
                problem.operator_func.func_map_block_update(
                    temp_F_store, temp_x[x_slice], x_prev[x_slice], x_slice
                )

                # Phase 2: y
                temp_p_prev[y_slice] = temp_p[y_slice]
                temp_p[y_slice] = temp_F_store[y_slice]
                q[y_slice] = temp_p[y_slice] + (a_prev / a) * (F_x_prev[y_slice] - temp_p_prev[y_slice])
                z[y_slice] = z_prev[y_slice] + a * q[y_slice]
                temp_x[y_slice] = problem.g_func.prox_opr_block(y_slice, x0[y_slice] - z[y_slice], A)
                problem.operator_func.func_map_block_update(
                    temp_F_store, temp_x[y_slice], x_prev[y_slice], y_slice
                )

            else:
                for block in blocks:
                    temp_p_prev[block] = temp_p[block]
                    temp_p[block] = temp_F_store[block]

                    q[block] = temp_p[block] + (a_prev / a) * (F_x_prev[block] - temp_p_prev[block])
                    z[block] = z_prev[block] + a * q[block]
                    temp_x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A)
                    problem.operator_func.func_map_block_update(temp_F_store, temp_x[block], x_prev[block], block)

            # Step 15
            norm_F_p = np.linalg.norm(temp_F_store - temp_p)
            norm_x = np.linalg.norm(temp_x - x_prev)
            iteration += 1

            if norm_F_p <= L * norm_x:
                x = temp_x
                p = temp_p
                p_prev = temp_p_prev
                F_store = temp_F_store
                break

        x_tilde_sum += a * x

        if iteration % exitcriterion.loggingfreq == 0:
            logging.info(f"L (linesearch) = {L}")
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x)
            logging.info(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    return results, x


def coder_linesearch_normalized(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    """CODER_linesearch with ADUCA-style preconditioners and SVM two-phase update."""

    d = problem.operator_func.d
    n = problem.operator_func.n
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

    L = 1e-07
    L_ = 1e-07

    # ADUCA-style diagonal preconditioners (per-coordinate step multipliers).
    #
    # IMPORTANT: keep these vectors separate from any scalar norms computed during
    # the line-search. Accidentally reusing names (e.g., `norm_x`) can silently
    # overwrite the preconditioner vector and destabilize the method.
    normalizers_1, normalizers_2 = _compute_normalizers(problem, blocks_1, blocks_2)
    precond_x = _concat_normalizers(normalizers_1)
    precond_y = _concat_normalizers(normalizers_2)

    precond = np.concatenate([precond_x, precond_y], axis=0)
    precond_recip = np.where(precond != 0, 1 / precond, 0)

    use_two_phase = bool(parameters.get("svm_two_phase", True)) and _is_svm_elastic_bipartite(problem)
    x_slice = slice(0, d)
    y_slice = slice(d, d + n)

    a, A = 0.0, 0.0

    x0 = np.zeros(problem.d) if x0 is None else x0
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

        F_x_prev = F_store.copy()
        A_prev = A
        a_prev = a
        L_ = L

        # Step 5
        L = L_ / 2

        # Step 6: line-search
        while True:
            # Step 7
            L = 2 * L

            temp_x = x.copy()
            temp_p = p.copy()
            temp_p_prev = p_prev.copy()
            temp_F_store = F_store.copy()

            # Step 8
            a = (1 + mu * A_prev) / (2 * L)
            A = A_prev + a

            # Step 9
            if use_two_phase:
                # Phase 1: x
                temp_p_prev[x_slice] = temp_p[x_slice]
                temp_p[x_slice] = temp_F_store[x_slice]
                q[x_slice] = temp_p[x_slice] + (a_prev / a) * (F_x_prev[x_slice] - temp_p_prev[x_slice])
                z[x_slice] = z_prev[x_slice] + a * precond_x * q[x_slice]
                temp_x[x_slice] = problem.g_func.prox_opr_block(
                    x_slice, x0[x_slice] - z[x_slice], A * precond_x
                )
                problem.operator_func.func_map_block_update(
                    temp_F_store, temp_x[x_slice], x_prev[x_slice], x_slice
                )

                # Phase 2: y
                temp_p_prev[y_slice] = temp_p[y_slice]
                temp_p[y_slice] = temp_F_store[y_slice]
                q[y_slice] = temp_p[y_slice] + (a_prev / a) * (F_x_prev[y_slice] - temp_p_prev[y_slice])
                z[y_slice] = z_prev[y_slice] + a * precond_y * q[y_slice]
                temp_x[y_slice] = problem.g_func.prox_opr_block(
                    y_slice, x0[y_slice] - z[y_slice], A * precond_y
                )
                problem.operator_func.func_map_block_update(
                    temp_F_store, temp_x[y_slice], x_prev[y_slice], y_slice
                )

            else:
                for idx, block in enumerate(blocks):
                    temp_p_prev[block] = temp_p[block]
                    temp_p[block] = temp_F_store[block]

                    q[block] = temp_p[block] + (a_prev / a) * (F_x_prev[block] - temp_p_prev[block])

                    if idx < m_1:
                        norm_vec = normalizers_1[idx]
                        z[block] = z_prev[block] + a * norm_vec * q[block]
                        temp_x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A * norm_vec)
                    else:
                        norm_vec = normalizers_2[idx - m_1]
                        z[block] = z_prev[block] + a * norm_vec * q[block]
                        temp_x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A * norm_vec)

                    problem.operator_func.func_map_block_update(temp_F_store, temp_x[block], x_prev[block], block)

            # Step 15
            # norm_F_p = np.inner(temp_F_store - temp_p, normalizer * (temp_F_store - temp_p)) ** 0.5
            # norm_x = np.inner(temp_x - x_prev, normalizer_recip * (temp_x - x_prev)) ** 0.5
            ls_norm_F_p = np.linalg.norm((temp_F_store - temp_p) * np.sqrt(precond))
            ls_norm_dx = np.linalg.norm((temp_x - x_prev) * np.sqrt(precond_recip))
            # norm_F_p = np.linalg.norm((temp_F_store - temp_p))
            # norm_x = np.linalg.norm((temp_x - x_prev))
            iteration += 1

            if ls_norm_F_p <= L * ls_norm_dx:
                x = temp_x
                p = temp_p
                p_prev = temp_p_prev
                F_store = temp_F_store
                break

        x_tilde_sum += a * x
        x_tilde = x_tilde_sum / A

        if iteration % exitcriterion.loggingfreq == 0:
            logging.info(f"L (linesearch) = {L}")
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x)
            logging.info(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    return results, x
