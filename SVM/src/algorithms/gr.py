import numpy as np
import time
import logging
import math
from src.algorithms.utils.results import Results, logresult
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.algorithms.utils.helper import construct_block_range


def _compute_normalizers(problem, blocks_1, blocks_2):
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


def gr(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, x_0=None):
    # Init of Golden-Ratio
    d = problem.operator_func.d
    n = problem.operator_func.n
    block_size = parameters['block_size']
    blocks_1 = construct_block_range(begin=0, end=d, block_size=block_size)
    block_size_2 = parameters['block_size_2']
    blocks_2 = construct_block_range(begin=d, end = d+n, block_size=block_size_2)
    blocks = blocks_1 + blocks_2
    m_1 = len(blocks_1)
    m_2 = len(blocks_2)
    m = len(blocks)
    logging.info(f"m_1 = {m_1}")
    logging.info(f"m_2 = {m_2}")
    logging.info(f"m = {m}")

    beta = parameters["beta"]
    rho = beta + beta**2
    if x_0 is None:
        x_0 = np.zeros(problem.d)
    x_1 = np.full(shape=problem.d, fill_value=-0.0001)

    x = np.copy(x_1)
    x_ = np.copy(x_0)
    v = np.copy(x_1)
    v_ = np.copy(x_1)


    a = 1
    a_ = 1
    A = 1

    x_hat = a * x

    F = problem.operator_func.func_map(x)
    F_ = problem.operator_func.func_map(x_)

    # Stepsize selection function
    def gr_stepsize(a , a_, x, x_, F, F_ ):
        step_1 = rho * a
 
        F_norm = np.linalg.norm(F - F_)
        if F_norm == 0 :
            return step_1
        
        x_norm = np.linalg.norm(x - x_)
        L = F_norm / x_norm

        step_2 = 1 / ((4 * beta**2 * a_) * L**2 )

        step = min(step_1, step_2)
        # print(f"!!! step: {step}")
        return step, L


    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        step, L = gr_stepsize(a, a_, x, x_, F, F_)
        a_ = a
        a = step
        A += a

        v = (1-beta) * x + beta * v_

        v_ = np.copy(v)
        x_ = np.copy(x)

        x = problem.g_func.prox_opr(v - a * F, a, d)
 
        F_ = np.copy(F)
        # F = problem.operator_func.func_map(x)
        problem.operator_func.func_map_block_update(F, x, x_, range(d+n))

        x_hat = (A - a)/A * x_hat + a/A * x

        iteration += m
        if iteration % ( m *  exit_criterion.loggingfreq) == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(x)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x


def gr_normalized(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, x_0=None):
    # Init of Golden-Ratio with ADUCA-style preconditioners
    d = problem.operator_func.d
    n = problem.operator_func.n
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

    normalizers_1, normalizers_2 = _compute_normalizers(problem, blocks_1, blocks_2)

    beta = parameters["beta"]
    rho = beta + beta**2
    if x_0 is None:
        x_0 = np.zeros(problem.d)
    x_1 = np.full(shape=problem.d, fill_value=-0.0001)

    x = np.copy(x_1)
    x_ = np.copy(x_0)
    v = np.copy(x_1)
    v_ = np.copy(x_1)

    a = 1
    a_ = 1
    A = 1

    x_hat = a * x

    F = problem.operator_func.func_map(x)
    F_ = problem.operator_func.func_map(x_)

    # Stepsize selection function
    def gr_stepsize(a, a_, x, x_, F, F_):
        step_1 = rho * a
 
        F_norm = np.linalg.norm(F - F_)
        if F_norm == 0:
            return step_1
        
        x_norm = np.linalg.norm(x - x_)
        L = F_norm / x_norm

        step_2 = 1 / ((4 * beta**2 * a_) * L**2)

        step = min(step_1, step_2)
        return step, L

    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0)
    logresult(results, 1, 0.0, init_opt_measure)

    # Flattened normalizers for vectorized updates
    normalizer_x = np.concatenate(normalizers_1) if normalizers_1 else np.array([], dtype=float)
    normalizer_y = np.concatenate(normalizers_2) if normalizers_2 else np.array([], dtype=float)

    while not exit_flag:
        step, L = gr_stepsize(a, a_, x, x_, F, F_)
        a_ = a
        a = step
        A += a

        v = (1 - beta) * x + beta * v_

        v_ = np.copy(v)
        x_prev = np.copy(x)
        x_ = np.copy(x)

        F_prev = np.copy(F)

        # Vectorized prox with preconditioning:
        #   x-part: elastic-net prox with tau scaled by normalizer_x
        #   y-part: hinge dual clamp with preconditioned gradient
        z_x = v[:d] - a * normalizer_x * F_prev[:d]
        tau_x = a * normalizer_x
        p1 = tau_x * problem.g_func.lambda1
        p2 = 1.0 / (1.0 + tau_x * problem.g_func.lambda2)
        x_new_x = p2 * np.sign(z_x) * np.maximum(0, np.abs(z_x) - p1)

        z_y = v[d:] - a * normalizer_y * F_prev[d:]
        x_new_y = np.minimum(0, np.maximum(-1, z_y))

        x = np.concatenate((x_new_x, x_new_y))

        # Single block update to refresh F using the full delta
        F = problem.operator_func.func_map_block_update(F, x, x_prev, slice(0, d + n))
        F_ = F_prev

        x_hat = (A - a) / A * x_hat + a / A * x

        iteration += m
        if iteration % (m * exit_criterion.loggingfreq) == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(x_hat)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x
