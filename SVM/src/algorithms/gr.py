import numpy as np
import time
import logging
import math
from src.algorithms.utils.results import Results, logresult
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.algorithms.utils.helper import construct_block_range


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
            opt_measure = problem.func_value(x_hat)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x
