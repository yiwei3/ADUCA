import numpy as np
import time
import numpy as np
import time
import logging
import math
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range

def pccm(problem, exitcriterion, parameters, x0=None):
    # Initialize parameters and variables
    d = problem.operator_func.d
    n = problem.operator_func.n
    L = parameters["L"]
    mu = parameters["mu"]
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

    a, A = 0, 0
    x0 = np.zeros(problem.d) if x0 is None else x0
    x, x_prev = x0.copy(), x0.copy()
    x_tilde_sum = np.zeros(problem.d)
    x_tilde = x0.copy()

    p = problem.operator_func.func_map(x0)
    p_prev = p.copy()
    F_store = np.copy(p)

    z, z_prev, q = np.zeros(problem.d), np.zeros(problem.d), np.zeros(problem.d)

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.func_value(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        x_prev = np.copy(x)
        p_prev = np.copy(p)
        z_prev = np.copy(z)

        # Update steps
        A_prev = A
        a_prev = a
        a = (1 + mu * A_prev) / (2 * L)
        A = A_prev + a

        F_x_prev = np.copy(F_store)
        for block in blocks:
            # Step 6
            p_prev[block] = p[block]
            p[block] = F_store[block]

            # Step 7
            q[block] = p[block]

            # Step 8
            z[block] = z_prev[block] + a * q[block]

            # Step 9
            x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A)

            F_store = problem.operator_func.func_map_block_update(F_store, x[block], x_prev[block], block)

        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x