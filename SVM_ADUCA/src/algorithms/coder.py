import numpy as np
import time
import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range

def coder(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
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

    # x_temp = np.copy(x)
    # F_store_temp = np.copy(F_store)

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
            q[block] = p[block] + (a_prev / a) * (F_x_prev[block] - p_prev[block])

            # Step 8
            z[block] = z_prev[block] + a * q[block]

            # Step 9
            # x[j] = problem.g_func.prox_opr_coordinate(j + 1, x0[j] - z[j], A)
            x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A)

            # F_store = problem.operator_func.func_map_coordinate_update(F_store, x[j], x_prev[j], j)
            F_store = problem.operator_func.func_map_block_update(F_store, x[block], x_prev[block], block)
            # print(f"!!! x[j]: {x[j]}")
            # print(f"!!! x_temp[j]: {x_temp[j]}")
            # if j < m-1:
            #     # print(f"!!! F_store[j+1]: {F_store[j+1]}")
            #     # print(f"!!! F_store_temp[j+1]: {F_store_temp[j+1]}")
            #     if (not np.array_equal(x, x_temp)) or (not np.array_equal(F_store, F_store_temp)):
            #         if not np.array_equal(x, x_temp):
            #             print("YES!!!")
            #         if not np.array_equal(F_store, F_store_temp):
            #             print("BAD!!!!")
            #             diff_indices = np.where(F_store != F_store_temp)
            #             print(diff_indices)
            #             print(f"!!! F_store[13]: {F_store[13]}")
            #             print(f"!!! F_store_temp[13]: {F_store_temp[13]}")
            #         print(f"!!! j: {j}")
            #         exit()
            # else:
            #     # print(f"equal? F_store and F_store_temp: {np.array_equal(F_store, F_store_temp)}")
            #     # print(f"equal? x and x_temp: {np.array_equal(x, x_temp)}")
            #     if not np.array_equal(F_store, F_store_temp) or not np.array_equal(x, x_temp):
            #         exit()

        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x_tilde)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x




def coder_linesearch(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # Initialize parameters and variables
    d = problem.operator_func.d
    n = problem.operator_func.n
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
    L = 0.01
    L_ = 0.01
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
        np.copyto(x_prev, x)
        np.copyto(p_prev, p)
        np.copyto(z_prev, z)
        F_x_prev = np.copy(F_store)
        A_prev = A
        a_prev = a
        L_ = L

        # Step 5
        L = L_ / 2

        # Step 6
        while{True}:
            # Step 7
            L = 2 * L

            temp_x = np.copy(x)
            temp_p = np.copy(p)
            temp_p_prev = np.copy(p_prev)
            temp_F_store = np.copy(F_store)

            # Step 8
            a = (1 + mu * A_prev) / (2 * L)
            A = A_prev + a

            # Step 9
            for block in blocks:
                # Step 10
                temp_p_prev[block] = temp_p[block]
                temp_p[block] = F_store[block]

                # Step 11
                q[block] = temp_p[block] + (a_prev / a) * (F_x_prev[block] - temp_p_prev[block])

                # Step 12
                z[block] = z_prev[block] + a * q[block]

                # Step 13
                temp_x[block] = problem.g_func.prox_opr_block(block, x0[block] - z[block], A)
                problem.operator_func.func_map_block_update(temp_F_store, temp_x[block], x_prev[block], block)
                
            # Step 15
            norm_F_p = np.linalg.norm(temp_F_store - temp_p)
            norm_x = np.linalg.norm(temp_x - x_prev)
            if norm_F_p <= L * norm_x:
                x = np.copy(temp_x)
                p = np.copy(temp_p)
                p_prev = np.copy(temp_p_prev)
                F_store = np.copy(temp_F_store)
                break

        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            print(f"!!! L: {L}")
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x_tilde)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x