import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range


def _compute_normalizers(problem: GMVIProblem, beta_param: float) -> np.ndarray:
    """Normalizer used by ADUCA; reused for the other algorithms."""
    op_L = problem.operator_func.L
    beta_vec = problem.operator_func.beta
    return np.power(1 / op_L, 1 / beta_vec) / beta_param

def coder(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # Initialize parameters and variables
    n = problem.operator_func.n
    L = parameters["L"]
    block_size = parameters['block_size']
    blocks = construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    a, A = 0, 0
    x0 = np.ones(n) if x0 is None else x0
    x, x_prev = x0.copy(), x0.copy()
    Q = np.sum(x)
    p = problem.operator_func.p(Q)
    p_ = p
    dp = problem.operator_func.dp(Q)
    dp_ = dp
    x_tilde = x0.copy()
    x_tilde_sum = np.zeros(n)

    F_tilde = problem.operator_func.func_map(x0)
    F_tilde_prev = F_tilde.copy()
    F_store = np.copy(F_tilde)

    z, z_prev, F_bar = np.zeros(n), np.zeros(n), np.zeros(n)

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.residual(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        F_tilde_prev = np.copy(F_tilde)
        z_prev = np.copy(z)

        # Update steps
        A_prev = A
        a_prev = a
        a = 1 / (2 * L)
        A = A_prev + a

        F_x_prev = np.copy(F_store)

        for idx, block in enumerate(blocks):

            # Step 6
            F_tilde_prev[block] = F_tilde[block]
            F_tilde[block] = F_store[block]

            # Step 7
            F_bar[block] = F_tilde[block] + (a_prev / a) * (F_x_prev[block] - F_tilde_prev[block])

            # Step 8
            z[block] = z_prev[block] + a * F_bar[block]

            # Step 9
            x_prev[block] = x[block]
            x[block] = problem.g_func.prox_opr_block(x0[block] - z[block])

            Q += np.sum(x[block] - x_prev[block])
            p_ = p
            p = problem.operator_func.p(Q)
            dp_ = dp
            dp = problem.operator_func.dp(Q)
            problem.operator_func.func_map_block_update(F_store, x, p, p_, dp, dp_, block)
            
        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.residual(x)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x


def coder_normalized(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # Initialize parameters and variables with ADUCA-style normalizers
    n = problem.operator_func.n
    L = parameters["L"]
    beta_param = parameters.get("beta")
    if beta_param is None:
        raise ValueError("Parameter 'beta' is required for CODER_normalized.")

    block_size = parameters["block_size"]
    blocks = construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    normalizers = _compute_normalizers(problem, beta_param)

    a, A = 0, 0
    x0 = np.ones(n) if x0 is None else x0
    x, x_prev = x0.copy(), x0.copy()
    Q = np.sum(x)
    p = problem.operator_func.p(Q)
    p_ = p
    dp = problem.operator_func.dp(Q)
    dp_ = dp
    x_tilde = x0.copy()
    x_tilde_sum = np.zeros(n)

    F_tilde = problem.operator_func.func_map(x0)
    F_tilde_prev = F_tilde.copy()
    F_store = np.copy(F_tilde)

    z, z_prev, F_bar = np.zeros(n), np.zeros(n), np.zeros(n)

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.residual(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        F_tilde_prev = np.copy(F_tilde)
        z_prev = np.copy(z)

        # Update steps
        A_prev = A
        a_prev = a
        a = 1 / (2 * L)
        A = A_prev + a

        F_x_prev = np.copy(F_store)

        for idx, block in enumerate(blocks):

            # Step 6
            F_tilde_prev[block] = F_tilde[block]
            F_tilde[block] = F_store[block]

            # Step 7
            F_bar[block] = F_tilde[block] + (a_prev / a) * (F_x_prev[block] - F_tilde_prev[block])

            # Step 8
            z[block] = z_prev[block] + a * normalizers[block] * F_bar[block]

            # Step 9
            x_prev[block] = x[block]
            x[block] = problem.g_func.prox_opr_block(x0[block] - z[block])

            Q += np.sum(x[block] - x_prev[block])
            p_ = p
            p = problem.operator_func.p(Q)
            dp_ = dp
            dp = problem.operator_func.dp(Q)
            problem.operator_func.func_map_block_update(F_store, x, p, p_, dp, dp_, block)
            
        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.residual(x)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x




def coder_linesearch(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # Initialize parameters and variables
    n = problem.operator_func.n
    L = 1e-7
    block_size = parameters['block_size']
    blocks = construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    a, A = 0, 0
    x0 = np.ones(n) if x0 is None else x0
    x, x_prev = x0.copy(), x0.copy()
    Q = np.sum(x)
    p = problem.operator_func.p(Q)
    p_ = p
    dp = problem.operator_func.dp(Q)
    dp_ = dp
    x_tilde = x0.copy()
    x_tilde_sum = np.zeros(n)

    F_tilde = problem.operator_func.func_map(x0)
    F_tilde_prev = F_tilde.copy()
    F_store = np.copy(F_tilde)

    z, z_prev, F_bar = np.zeros(n), np.zeros(n), np.zeros(n)

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.residual(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        np.copyto(x_prev, x)
        np.copyto(F_tilde_prev, F_tilde)
        np.copyto(z_prev, z)
        F_x_prev = np.copy(F_store)
        Q = np.sum(x)
        A_prev = A
        a_prev = a
        L_ = L

        # Step 5
        L = L_ / 2

        # Step 6
        while True:
            # Step 7
            L = 2 * L
            temp_x = np.copy(x)
            temp_F_tilde = np.copy(F_tilde)
            temp_F_tilde_prev = np.copy(F_tilde_prev)
            temp_F_store = np.copy(F_store)
            temp_Q = Q
            temp_p = p
            temp_dp = dp

            # Step 8
            a = 1 / (2 * L)
            if a < 0.000001:
                break
            A = A_prev + a

            # Step 9
            for idx, block in enumerate(blocks):
                # Step 10
                temp_F_tilde_prev[block] = temp_F_tilde[block]
                temp_F_tilde[block] = F_store[block]

                # Step 11
                F_bar[block] = temp_F_tilde[block] + (a_prev / a) * (F_x_prev[block] - temp_F_tilde_prev[block])

                # Step 12
                z[block] = z_prev[block] + a * F_bar[block]

                # Step 13
                temp_x[block] = problem.g_func.prox_opr_block(x0[block] - z[block])

                temp_Q += np.sum(temp_x[block] - x_prev[block])
                p_ = temp_p
                temp_p = problem.operator_func.p(temp_Q)
                dp_ = temp_dp
                temp_dp = problem.operator_func.dp(temp_Q)
                problem.operator_func.func_map_block_update(temp_F_store, temp_x, temp_p, p_, temp_dp, dp_, block)
                
                
            # Step 15
            norm_F_p = np.linalg.norm(temp_F_store - temp_F_tilde)
            norm_x = np.linalg.norm(temp_x - x_prev)
            if norm_F_p <= L * norm_x:
                x = np.copy(temp_x)
                F_tilde = np.copy(temp_F_tilde)
                F_tilde_prev = np.copy(temp_F_tilde_prev)
                F_store = np.copy(temp_F_store)
                Q = temp_Q
                p = temp_p
                dp = temp_dp
                break

        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            print(f"!!! L: {L}")
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.residual(x)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x


def coder_linesearch_normalized(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # CODER with backtracking line-search and ADUCA-style normalizers
    n = problem.operator_func.n
    L = 1e-7
    beta_param = parameters.get("beta")
    if beta_param is None:
        raise ValueError("Parameter 'beta' is required for CODER_linesearch_normalized.")

    block_size = parameters['block_size']
    blocks = construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    normalizers = _compute_normalizers(problem, beta_param)

    a, A = 0, 0
    x0 = np.ones(n) if x0 is None else x0
    x, x_prev = x0.copy(), x0.copy()
    Q = np.sum(x)
    p = problem.operator_func.p(Q)
    p_ = p
    dp = problem.operator_func.dp(Q)
    dp_ = dp
    x_tilde = x0.copy()
    x_tilde_sum = np.zeros(n)

    F_tilde = problem.operator_func.func_map(x0)
    F_tilde_prev = F_tilde.copy()
    F_store = np.copy(F_tilde)

    z, z_prev, F_bar = np.zeros(n), np.zeros(n), np.zeros(n)

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.residual(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        np.copyto(x_prev, x)
        np.copyto(F_tilde_prev, F_tilde)
        np.copyto(z_prev, z)
        F_x_prev = np.copy(F_store)
        Q = np.sum(x)
        A_prev = A
        a_prev = a
        L_ = L

        # Step 5
        L = L_ / 2

        # Step 6
        while True:
            # Step 7
            L = 2 * L
            temp_x = np.copy(x)
            temp_F_tilde = np.copy(F_tilde)
            temp_F_tilde_prev = np.copy(F_tilde_prev)
            temp_F_store = np.copy(F_store)
            temp_Q = Q
            temp_p = p
            temp_dp = dp

            # Step 8
            a = 1 / (2 * L)
            if a < 0.000001:
                break
            A = A_prev + a

            # Step 9
            for idx, block in enumerate(blocks):
                # Step 10
                temp_F_tilde_prev[block] = temp_F_tilde[block]
                temp_F_tilde[block] = F_store[block]

                # Step 11
                F_bar[block] = temp_F_tilde[block] + (a_prev / a) * (F_x_prev[block] - temp_F_tilde_prev[block])

                # Step 12
                z[block] = z_prev[block] + a * normalizers[block] * F_bar[block]

                # Step 13
                temp_x[block] = problem.g_func.prox_opr_block(x0[block] - z[block])

                temp_Q += np.sum(temp_x[block] - x_prev[block])
                p_ = temp_p
                temp_p = problem.operator_func.p(temp_Q)
                dp_ = temp_dp
                temp_dp = problem.operator_func.dp(temp_Q)
                problem.operator_func.func_map_block_update(temp_F_store, temp_x, temp_p, p_, temp_dp, dp_, block)
                
            # Step 15
            norm_F_p = np.linalg.norm((temp_F_store - temp_F_tilde) * np.sqrt(normalizers))
            norm_x = np.linalg.norm((temp_x - x_prev) * np.sqrt(normalizers))
            if norm_F_p <= L * norm_x:
                x = np.copy(temp_x)
                F_tilde = np.copy(temp_F_tilde)
                F_tilde_prev = np.copy(temp_F_tilde_prev)
                F_store = np.copy(temp_F_store)
                Q = temp_Q
                p = temp_p
                dp = temp_dp
                break

        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            print(f"!!! L: {L}")
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.residual(x)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x
