import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range

""" 
Aduca rescaled for Nash Equilibrium problem. Stepsize rule follows the SVM variant (without mu > 0 case).
"""
def aduca(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of ADUCA.
    n = problem.operator_func.n
    op_L = problem.operator_func.L
    op_gamma = problem.operator_func.gamma
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    rho = parameters["rho"]

    rho_0 = min(rho, beta * (1 + beta) * (1 - gamma))
    eta = ((gamma * (1 + beta)) / (1 + beta**2)) ** 0.5
    tau = (3 * rho_0**2 * (1 + rho * beta) / (2 * (rho * beta) ** 2 + 3 * rho_0**2 * (1 + rho * beta)))
    C = eta / (2 * beta**0.5) * (tau**0.5 * rho * beta) / (3**0.5 * (1 + rho * beta) ** 0.5)
    C_hat = eta / (2 * beta**0.5) * ((1 - tau) * rho * beta) ** 0.5 / 2**0.5
    logging.info(f"rho = {rho_0}")
    logging.info(f"C = {C}")
    logging.info(f"C_hat = {C_hat}")

    block_size = parameters['block_size']
    blocks= construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    # normalizers
    time_start_initialization = time.time()

    normalizers = np.power(np.copy(1 / op_L), np.copy(1 / problem.operator_func.beta)) * 1 / beta
    normalizers_recip = np.where(normalizers != 0, 1 / normalizers, 0)

    # normalizers = np.ones(n)
    # normalizers_recip = np.ones(n)

    time_end_initialization = time.time()
    logging.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

    a = 0
    a_ = 0
    A = 0

    if u_0 is None:
        u_0 = np.ones(n)
    u_ = np.copy(u_0)
    u_hat = np.zeros(n)
    v = np.zeros(n)
    v_ = np.zeros(n)

    F = np.zeros(n)
    F_ = np.zeros(n)
    F_tilde = np.zeros(n)
    F_tilde_ = np.zeros(n)
    F_bar = np.copy(F_tilde)


    k = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.residual(u_)
    logresult(results, 1, 0.0, init_opt_measure)

    F_0 = problem.operator_func.func_map(u_0)
    F_tilde_0 = np.copy(F_0)
    F_tilde_1 = np.copy(F_tilde_0)

    ###
    """ 
    Stepsize selection function
    """
    def aduca_stepsize(normalizer, normalizer_recip, u, u_, a, a_, F, F_, F_tilde):
        # Stepsize option 1
        step_1 = rho_0 * a 

        # Stepsize option 2
        u_diff = np.copy(u - u_)
        F_diff = np.copy(F-F_)
        L_k = np.sqrt(np.inner(F_diff, (normalizer * F_diff)) / (np.inner(u_diff, (normalizer_recip * u_diff))))
        # den = max(np.inner(u_diff, normalizer_recip * u_diff), 1e-24)
        # num = max(np.inner(F_diff, normalizer * F_diff), 1e-24)
        # L_k = np.sqrt(num / den)

        if L_k == 0:
            step_2 = np.inf
        else:
            step_2 = C / L_k * (a / a_)**0.5 

        # Stepsize option 3
        F_tilde_diff = np.copy(F-F_tilde)
        L_hat_k = np.sqrt(np.inner(F_tilde_diff, (normalizer * F_tilde_diff)) / (np.inner(u_diff, (normalizer_recip * u_diff))))
        if L_hat_k == 0:
            step_3 = np.inf
        else:    
            step_3 = (C_hat / L_hat_k) * (a / a_)**0.5 

        # Stepsize selection
        step = min(step_1, step_2, step_3)
        return step, L_k , L_hat_k
    ###


    #line-search for the first step
    alpha = 2**0.5
    i = -1
    # First Loop
    while True:
        i += 1
        F_store = np.copy(F_0)
        a_0 = alpha**(-i)

        u = np.copy(u_0)
        Q = np.sum(u)
        p = problem.operator_func.p(Q)
        p_ = p
        dp = problem.operator_func.dp(Q)
        dp_ = dp

        u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0)

        for block in blocks:
            u[block] = u_1[block]
            F_tilde_1[block] = F_store[block]

            Q += np.sum(u[block] - u_0[block])
            p_ = p
            p = problem.operator_func.p(Q)
            dp_ = dp
            dp = problem.operator_func.dp(Q)
            problem.operator_func.func_map_block_update(F_store, u, p, p_, dp, dp_, block)
        
        F_1 = np.copy(F_store)
        norm_F = np.linalg.norm((F_1 - F_0))
        norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
        norm_u = np.linalg.norm((u_1 - u_0))
        if (a_0 * norm_F_tilde <= C_hat * norm_u) and (a_0 * norm_F <= C * norm_u):
            break
    # Second Loop
    while True:
        if (a_0 * norm_F_tilde >= C_hat / alpha * norm_u) or (a_0 * norm_F >= C / alpha * norm_u):
            break
        else:
            a_0 = a_0 * alpha
            F_store = np.copy(F_0)
            u = np.copy(u_0)
            Q = np.sum(u)
            p = problem.operator_func.p(Q)
            p_ = p
            dp = problem.operator_func.dp(Q)
            dp_ = dp

            u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0)

            for block in blocks:
                u[block] = u_1[block]
                F_tilde_1[block] = F_store[block]
                Q += np.sum(u[block] - u_0[block])
                p_ = p
                p = problem.operator_func.p(Q)
                dp_ = dp
                dp = problem.operator_func.dp(Q)
                problem.operator_func.func_map_block_update(F_store, u, p, p_, dp, dp_, block)
            
            F_1 = np.copy(F_store)
            norm_F = np.linalg.norm((F_1 - F_0))
            norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
            norm_u = np.linalg.norm((u_1 - u_0))

    a_ = a_0
    a = a_0
    A = 0

    u = np.copy(u_1)
    u_ = np.copy(u_0)
    v_ = np.copy(u_)
    u_hat = A * u_

    Q = np.sum(u)
    p = problem.operator_func.p(Q)
    p_ = p
    dp = problem.operator_func.dp(Q)
    dp_ = dp

    F = np.copy(F_1)
    F_ = np.copy(F_0)
    F_tilde = np.copy(F_tilde_1)
    F_tilde_ = np.copy(F_tilde_0)
    F_bar = np.zeros(n)

    while not exit_flag:
        # Step 6
        step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_, F, F_, F_tilde)
        a_ = a
        a = step
        A += a

        for block in blocks:
            # Step 8
            F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
            
            # Step 9
            v[block] = (1-beta) * u[block] + beta * v_[block]

            # Step 10
            u_[block] = u[block]
            u[block] = problem.g_func.prox_opr_block(v[block] - a * normalizers[block] * F_bar[block])

            # Step 11
            F_tilde_[block] = F_tilde[block]
            F_tilde[block] = F_store[block]
            Q += np.sum(u[block] - u_[block])
            p_ = p
            p = problem.operator_func.p(Q)
            dp_ = dp
            dp = problem.operator_func.dp(Q)
            problem.operator_func.func_map_block_update(F_store, u, p, p_, dp, dp_, block)
            
        # if k % (m * 50) == 0:
        #     F_store = problem.operator_func.func_map(u)        # fresh F(u)
        #     F_tilde = np.copy(F_store)
            
        np.copyto(F_, F)
        F = np.copy(F_store)
        np.copyto(v_, v)

        u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

        # Increment iteration counters
        k += m
        
        if k % (m *  exit_criterion.loggingfreq) == 0:
            # Compute averaged variables
            # step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_, F, F_, F_tilde)
            # a_ = a
            # a = step
            # A += a
            # u_hat = ((A - a) * u_hat / A) + (a*u / A)
            elapsed_time = time.time() - start_time
            opt_measure = problem.residual(u)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
            logresult(results, k, elapsed_time, opt_measure, L=L, L_hat=L_hat)
            exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)
            if exit_flag:
                break
            
    return results, u
