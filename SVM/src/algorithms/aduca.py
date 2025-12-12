import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

""" 
Aduca rescaled for SVM. Use simple stepsize rule (3.9) in the paper (without mu > 0 case).
"""
def aduca(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of ADUCA.
    d = problem.operator_func.d
    n = problem.operator_func.n
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    rho = parameters["rho"]
    eps = parameters.get("eps", 1e-8)

    rho_0 = min(rho, beta * (1+beta) * (1-gamma))
    eta = ( (gamma * (1+beta)) / (1 + beta**2))**0.5
    tau = (3 * rho_0**2 * (1+rho*beta) / (2 *(rho*beta)**2 + 3*rho_0**2*(1+rho*beta)))
    C = eta / (2*beta**0.5) * (tau**0.5*rho*beta) / (3**0.5 * (1+rho*beta)**0.5)
    C_hat = eta / (2*beta**0.5) * ((1-tau)*rho*beta)**0.5 / 2**0.5
    logging.info(f"rho = {rho_0}")
    logging.info(f"C = {C}")
    logging.info(f"C_hat = {C_hat}")
    
    # Scale the blocks with respect to different variables (x and y).
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
            if norm  != 0:
                normalizer[i-block.start] = 1 / norm
            else:
                normalizer[i-block.start] = 1
        normalizers_1.append(normalizer)
    
    normalizers_2 = []

    max_norm = 0

    for block in blocks_2:
        size = block.stop - block.start
        normalizer = np.zeros(size)
        for i in range(block.start, block.stop):
            norm = np.linalg.norm(b[i-d] * A_matrix[i-d])
            if norm > max_norm:
                max_norm = norm
            if norm  != 0:
                normalizer[i-block.start] = 1 / norm
            else:
                normalizer[i-block.start] = 1
        normalizers_2.append(normalizer)

    # # Compute Lipschitz constant estimates without GPU SVD (rcv1 is too wide for cusolver)
    # def power_iteration_lipschitz(A_mat, b_vec, num_iter=8):
    #     rng = np.random.default_rng(seed=0)
    #     v = rng.standard_normal(A_mat.shape[1])
    #     v_norm = np.linalg.norm(v)
    #     if v_norm == 0 or not np.isfinite(v_norm):
    #         return 0.0
    #     v /= v_norm

    #     for _ in range(num_iter):
    #         w = (b_vec * (A_mat @ v)) / n
    #         if not np.all(np.isfinite(w)):
    #             return 0.0
    #         v = (A_mat.T @ (b_vec * w)) / n
    #         v_norm = np.linalg.norm(v)
    #         if v_norm == 0 or not np.isfinite(v_norm):
    #             return 0.0
    #         v /= v_norm

    #     w = (b_vec * (A_mat @ v)) / n
    #     return float(np.linalg.norm(w))

    # try:
    #     sigma_max = power_iteration_lipschitz(A_matrix, b, num_iter=6)
    # except Exception as exc:
    #     logging.warning(f"Power iteration for Lipschitz estimate failed ({exc}); falling back to row-norm bound.")
    #     sigma_max = max_norm / np.sqrt(n)
    # norm_Q = sigma_max ** 2

    # logging.info(f"Estimated L: {max_norm / np.sqrt(n)}")
    # logging.info(f"Estimated L_hat: {norm_Q}")

    normalizers = normalizers_1 + normalizers_2
    normalizers = np.concatenate(normalizers, axis=0)
    normalizers_recip = np.where(normalizers != 0, 1 / normalizers, 0)

    time_end_initialization = time.time()
    logging.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

    a = 0
    a_ = 0
    A = 0

    if u_0 is None:
        # u_0 = np.full(shape=problem.d, fill_value=-0.0001)
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

        u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0, a_0 * normalizers[:d], d)

        for block in blocks:
            F_tilde_1[block] = F_store[block]
            F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
        
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
            u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0, a_0 * normalizers[:d], d)

            for block in blocks:
                F_tilde_1[block] = F_store[block]
                F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
            
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

    F = np.copy(F_1)
    F_ = np.copy(F_0)
    F_tilde = np.copy(F_tilde_1)
    F_tilde_ = np.copy(F_tilde_0)
    F_bar = np.zeros(problem.d)

    while not exit_flag:
        # Step 6
        step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_, F, F_, F_tilde)
        a_ = a
        a = step
        A += a

        for index, block in enumerate(blocks, start=0):
            # Step 8
            F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
            
            # Step 9
            v[block] = (1-beta) * u[block] + beta * v_[block]

            # Step 10
            u_[block] = u[block]
            if block.stop <= d:
                u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * normalizers_1[index] * F_bar[block], a * normalizers_1[index])
            else:
                u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * normalizers_2[index-m_1] * F_bar[block], a)

            # Step 11
            F_tilde_[block] = F_tilde[block]
            F_tilde[block] = F_store[block]
            F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block)

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
            opt_measure = problem.func_value(u)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
            logresult(results, k, elapsed_time, opt_measure, L=L, L_hat=L_hat)
            exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)
            if exit_flag:
                break
            
    return results, u

    
                
