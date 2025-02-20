import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range


### aduca tailored for SVM
def aduca_scale(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of adapCODER
    d = problem.operator_func.d
    n = problem.operator_func.n
    mu = parameters["mu"]
    beta = parameters["beta"]
    xi = parameters["xi"]
    logging.info(f"mu = {mu}")

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
    # print(f"!!! m: {m}")

    phi_2 = xi * beta * (1+beta)
    phi_3 = 4 / (7 * beta * (1+beta) * (1-xi) )
    phi_4 = (((1-xi) * (1 + beta))  /  7 * beta)**0.5 * 0.5
    phi_5 = 1 / (7 * beta)

    ### normalizers
    time_start_initialization = time.time()
    A_matrix = problem.operator_func.A
    A_matrix_T = A_matrix.T
    b = problem.operator_func.b

    # normalizers = []
    # normalizer_0 = np.linalg.norm(A_matrix.T @ b)
    # normalizers.append(normalizer_0)
    # # logging.info(f"normalizer_0 = {normalizer_0}")

    # normalizer_0 = np.zeros(shape=(d,d))
    # for i in range(d):
    #     if np.linalg.norm(b * A_matrix_T[i]) != 0:
    #     # print(np.linalg.norm(b * A_matrix_T[i]))
    #         normalizer_0[i,i] = 1 / np.linalg.norm(b * A_matrix_T[i])
    #     # print(normalizer_0[i,i])

    # for i in range(n):
    #     normalizer = np.linalg.norm(A_matrix[i] * b[i])
    #     normalizers.append(normalizer)
    #     # logging.info(f"normalizer_{i+1} = {normalizer}")
    # normalizers = np.asarray(normalizers)
    # # normal = np.sum(normalizers)
    # # logging.info(f"normalizer = {normal}")
    normalizers_1 = []
    for block in blocks_1:
        size = block.stop - block.start
        normalizer = np.zeros(shape=size)
        for i in block:
            norm = np.linalg.norm(b * A_matrix_T[i])
            if norm  != 0:
                normalizer[i-block.start] = 1 / norm
            else:
                normalizer[i-block.start] = 1
        normalizers_1.append(normalizer)
    
    normalizers_2 = []

    sum_norm = 0
    max_norm = 0

    for block in blocks_2:
        size = block.stop - block.start
        normalizer = np.zeros(size)
        for i in block:
            norm = np.linalg.norm(b[i-d] * A_matrix[i-d])
            sum_norm += norm
            if norm > max_norm:
                max_norm = norm
            if norm  != 0:
                normalizer[i-block.start] = 1 / norm
            else:
                normalizer[i-block.start] = 1
        normalizers_2.append(normalizer)
    print(f"!!! The L: {sum_norm / n}")
    print(f"!!! max_norm: {max_norm}")
    print(f"!!! The L_hat: {np.sqrt(max_norm**2 / n) }")
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

    if mu == 0:
        alpha = min(phi_2, phi_3)
        # Stepsize selection function
        def aduca_stepsize(normalizer, normalizer_recip, u, u_, a, a_, F, F_, F_tilde):
            step_1 = alpha * a 

            u_diff = np.copy(u - u_)

            ### we can heuristically scale the step
            # L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
            F_tilde_diff = np.copy(F-F_tilde)
            L_hat_k = np.sqrt(np.inner(F_tilde_diff, (normalizer * F_tilde_diff)) / np.inner(u_diff, (normalizer_recip * u_diff))) 
            if L_hat_k == 0:
                step_2 = 100000
            else:    
                step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    

            F_diff = np.copy(F-F_)
            L_k = np.sqrt(np.inner(F_diff, (normalizer * F_diff)) / np.inner(u_diff, (normalizer_recip * u_diff)))
            # print(f"!!! L_k: {L_k}")
            if L_k == 0:
                step_3 = 100000
            else:
                step_3 = (phi_3 ** 2) / (a_ * L_k**2)  
                # print(f"!!! step_3: {step_3}")
            
            step = min(step_1, step_2, step_3)
            # print(f" !!! Stepsize: {step}")
            return step, L_k , L_hat_k

        ## line-search for the first step
        a_0 = 10 * phi_1
        while True:
            F_store = np.copy(F_0)
            a_0 = a_0 / 2
            u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0, a_0 * normalizers[:d], d)

            for block in blocks:
                F_tilde_1[block] = F_store[block]
                F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
            
            F_1 = np.copy(F_store)
            norm_F = np.linalg.norm((F_1 - F_0))
            norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
            norm_u = np.linalg.norm((u_1 - u_0))

            # print(f"phi_2: {phi_2}")
            # print(f"a_0: {a_0}")
            if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
                break

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
                    # u[block] = problem.g_func.prox_opr_block(block ,v[block] -  a * F_bar[block], a)
                else:
                    u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * normalizers_2[index-m_1] * F_bar[block], a)
                    # u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * F_bar[block], a)

                # Step 11
                F_tilde_[block] = F_tilde[block]
                F_tilde[block] = F_store[block]
                F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block)

            np.copyto(F_, F)
            F = np.copy(F_store)
            # print(f"If F equal to F_tilde")
            np.copyto(v_, v)

            # print(f"!!! (a / A): {(a / A)}")
            u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

            # Increment iteration counters
            k += m
            
            if k % (m *  exit_criterion.loggingfreq) == 0:
                # Compute averaged variables
                # step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
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
    
    elif mu > 0:
        phi_1 = parameters["phi_1"]
        alpha = min(phi_1, phi_2, phi_3)
        print(f"!!! alpha: {alpha}")
        # Stepsize selection function
        def aduca_stepsize(normalizer, normalizer_recip, u, u_, a, a_, theta, theta_, F, F_, F_tilde):
            step_1_1 = alpha * a 
            step_1_3 = phi_3*(1+beta*mu*a)*theta_/theta * a
            if (1-xi*beta**2*mu*a) > 0:
                step_1_2 = phi_2 / (1-xi*beta**2*mu*a) * a
                step_1 = min(step_1_1, step_1_2, step_1_3)
            else:
                step_1 = min(step_1_1, step_1_3)

            u_diff = np.copy(u - u_)

            ### we can heuristically scale the step
            # L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
            F_tilde_diff = np.copy(F-F_tilde)
            L_hat_k = np.sqrt(np.inner(F_tilde_diff, (normalizer * F_tilde_diff)) / np.inner(u_diff, (normalizer_recip * u_diff))) 
            if L_hat_k == 0:
                step_2 = 100000
            else:    
                step_2 = (phi_4 / L_hat_k) * ((1+beta*mu*a_)*a / a_)**0.5 * (theta_ / theta)**0.5     

            F_diff = np.copy(F-F_)
            L_k = np.sqrt(np.inner(F_diff, (normalizer * F_diff)) / np.inner(u_diff, (normalizer_recip * u_diff)))
            # print(f"!!! L_k: {L_k}")
            if L_k == 0:
                step_3 = 100000
            else:
                step_3 = (phi_5**2 * (1+beta*mu*a_) * (1+beta*mu*a)) / (a_ * L_k**2)  * (theta_ / theta)
                # print(f"!!! step_3: {step_3}")
            
            step = min(step_1, step_2, step_3)
            # print(f" !!! Stepsize: {step}")
            if step < 0.000001:
                step = 0.000001
            return step, L_k , L_hat_k

        ## line-search for the first step
        a_0 = 10 * phi_1
        while True:
            F_store = np.copy(F_0)
            a_0 = a_0 / 2
            u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0, a_0 * normalizers[:d], d)

            for block in blocks:
                F_tilde_1[block] = F_store[block]
                F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
            
            F_1 = np.copy(F_store)
            norm_F = np.linalg.norm((F_1 - F_0))
            norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
            norm_u = np.linalg.norm((u_1 - u_0))

            # print(f"phi_2: {phi_2}")
            # print(f"a_0: {a_0}")
            if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
                break

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
        theta = 1
        theta_ = 1

        while not exit_flag:
            # Step 6
            theta_ = theta
            theta = (1+mu*a) / (1+mu*beta*phi_1*a) * theta

            step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_,theta,theta_, F, F_, F_tilde)
            a_ = a
            a = step
            A += a

            for index, block in enumerate(blocks, start=0):
                # Step 8
                F_bar[block] = F_tilde[block] + (theta_*a_ / (theta*a)) * (F_[block] - F_tilde_[block])
                
                # Step 9
                v[block] = (1-beta) * u[block] + beta * v_[block]

                # Step 10
                u_[block] = u[block]
                if block.stop <= d:
                    u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * normalizers_1[index] * F_bar[block], a * normalizers_1[index])
                    # u[block] = problem.g_func.prox_opr_block(block ,v[block] -  a * F_bar[block], a)
                else:
                    u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * normalizers_2[index-m_1] * F_bar[block], a)
                    # u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * F_bar[block], a)

                # Step 11
                F_tilde_[block] = F_tilde[block]
                F_tilde[block] = F_store[block]
                F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block)

            np.copyto(F_, F)
            F = np.copy(F_store)
            # print(f"If F equal to F_tilde")
            np.copyto(v_, v)

            # print(f"!!! (a / A): {(a / A)}")
            u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

            # Increment iteration counters
            k += m
            
            if k % (m *  exit_criterion.loggingfreq) == 0:
                # Compute averaged variables
                # step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
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


# ### aduca tailored for SVM
# def aduca_restart_scale(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
#     # Init of adapCODER
#     d = problem.operator_func.d
#     n = problem.operator_func.n
#     beta = parameters["beta"]
#     c = parameters["c"]
#     restartfreq = parameters["restartfreq"]

#     block_size = parameters['block_size']
#     blocks_1 = construct_block_range(begin=0, end=d, block_size=block_size)
#     block_size_2 = parameters['block_size_2']
#     blocks_2 = construct_block_range(begin=d, end = d+n, block_size=block_size_2)
#     blocks = blocks_1 + blocks_2
#     m_1 = len(blocks_1)
#     m_2 = len(blocks_2)
#     m = len(blocks)
#     logging.info(f"m_1 = {m_1}")
#     logging.info(f"m_2 = {m_2}")
#     logging.info(f"m = {m}")
#     # print(f"!!! m: {m}")

#     phi_2 = xi * beta * (1+beta)
#     phi_3 = 4 / (7 * beta * (1+beta) * (1-xi) )
#     phi_4 = (((1-xi) * (1 + beta))  /  7 * beta)**0.5 * 0.5
#     phi_5 = 1 / (7 * beta)

#     # Stepsize selection function
#     def aduca_stepsize(normalizer, normalizer_recip, u, u_, a, a_, F, F_, F_tilde):
#         step_1 = alpha * a 

#         u_diff = np.copy(u - u_)

#         ### we can heuristically scale the step
#         # L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
#         F_tilde_diff = np.copy(F-F_tilde)
#         L_hat_k = np.sqrt(np.inner(F_tilde_diff, (normalizer * F_tilde_diff)) / np.inner(u_diff, (normalizer_recip * u_diff))) 
#         if L_hat_k == 0:
#             step_2 = 1000
#         else:    
#             step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    

#         F_diff = np.copy(F-F_)
#         L_k = np.sqrt(np.inner(F_diff, (normalizer * F_diff)) / np.inner(u_diff, (normalizer_recip * u_diff)))
#         # print(f"!!! L_k: {L_k}")
#         if L_k == 0:
#             step_3 = 1000
#         else:
#             step_3 = (phi_3 ** 2) / (a_ * L_k**2)  
#             # print(f"!!! step_3: {step_3}")
        
#         step = min(step_1, step_2, step_3)
#         # print(f" !!! Stepsize: {step}")
#         return step, L_k , L_hat_k

#     ### normalizers
#     time_start_initialization = time.time()
#     A_matrix = problem.operator_func.A
#     A_matrix_T = A_matrix.T
#     b = problem.operator_func.b

#     normalizers_1 = []
#     for block in blocks_1:
#         size = block.stop - block.start
#         normalizer = np.zeros(shape=size)
#         for i in block:
#             norm = np.linalg.norm(b * A_matrix_T[i])
#             if norm  != 0:
#                 normalizer[i-block.start] = 1 / norm
#             else:
#                 normalizer[i-block.start] = 1
#         normalizers_1.append(normalizer)
    
#     normalizers_2 = []
#     # sum_norm = 0
#     # max_norm = 0
#     for block in blocks_2:
#         size = block.stop - block.start
#         normalizer = np.zeros(size)
#         for i in block:
#             norm = np.linalg.norm(b[i-d] * A_matrix[i-d])
#             # sum_norm += norm
#             # if norm > max_norm:
#             #     max_norm = norm
#             if norm  != 0:
#                 normalizer[i-block.start] = 1 / norm
#             else:
#                 normalizer[i-block.start] = 1
#         normalizers_2.append(normalizer)
#     # print(f"!!! The L: {sum_norm / n}")
#     # print(f"!!! max_norm: {max_norm}")
#     # print(f"!!! The L_hat: {np.sqrt(max_norm**2 / n) }")
#     # exit()
    
#     normalizers = normalizers_1 + normalizers_2
#     normalizers = np.concatenate(normalizers, axis=0)
#     normalizers_recip = np.where(normalizers != 0, 1 / normalizers, 0)
#     # normalizers = []
#     # normalizer_0 = np.linalg.norm(A_matrix.T @ b)
#     # normalizers.append(normalizer_0)
#     # # logging.info(f"normalizer_0 = {normalizer_0}")

#     # normalizer_0 = np.zeros(shape=(d,d))
#     # for i in range(d):
#     #     if np.linalg.norm(b * A_matrix_T[i]) != 0:
#     #     # print(np.linalg.norm(b * A_matrix_T[i]))
#     #         normalizer_0[i,i] = 1 / np.linalg.norm(b * A_matrix_T[i])
#     #     # print(normalizer_0[i,i])

#     # for i in range(n):
#     #     normalizer = np.linalg.norm(A_matrix[i] * b[i])
#     #     normalizers.append(normalizer)
#     #     # logging.info(f"normalizer_{i+1} = {normalizer}")
#     # normalizers = np.asarray(normalizers)
#     # # normal = np.sum(normalizers)
#     # # logging.info(f"normalizer = {normal}")
#     time_end_initialization = time.time()
#     logging.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

#     a = 0
#     a_ = 0
#     A = 0

#     if u_0 is None:
#         u_0 = np.zeros(problem.d)
#         # u_0 = np.zeros(problem.d)
#     u_ = np.copy(u_0)
#     u_hat = np.zeros(problem.d)
#     v = np.zeros(problem.d)
#     v_ = np.zeros(problem.d)

#     F = np.zeros(problem.d)
#     F_ = np.zeros(problem.d)
#     F_tilde = np.zeros(problem.d)
#     F_tilde_ = np.zeros(problem.d)
#     F_bar = np.copy(F_tilde)


#     outer_k = 0
#     exit_flag = False
#     start_time = time.time()
#     results = Results()
#     init_opt_measure = problem.func_value(u_)
#     logresult(results, 1, 0.0, init_opt_measure)

#     while not exit_flag:
#         u = np.copy(u_0)
#         u_ = np.copy(u_0)
    
#         F_0 = problem.operator_func.func_map(u_0)
#         F_tilde_0 = np.copy(F_0)
#         F_tilde_1 = np.copy(F_tilde_0)

#         ## line-search for the first step
#         a_0 = 10 * phi_1
#         while True:
#             F_store = np.copy(F_0)
#             a_0 = a_0 / 2
#             u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0, a_0 * normalizers[:d], d)

#             for block in blocks:
#                 F_tilde_1[block] = F_store[block]
#                 F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
            
#             F_1 = np.copy(F_store)
#             norm_F = np.linalg.norm((F_1 - F_0))
#             norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
#             norm_u = np.linalg.norm((u_1 - u_0))
#             # print(f"phi_2: {phi_2}")
#             # print(f"a_0: {a_0}")
#             if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
#                 break

#         a_ = a_0
#         a = a_0
#         A = 0

#         u = np.copy(u_1)
#         u_ = np.copy(u_0)
#         v_ = np.copy(u_)
#         u_hat = A * u_

#         F = np.copy(F_1)
#         F_ = np.copy(F_0)
#         F_tilde = np.copy(F_tilde_1)
#         F_tilde_ = np.copy(F_tilde_0)
#         F_bar = np.zeros(problem.d)
        

#         k = 0
#         restart_flag = False
#         while not exit_flag and not restart_flag:

#             # Step 6
#             step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_, F, F_, F_tilde)
#             a_ = a
#             a = step
#             A += a

#             for index, block in enumerate(blocks, start=0):
#                 # Step 8
#                 F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
                
#                 # Step 9
#                 v[block] = (1-beta) * u[block] + beta * v_[block]

#                 # Step 10
#                 u_[block] = u[block]
#                 if block.stop <= d:
#                     u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * normalizers_1[index] * F_bar[block], a * normalizers_1[index])
#                     # u[block] = problem.g_func.prox_opr_block(block ,v[block] -  a * F_bar[block], a)
#                 else:
#                     u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * normalizers_2[index-m_1] * F_bar[block], a)
#                     # u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * F_bar[block], a)

#                 # Step 11
#                 F_tilde_[block] = F_tilde[block]
#                 F_tilde[block] = F_store[block]
#                 F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block)

#             np.copyto(F_, F)
#             F = np.copy(F_store)
#             # print(f"If F equal to F_tilde")
#             np.copyto(v_, v)

#             # print(f"!!! (a / A): {(a / A)}")
#             u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

#             # Increment iteration counters
#             outer_k += m
#             k += m
            
#             if outer_k % (m *  exit_criterion.loggingfreq) == 0:
#                 # Compute averaged variables
#                 # step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
#                 # a_ = a
#                 # a = step
#                 # A += a      
#                 # u_hat = ((A - a) * u_hat / A) + (a*u / A)
#                 elapsed_time = time.time() - start_time
#                 opt_measure = problem.func_value(u_hat)
#                 logging.info(f"elapsed_time: {elapsed_time}, iteration: {outer_k}, opt_measure: {opt_measure}")
#                 logresult(results, outer_k, elapsed_time, opt_measure, L=L, L_hat=L_hat)
#                 exit_flag = CheckExitCondition(exit_criterion, outer_k, elapsed_time, opt_measure)
#                 if exit_flag:
#                     break
            
#             if (k >= restartfreq):
#                 elapsed_time = time.time() - start_time
#                 opt_measure = problem.func_value(u_hat)
#                 logging.info("<===== RESTARTING")
#                 logging.info(f"k: {k}")
#                 logging.info(f"elapsed_time: {elapsed_time}, iteration: {outer_k}, opt_measure: {opt_measure}")
#                 # Compute averaged variables
#                 step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip,u,u_,a,a_,F,F_,F_tilde)
#                 a_ = a
#                 a = step
#                 A += a      
#                 u_hat = ((A - a) * u_hat / A) + (a*u / A)
#                 u_0 = np.copy(u_hat)
#                 # Update x0 and y0 for restart
#                 init_opt_measure = opt_measure
#                 restart_flag = True
#                 break
            
#     return results, u




# def aduca(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
#     # Init of adapCODER
#     d = problem.operator_func.d
#     n = problem.operator_func.n
#     beta = parameters["beta"]
#     c = parameters["c"]

#     block_size = parameters['block_size']
#     blocks = construct_block_range(begin=0, end=d+n, block_size=block_size)
#     m = len(blocks)

#     phi_1 = 2 * c * beta * (1+beta)
#     phi_2 = (((1-2*c) * (1 + beta))  /  7 * beta)**0.5 * 0.5
#     phi_3 = 1 / (7 * beta)

#     alpha = min(phi_1, 4 / (7 * beta * (1+beta) * (1-2 * c) ))

#     def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
#         step_1 = alpha * a 

#         ### we can heuristically scale the step
#         L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
#         if L_hat_k == 0:
#             step_2 = 100
#         else:    
#             step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    

#         L_k = np.linalg.norm(F - F_) / (np.linalg.norm(u - u_)) 
#         # print(f"!!! L_k: {L_k}")
#         if L_k == 0:
#             step_3 = 100
#         else:
#             step_3 = (phi_3 ** 2) / (a_ * L_k**2)  
#             # print(f"!!! step_3: {step_3}")
        
#         step = min(step_1, step_2, step_3)
#         # print(f" !!! Stepsize: {step}")
#         return step
    
#     # time_start_initialization = time.time()

#     # time_end_initialization = time.time()
#     # logging.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

#     a = 0
#     a_ = 0
#     A = 0

#     u = np.zeros(problem.d)
#     u_ = np.copy(u)
#     # u_new = np.copy(u)
#     u_hat = np.copy(u)
#     v = np.zeros(problem.d)
#     v_ = np.zeros(problem.d)

#     F = np.zeros(problem.d)
#     F_ = np.zeros(problem.d)
#     F_tilde = np.zeros(problem.d)
#     F_tilde_ = np.zeros(problem.d)
#     F_bar = np.copy(F_tilde)

#     if u_0 is None:
#         u_0 = np.full(shape=problem.d, fill_value=-0.0001)
#         # u_0 = np.zeros(problem.d)
#     u_1 = u_0
    
#     F_0 = problem.operator_func.func_map(u_0)
#     F_tilde_0 = np.copy(F_0)
#     F_tilde_1 = np.copy(F_tilde_0)

#     ## line-search for the first step
#     a_0 = 10 * phi_1
#     while True:
#         F_store = np.copy(F_0)
#         a_0 = a_0 / 2
#         u_1 = problem.g_func.prox_opr(u_0 - a_0 * F_0, a_0, d)

#         for block in blocks:
#             F_tilde_1[block] = F_store[block]
#             F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
        
#         F_1 = np.copy(F_store)
#         norm_F = np.linalg.norm((F_1 - F_0))
#         norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
#         norm_u = np.linalg.norm((u_1 - u_0))

#         # print(f"phi_2: {phi_2}")
#         # print(f"a_0: {a_0}")
#         if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
#             break

#     a_ = a_0
#     a = a_0
#     A = 0

#     u = np.copy(u_1)
#     u_ = np.copy(u_0)
#     v_ = np.copy(u_)
#     u_hat = A * u_

#     F = np.copy(F_1)
#     F_ = np.copy(F_0)
#     F_tilde = np.copy(F_tilde_1)
#     F_tilde_ = np.copy(F_tilde_0)
#     F_bar = np.zeros(problem.d)

#     # Run init
#     iteration = 0
#     exit_flag = False
#     start_time = time.time()
#     results = Results()
#     init_opt_measure = problem.func_value(u_)
#     logresult(results, 1, 0.0, init_opt_measure)

#     while not exit_flag:

#         # Step 6
#         step = aduca_stepsize(u, u_, a, a_, F, F_, F_tilde)
#         a_ = a
#         a = step
#         A += a

#         # Step 7
#         for block in blocks:

#             # Step 8
#             F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
            
#             # Step 9
#             v[block] = (1-beta) * u[block] + beta * v_[block]

#             # Step 10
#             u_[block] = u[block]
#             u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * F_bar[block], a) ## Rescaling

#             # Step 11
#             F_tilde_[block] = F_tilde[block]
#             F_tilde[block] = F_store[block]
#             F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block) 

#         np.copyto(F_, F)
#         F = np.copy(F_store)
#         # print(f"If F equal to F_tilde")
#         np.copyto(v_, v)

#         # print(f"!!! (a / A): {(a / A)}")
#         u_hat = ((A - a) * u_hat / A) + (a*u_ / A)


#         iteration += m
#         if iteration % (m *  exit_criterion.loggingfreq) == 0:
#             step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
#             a_ = a
#             a = step
#             A += a      
#             u_hat = ((A - a) * u_hat / A) + (a*u / A)
            
#             elapsed_time = time.time() - start_time
#             opt_measure = problem.func_value(u_hat)
#             logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
#             logresult(results, iteration, elapsed_time, opt_measure)
#             exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

#     return results, u






# ### aduca with restart
# def aduca_restart(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
#     # Init of adapCODER
#     d = problem.operator_func.d
#     n = problem.operator_func.n
#     beta = parameters["beta"]
#     c = parameters["c"]
#     restartfreq = parameters["restartfreq"]

#     block_size = parameters['block_size']
#     blocks = construct_block_range(begin=0,end=d+n, block_size=block_size)
#     m= len(blocks)

#     phi_1 = 2 * c * beta * (1+beta)
#     phi_2 = (((1-2*c) * (1 + beta))  /  7 * beta)**0.5 * 0.5
#     phi_3 = 1 / (7 * beta)
#     alpha = min(phi_1, 4 / (7 * beta * (1+beta) * (1-2 * c) ))
#     # Stepsize selection function
#     def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
#         step_1 = alpha * a 

#         ### we can heuristically scale the step
#         L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
#         if L_hat_k == 0:
#             step_2 = 1000
#         else:    
#             step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    

#         L_k = np.linalg.norm(F - F_) / (np.linalg.norm(u - u_)) 
#         # print(f"!!! L_k: {L_k}")
#         if L_k == 0:
#             step_3 = 1000
#         else:
#             step_3 = (phi_3 ** 2) / (a_ * L_k**2)  
#             # print(f"!!! step_3: {step_3}")
        
#         step = min(step_1, step_2, step_3)
#         # print(f" !!! Stepsize: {step}")
#         return step

#     # ### normalizers
#     # time_start_initialization = time.time()
#     # A_matrix = problem.operator_func.A
#     # b = problem.operator_func.b
#     # normalizers = []
#     # normalizer_1 = np.linalg.norm(A_matrix.T @ b)
#     # normalizers.append(normalizer_1)
#     # for i in range(n):
#     #     normalizer = A[i] * b[i]
#     #     normalizers.append(normalizer)
#     # time_end_initialization = time.time()
#     # logging.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

#     a = 0
#     a_ = 0
#     A = 0

#     if u_0 is None:
#         u_0 = np.full(shape=problem.d, fill_value=-0.0001)
#         # u_0 = np.zeros(problem.d)
#     u_ = np.copy(u_0)
#     u_hat = np.zeros(problem.d)
#     v = np.zeros(problem.d)
#     v_ = np.zeros(problem.d)

#     F = np.zeros(problem.d)
#     F_ = np.zeros(problem.d)
#     F_tilde = np.zeros(problem.d)
#     F_tilde_ = np.zeros(problem.d)
#     F_bar = np.copy(F_tilde)


#     outer_k = 0
#     exit_flag = False
#     start_time = time.time()
#     results = Results()
#     init_opt_measure = problem.func_value(u_)
#     logresult(results, 1, 0.0, init_opt_measure)

#     while not exit_flag:
#         u = np.copy(u_0)
#         u_ = np.copy(u_0)
    
#         F_0 = problem.operator_func.func_map(u_0)
#         F_tilde_0 = np.copy(F_0)
#         F_tilde_1 = np.copy(F_tilde_0)

#         ## line-search for the first step
#         a_0 = 2 * phi_1
#         while True:
#             F_store = np.copy(F_0)
#             a_0 = a_0 / 2
#             u_1 = problem.g_func.prox_opr(u_0 - a_0 * F_0, a_0, d)

#             for block in blocks:
#                 F_tilde_1[block] = F_store[block]
#                 F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
            
#             F_1 = np.copy(F_store)
#             norm_F = np.linalg.norm((F_1 - F_0))
#             norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
#             norm_u = np.linalg.norm((u_1 - u_0))

#             # print(f"phi_2: {phi_2}")
#             # print(f"a_0: {a_0}")
#             if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
#                 break

#         a_ = a_0
#         a = a_0
#         A = 0

#         u = np.copy(u_1)
#         u_ = np.copy(u_0)
#         v_ = np.copy(u_)
#         u_hat = A * u_

#         F = np.copy(F_1)
#         F_ = np.copy(F_0)
#         F_tilde = np.copy(F_tilde_1)
#         F_tilde_ = np.copy(F_tilde_0)
#         F_bar = np.zeros(problem.d)
        

#         k = 0
#         restart_flag = False
#         while not exit_flag and not restart_flag:

#             # Step 6
#             step = aduca_stepsize(u, u_, a, a_, F, F_, F_tilde)
#             a_ = a
#             a = step
#             A += a

#             for block in blocks:
#                 # Step 8
#                 F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
                
#                 # Step 9
#                 v[block] = (1-beta) * u[block] + beta * v_[block]

#                 # Step 10
#                 u_[block] = u[block]
#                 u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * F_bar[block], a)

#                 # Step 11
#                 F_tilde_[block] = F_tilde[block]
#                 F_tilde[block] = F_store[block]
#                 F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block) 
#             np.copyto(F_, F)
#             F = np.copy(F_store)
#             # print(f"If F equal to F_tilde")
#             np.copyto(v_, v)

#             # print(f"!!! (a / A): {(a / A)}")
#             u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

#             # Increment iteration counters
#             outer_k += m
#             k += m
            
#             if outer_k % (m *  exit_criterion.loggingfreq) == 0:
#                 # Compute averaged variables
#                 # step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
#                 # a_ = a
#                 # a = step
#                 # A += a      
#                 # u_hat = ((A - a) * u_hat / A) + (a*u / A)
#                 elapsed_time = time.time() - start_time
#                 opt_measure = problem.func_value(u_hat)
#                 logging.info(f"elapsed_time: {elapsed_time}, iteration: {outer_k}, opt_measure: {opt_measure}")
#                 logresult(results, outer_k, elapsed_time, opt_measure)
#                 exit_flag = CheckExitCondition(exit_criterion, outer_k, elapsed_time, opt_measure)
#                 if exit_flag:
#                     break
            
#             if (k >= restartfreq):
#                 elapsed_time = time.time() - start_time
#                 opt_measure = problem.func_value(u_hat)
#                 logging.info("<===== RESTARTING")
#                 logging.info(f"k: {k}")
#                 logging.info(f"elapsed_time: {elapsed_time}, iteration: {outer_k}, opt_measure: {opt_measure}")
#                 # Compute averaged variables
#                 step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
#                 a_ = a
#                 a = step
#                 A += a      
#                 u_hat = ((A - a) * u_hat / A) + (a*u / A)
#                 u_0 = np.copy(u_hat)
#                 # Update x0 and y0 for restart
#                 init_opt_measure = opt_measure
#                 restart_flag = True
#                 break
            
#     return results, u



