### SVM

import argparse
import datetime
import numpy as np
import numpy as np
import logging
import sys
import json
import os

from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.coder import coder, coder_linesearch
from src.algorithms.codervr import codervr, CODERVRParams
from src.algorithms.pccm import pccm
from src.algorithms.prcm import prcm
from src.algorithms.rapd import rapd
from src.algorithms.gr import gr
from src.algorithms.aduca import aduca_scale
from src.problems.utils.data_parsers import libsvm_parser
from src.problems.utils.data import Data
from src.problems.GMVI_func import GMVIProblem
from src.problems.operator_func.svmelastic_opr_func import SVMElasticOprFunc
from src.problems.g_func.svmelastic_g_func import SVMElasticGFunc
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulating BLAS.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

## (Dimension, Sample size)
DATASET_INFO = { 
    "sonar_scale": (60, 208),
    "a1a": (123, 1605),
    "a9a": (123, 32561),
    "gisette_scale": (5000, 6000),
    "news20": (1355191, 19996),
    "rcv1": (47236, 20242),
    "phishing": (68, 11055),
    "colon-cancer": (2000, 62),
    "madelon": (500, 2000),
    "mushrooms": (112, 8124),
    "skin_nonskin": (3, 245057),
    "SUSY": (18,5000000),
    "epsilon_normalized": (2000,400000),
    "HIGGS": (28, 11000000),
    "ijcnn1": (22, 49990),
    "w1a": (300, 2477),
    "w7a":(300, 24692),
    "w8a": (300, 49749),
    "covtype":(54, 581012),
    "real-sim":(20958, 72309),
    "cod-rna": (8, 59535),
    "cod-rna_t": (8, 271617),
    "cod-rna_r": (8, 157413),
    "epsilon_normalized_t": (2000,100000)
}

def parse_commandline():
    parser = argparse.ArgumentParser(description='Run optimization algorithms.')
    parser.add_argument('--outputdir', required=True, help='Output directory')
    parser.add_argument('--maxiter', required=True, type=int, help='Max iterations')
    parser.add_argument('--maxtime', required=True, type=int, help='Max execution time in seconds')
    parser.add_argument('--targetaccuracy', required=True, type=float, help='Target accuracy')
    parser.add_argument('--optval', type=float, default=0.0, help='Known optimal value')
    parser.add_argument('--loggingfreq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--dataset', required=True, help='Choice of dataset')
    parser.add_argument('--lossfn', default='SVM', help='Choice of loss function')
    parser.add_argument('--lambda1', type=float, default=0.0, help='Elastic net lambda 1')
    parser.add_argument('--lambda2', type=float, default=0.0, help='Elastic net lambda 2')
    parser.add_argument('--algo', required=True, help='Algorithm to run')
    parser.add_argument('--lipschitz', required=True, type=float, help='Lipschitz constant')
    parser.add_argument('--M', required=False, type=float, help='Lipschitz constant for codervr')
    parser.add_argument('--mu', type=float, default=0.0, help='Mu')
    parser.add_argument('--K', type=int, default=0, help='Variance reduction K')
    parser.add_argument('--beta', type = float, help='aduca constant parameter')
    parser.add_argument('--xi', type = float, help='aduca constant parameter')
    parser.add_argument('--phi_1', type = float, default=0.0, help='aduca constant parameter')
    parser.add_argument('--restartfreq', type = int, default=float('inf'), help='aduca_restart constant parameter')
    parser.add_argument('--block_size', type = int, default=1, help='block_size parameter >= 1, <= n')
    parser.add_argument('--block_size_2', type = int, default=float('inf'), help='block_size parameter >= 1, <= n')

    return parser.parse_args()

def main():
    # Run setup
    args = parse_commandline()
    outputdir = args.outputdir
    algorithm = args.algo
    # Problem Setup
    dataset = args.dataset
    filepath = f"../data/{dataset}"
    lambda1 = args.lambda1
    lambda2 = args.lambda2

    if dataset not in DATASET_INFO:
        raise ValueError("Invalid dataset name supplied.")

    d, n = DATASET_INFO[dataset]
    logging.info(f"dataset: {dataset}, d: {d}, n: {n}")
    logging.info(f"elasticnet_λ₁ = {lambda1}; elasticnet_λ₂ = {lambda2}")
    logging.info("--------------------------------------------------")
    
    # Exit criterion
    maxiter = args.maxiter
    maxtime = args.maxtime
    targetaccuracy = args.targetaccuracy + args.optval
    loggingfreq = args.loggingfreq
    exitcriterion = ExitCriterion(maxiter, maxtime, targetaccuracy, loggingfreq)

    # Runing
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    logging.info(f"timestamp = {timestamp}")
    logging.info("Completed initialization")
    outputfilename = f"{outputdir}/{dataset}-{lambda1}_{lambda2}-{algorithm}-{args.lipschitz}-output-{timestamp}.json"
    logging.info(f"outputfilename = {outputfilename}")
    logging.info("--------------------------------------------------")

    # Problem instance instantiation
    data = libsvm_parser(filepath, n, d)
    F = SVMElasticOprFunc(data)
    g = SVMElasticGFunc(d, n, lambda1, lambda2)
    problem = GMVIProblem(F, g)

    if algorithm == "CODER":
        logging.info("Running CODER...")
        L = args.lipschitz
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        coder_params = {"L": L, "mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = coder(problem, exitcriterion, coder_params)
    elif algorithm == "CODER_linesearch":
        logging.info("Running CODER_linesearch...")
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        coder_params = {"mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = coder_linesearch(problem, exitcriterion, coder_params)

    elif algorithm == "CODERVR":
        logging.info("Running CODERVR...")
        L = args.lipschitz
        mu = args.mu
        M = args.M
        K = args.K
        block_size = args.block_size
        codervr_params = {"L": L, "mu": mu, "M": M, "K": K, "block_size": block_size}
        output, output_x = codervr(problem, exitcriterion, codervr_params)
        print(f"Output saved to {outputfilename}")

    elif algorithm == "RAPD":
        logging.info("Running RAPD...")
        L = args.lipschitz
        x0 = np.zeros(d)
        y0 = np.zeros(n)
        epochs = int(1e12)
        # print(f"data.values.shape: {data.values.shape}")
        # print(f"data.features.shape: {data.features.shape}")
        B = np.array(data.values).reshape(-1, 1) * np.array(data.features)
        output, output_x = rapd(B, x0, y0, L, epochs, lambda1, lambda2, exitcriterion)

    elif algorithm == "PCCM":
        logging.info("Running PCCM...")
        L = args.lipschitz
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        pccm_params = {"L": L, "mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = pccm(problem, exitcriterion, pccm_params)

    elif algorithm == "PRCM":
        logging.info("Running PRCM...")
        L = args.lipschitz
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        prcm_params = {"L": L, "mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = prcm(problem, exitcriterion, prcm_params)
    elif algorithm == "GR":
        beta = args.beta
        block_size = args.block_size
        block_size_2 = args.block_size_2
        logging.info("Running Golden Ratio...")
        param = {"beta": beta, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = gr(problem, exitcriterion, param)
    # elif algorithm == "ADUCA_restart":
    #     beta = args.beta
    #     c = args.c
    #     restartfreq = args.restartfreq
    #     block_size = args.block_size
    #     logging.info("Running ADUCA_restart...")
    #     param = {"beta": beta, "c": c, "restartfreq": restartfreq, "block_size": block_size}
    #     output, output_x = aduca_restart(problem, exitcriterion, param)
    # elif algorithm == "ADUCA":
    #     beta = args.beta
    #     c = args.c
    #     block_size = args.block_size
    #     logging.info("Running ADUCA...")
    #     param = {"beta": beta, "c": c, "block_size": block_size}
    #     output, output_x = aduca(problem, exitcriterion, param)
    # elif algorithm == "ADUCA_restart_scale":
    #     beta = args.beta
    #     xi = args.xi
    #     restartfreq = args.restartfreq
    #     block_size = args.block_size
    #     block_size_2 = args.block_size_2
    #     logging.info("Running ADUCA_restart_scale...")
    #     param = {"beta": beta, "xi": xi, "restartfreq": restartfreq, "block_size": block_size, "block_size_2": block_size_2}
    #     output, output_x = aduca_restart_scale(problem, exitcriterion, param)
    elif algorithm == "ADUCA_scale":
        beta = args.beta
        xi = args.xi
        mu = args.mu
        phi_1= args.phi_1
        block_size = args.block_size
        block_size_2 = args.block_size_2
        logging.info("Running ADUCA_scale...")
        param = {"beta": beta, "xi": xi, "mu": mu, "phi_1": phi_1, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = aduca_scale(problem, exitcriterion, param)

    else:
        print("Wrong algorithm name supplied")
    
    with open(outputfilename, 'w') as outfile:
        json.dump({"args": vars(args), 
                "output_x": output_x.tolist(),
                "iterations": output.iterations, 
                "times": output.times,
                "optmeasures": output.optmeasures,
                "L": output.L,
                "L_hat": output.L_hat}, 
                outfile)
        logging.info(f"output saved to {outputfilename}")

if __name__ == "__main__":
    main()