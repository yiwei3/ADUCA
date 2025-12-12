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
from src.algorithms.aduca import aduca
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
    "a9a": (123, 32561),
    "gisette_scale.bz2": (5000, 6000),
    "rcv1_train.binary.bz2": (47236, 20242),
    "w8a": (300, 49749),
    "real-sim":(20958, 72309),
    "epsilon_normalized.t.bz2": (2000,100000)
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
    parser.add_argument('--mu', type=float, default=0.0, help='Mu')
    parser.add_argument('--beta', type = float, help='aduca constant parameter')
    parser.add_argument('--gamma', type = float, help='aduca constant parameter')
    parser.add_argument('--rho', type = float, default=0.0, help='aduca constant parameter')
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
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"timestamp = {timestamp}")
    logging.info("Completed initialization")
    traj_dir = os.path.join(outputdir, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    outputfilename = os.path.join(
        traj_dir,
        f"{dataset}-beta-{args.beta}-{algorithm}-blocksize-{args.block_size}-{args.block_size_2}-time-{timestamp}.json",
    )
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

    elif algorithm == "ADUCA":
        beta = args.beta
        block_size = args.block_size
        block_size_2 = args.block_size_2
        logging.info("Running ADUCA...")
        param = {
            "beta": beta,
            "gamma": args.gamma,
            "rho": args.rho,
            "mu": args.mu,
            "block_size": block_size,
            "block_size_2": block_size_2,
        }
        output, output_x = aduca(problem, exitcriterion, param)

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
