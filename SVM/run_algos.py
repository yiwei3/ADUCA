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
from src.algorithms.coder import (
    coder,
    coder_linesearch,
    coder_normalized,
    coder_linesearch_normalized,
)
from src.algorithms.codervr import codervr, CODERVRParams
from src.algorithms.pccm import pccm, pccm_normalized
from src.algorithms.prcm import prcm
from src.algorithms.rapd import rapd
from src.algorithms.gr import gr, gr_normalized
from src.algorithms.gr_torch import gr_torch, gr_torch_normalized
from src.algorithms.aduca import aduca
from src.algorithms.aduca_torch_dist import aduca_distributed
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
    "real-sim": (20958, 72309),
    "epsilon_normalized.t.bz2": (2000, 100000),
    "news20.binary.bz2": (1355191, 19996),
    "SUSY": (18, 5000000),
    "HIGGS": (28, 11000000),
    "ijcnn1": (22, 49990),
    "cod-rna": (8, 59535),
    "phishing": (68, 11055),
    "covtype.binary": (54, 581012),
}

DATASET_FILES = {
    "a9a": "a9a",
    "gisette_scale.bz2": "gisette_scale.bz2",
    "rcv1_train.binary.bz2": "rcv1_train.binary.bz2",
    "w8a": "w8a",
    "real-sim": "real-sim",
    "epsilon_normalized.t.bz2": "epsilon_normalized.t.bz2",
    "news20.binary.bz2": "news20.binary.bz2",
    "SUSY": "SUSY.xz",
    "HIGGS": "HIGGS.xz",
    "ijcnn1": "ijcnn1.bz2",
    "cod-rna": "cod-rna",
    "phishing": "phishing",
    "covtype.binary": "covtype.libsvm.binary.bz2",
}

def _coerce_optional_bool(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError(f"Cannot interpret boolean value from {value!r}.")

def parse_commandline():
    parser = argparse.ArgumentParser(description='Run optimization algorithms.')
    parser.add_argument('--outputdir', required=True, help='Output directory')
    parser.add_argument('--maxiter', required=True, type=int, help='Max iterations')
    parser.add_argument('--maxtime', required=True, type=int, help='Max execution time in seconds')
    parser.add_argument('--targetaccuracy', required=True, type=float, help='Target accuracy')
    parser.add_argument('--optval', type=float, default=0.0, help='Known optimal value')
    parser.add_argument('--loggingfreq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--dataset', required=True, choices=sorted(DATASET_INFO), help='Choice of dataset')
    parser.add_argument('--lossfn', default='SVM', help='Choice of loss function')
    parser.add_argument('--lambda1', type=float, default=0.0, help='Elastic net lambda 1')
    parser.add_argument('--lambda2', type=float, default=0.0, help='Elastic net lambda 2')
    parser.add_argument('--algo', required=True, help='Algorithm to run')
    parser.add_argument('--lipschitz', type=float, default=None, help='Lipschitz constant')
    parser.add_argument('--mu', type=float, default=0.0, help='Mu')
    parser.add_argument('--beta', type = float, help='aduca constant parameter')
    parser.add_argument('--gamma', type = float, help='aduca constant parameter')
    parser.add_argument('--rho', type = float, default=0.0, help='aduca constant parameter')
    parser.add_argument('--block_size', type = int, default=1, help='block_size parameter >= 1, <= n')
    parser.add_argument('--block_size_2', type = int, default=float('inf'), help='block_size parameter >= 1, <= n')
    parser.add_argument('--device', default=None, help='Torch device (e.g. cuda:0)')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float64'], help='Torch dtype')
    parser.add_argument('--use_dense', nargs='?', const='true', default=None, help='Force dense GEMV for torch/distributed backends (true/false)')
    parser.add_argument('--dense_threshold', type=float, default=0.25, help='Auto-dense switch threshold')
    parser.add_argument('--backend', default=None, choices=['numpy', 'torch_dist'], help='ADUCA backend override')
    parser.add_argument('--dist_backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='torch.distributed backend')
    parser.add_argument('--strong-convexity', '--strong_convexity', dest='strong_convexity',
                        action='store_true', help='Enable strong convexity ratio_bar update for distributed ADUCA')

    return parser.parse_args() 

def main():
    # Run setup
    args = parse_commandline()
    outputdir = args.outputdir
    os.makedirs(outputdir, exist_ok=True)
    algorithm = args.algo
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend_arg = None if args.backend is None else str(args.backend).lower()
    use_dense = _coerce_optional_bool(args.use_dense)
    use_dist_aduca = (
        algorithm in {"ADUCA_TORCH_DIST", "ADUCA_DISTRIBUTED"}
        or (algorithm == "ADUCA" and backend_arg == "torch_dist")
    )
    output_algorithm = "ADUCA_TORCH_DIST" if use_dist_aduca else algorithm
    # Problem Setup
    dataset = args.dataset
    filepath = f"../data/{DATASET_FILES[dataset]}"
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
    if use_dist_aduca:
        logging.info(f"Detected rank {rank} / world_size {world_size}")
    logging.info("Completed initialization")
    outputfilename = os.path.join(
        outputdir,
        f"{dataset}-beta-{args.beta}-mu-{args.mu}-{output_algorithm}-blocksize-{args.block_size}-{args.block_size_2}-time-{timestamp}.json",
    )
    logging.info(f"outputfilename = {outputfilename}")
    logging.info("--------------------------------------------------")

    # Problem instance instantiation
    data = libsvm_parser(filepath, n, d)
    F = SVMElasticOprFunc(data)
    g = SVMElasticGFunc(d, n, lambda1, lambda2)
    problem = GMVIProblem(F, g)

    def require_lipschitz():
        if args.lipschitz is None:
            raise ValueError(f"--lipschitz is required for algorithm {algorithm}.")
        return args.lipschitz

    if algorithm == "CODER":
        logging.info("Running CODER...")
        L = require_lipschitz()
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        coder_params = {"L": L, "mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = coder(problem, exitcriterion, coder_params)

    elif algorithm == "CODER_normalized":
        logging.info("Running CODER_normalized...")
        L = require_lipschitz()
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        coder_params = {"L": L, "mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = coder_normalized(problem, exitcriterion, coder_params)

    elif algorithm == "CODER_linesearch":
        logging.info("Running CODER_linesearch...")
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        coder_params = {"mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = coder_linesearch(problem, exitcriterion, coder_params)

    elif algorithm == "CODER_linesearch_normalized":
        logging.info("Running CODER_linesearch_normalized...")
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        coder_params = {"mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = coder_linesearch_normalized(problem, exitcriterion, coder_params)

    elif algorithm == "PCCM":
        logging.info("Running PCCM...")
        L = require_lipschitz()
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        pccm_params = {"L": L, "mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = pccm(problem, exitcriterion, pccm_params)

    elif algorithm == "PCCM_normalized":
        logging.info("Running PCCM_normalized...")
        L = require_lipschitz()
        mu = args.mu
        block_size = args.block_size
        block_size_2 = args.block_size_2
        pccm_params = {"L": L, "mu": mu, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = pccm_normalized(problem, exitcriterion, pccm_params)

    elif algorithm == "PRCM":
        logging.info("Running PRCM...")
        L = require_lipschitz()
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

    elif algorithm == "GR_normalized":
        beta = args.beta
        block_size = args.block_size
        block_size_2 = args.block_size_2
        logging.info("Running Golden Ratio (normalized)...")
        param = {"beta": beta, "block_size": block_size, "block_size_2": block_size_2}
        output, output_x = gr_normalized(problem, exitcriterion, param)

    elif algorithm == "GR_TORCH":
        beta = args.beta
        block_size = args.block_size
        block_size_2 = args.block_size_2
        logging.info("Running Golden Ratio (torch)...")
        param = {
            "beta": beta,
            "block_size": block_size,
            "block_size_2": block_size_2,
            "device": args.device,
            "dtype": args.dtype,
            "use_dense": args.use_dense,
            "dense_threshold": args.dense_threshold,
        }
        output, output_x = gr_torch(problem, exitcriterion, param)

    elif algorithm == "GR_TORCH_normalized":
        beta = args.beta
        block_size = args.block_size
        block_size_2 = args.block_size_2
        logging.info("Running Golden Ratio (torch normalized)...")
        param = {
            "beta": beta,
            "block_size": block_size,
            "block_size_2": block_size_2,
            "device": args.device,
            "dtype": args.dtype,
            "use_dense": args.use_dense,
            "dense_threshold": args.dense_threshold,
        }
        output, output_x = gr_torch_normalized(problem, exitcriterion, param)

    elif algorithm in {"ADUCA", "ADUCA_TORCH_DIST", "ADUCA_DISTRIBUTED"}:
        beta = args.beta
        block_size = args.block_size
        block_size_2 = args.block_size_2
        param = {
            "beta": beta,
            "gamma": args.gamma,
            "rho": args.rho,
            "mu": args.mu,
            "block_size": block_size,
            "block_size_2": block_size_2,
        }
        if use_dist_aduca:
            backend = backend_arg if backend_arg is not None else ("torch_dist" if world_size > 1 else "numpy")
            logging.info(f"Running distributed ADUCA with backend={backend}...")
            param.update({
                "backend": backend,
                "dist_backend": args.dist_backend,
                "dtype": args.dtype,
                "use_dense": bool(use_dense),
                "dense_threshold": args.dense_threshold,
                "strong_convexity": args.strong_convexity,
            })
            output, output_x = aduca_distributed(problem, exitcriterion, param)
        else:
            logging.info("Running ADUCA...")
            output, output_x = aduca(problem, exitcriterion, param)

    else:
        print("Wrong algorithm name supplied")
    
    if (not use_dist_aduca) or rank == 0:
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
