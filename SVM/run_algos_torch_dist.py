
### SVM (Torch Distributed ADUCA runner)

import argparse
import datetime
import numpy as np
import logging
import json
import os
from pathlib import Path

from src.algorithms.utils.exitcriterion import ExitCriterion
from src.algorithms.aduca_torch_dist import aduca_distributed  # <-- distributed-aware ADUCA

from src.problems.utils.data_parsers import libsvm_parser
from src.problems.GMVI_func import GMVIProblem
from src.problems.operator_func.svmelastic_opr_func import SVMElasticOprFunc
from src.problems.g_func.svmelastic_g_func import SVMElasticGFunc

# Configure logging (rank filtering happens inside ADUCA as well)
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
}

def parse_commandline():
    parser = argparse.ArgumentParser(description='Run distributed ADUCA for SVM-ElasticNet.')
    parser.add_argument('--outputdir', required=True, help='Output directory')
    parser.add_argument('--maxiter', required=True, type=int, help='Max iterations')
    parser.add_argument('--maxtime', required=True, type=int, help='Max execution time in seconds')
    parser.add_argument('--targetaccuracy', required=True, type=float, help='Target accuracy')
    parser.add_argument('--optval', type=float, default=0.0, help='Known optimal value')
    parser.add_argument('--loggingfreq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--dataset', required=True, help='Choice of dataset')
    parser.add_argument('--lambda1', type=float, default=0.0, help='Elastic net lambda 1')
    parser.add_argument('--lambda2', type=float, default=0.0, help='Elastic net lambda 2')
    parser.add_argument('--beta', type=float, required=True, help='ADUCA parameter beta')
    parser.add_argument('--gamma', type=float, required=True, help='ADUCA parameter gamma')
    parser.add_argument('--rho', type=float, default=0.0, help='ADUCA parameter rho')
    parser.add_argument('--mu', type=float, default=0.0, help='Mu (unused in this ADUCA variant)')
    parser.add_argument('--strong-convexity', '--strong_convexity', dest='strong_convexity',
                        action='store_true', help='Enable strong convexity ratio_bar update')
    parser.add_argument('--block_size', type=int, default=1, help='x-block size (for iteration accounting)')
    parser.add_argument('--block_size_2', type=int, default=10**9, help='y-block size (for iteration accounting)')
    parser.add_argument('--dist_backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='torch.distributed backend')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'], help='Torch dtype')
    parser.add_argument('--use_dense', action='store_true', help='Force dense GEMV (useful if A is dense)')
    parser.add_argument('--dense_threshold', type=float, default=0.25, help='Auto-dense switch threshold if not forcing')

    return parser.parse_args()

def main():
    args = parse_commandline()

    # Detect torchrun rank/world-size (if not present, default to single process)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Problem Setup
    dataset = args.dataset
    if dataset not in DATASET_INFO:
        raise ValueError("Invalid dataset name supplied.")
    algo = "ADUCA_distributed"

    d, n = DATASET_INFO[dataset]
    if rank == 0:
        logging.info(f"dataset: {dataset}, d: {d}, n: {n}")
        logging.info(f"elasticnet_λ₁ = {args.lambda1}; elasticnet_λ₂ = {args.lambda2}")
        logging.info(f"world_size = {world_size}")
        logging.info("--------------------------------------------------")

    # Exit criterion
    targetaccuracy = args.targetaccuracy + args.optval
    exitcriterion = ExitCriterion(args.maxiter, args.maxtime, targetaccuracy, args.loggingfreq)

    # Output filename (rank 0 only writes)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    traj_dir = Path(args.outputdir)
    if 'traj' not in traj_dir.parts:
        traj_dir = traj_dir / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)
    outputfilename = traj_dir / f"{dataset}-beta-{args.beta}-ADUCA-torchdist-blocksize-{args.block_size}-{args.block_size_2}-time-{timestamp}.json"
    if rank == 0:
        logging.info(f"outputfilename = {outputfilename}")
        logging.info("--------------------------------------------------")

    # Load data (each rank loads and slices locally inside ADUCA)
    filepath = f"../data/{dataset}"
    data = libsvm_parser(filepath, n, d)
    F = SVMElasticOprFunc(data)
    g = SVMElasticGFunc(d, n, args.lambda1, args.lambda2)
    problem = GMVIProblem(F, g)

    # ADUCA parameters
    param = {
        "beta": args.beta,
        "gamma": args.gamma,
        "rho": args.rho,
        "mu": args.mu,
        "strong_convexity": args.strong_convexity,
        "block_size": args.block_size,
        "block_size_2": args.block_size_2,

        # Distributed controls
        "backend": "torch_dist" if world_size > 1 else "numpy",
        "dist_backend": args.dist_backend,
        "dtype": args.dtype,
        "use_dense": bool(args.use_dense),
        "dense_threshold": float(args.dense_threshold),
    }

    output, output_x = aduca_distributed(problem, exitcriterion, param)

    if rank == 0:
        with open(outputfilename, 'w') as outfile:
            json.dump({
                "args": vars(args),
                "algo": algo,
                "output_x": output_x.tolist(),
                "iterations": output.iterations,
                "times": output.times,
                "optmeasures": output.optmeasures,
                "L": output.L,
                "L_hat": output.L_hat
            }, outfile)
        logging.info(f"output saved to {outputfilename}")

if __name__ == "__main__":
    main()
