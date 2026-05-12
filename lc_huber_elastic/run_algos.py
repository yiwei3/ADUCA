"""Run script for the LC-Huber benchmark (Figure 3) and extensions.

This runner mirrors the structure of the other benchmark folders.

Important: opt-measure
----------------------
For this benchmark we monitor a *primal objective diagnostic* based on the
Huber loss, instead of the VI residual ||F(x)||.

Given an iterate x=(u,v), we compute the orthogonal projection of u onto the
feasible set {u: Au=b} and evaluate the Huber objective on that projected
vector. We then subtract the optimal objective value (which can be computed
exactly for this radial Huber objective) and log the resulting nonnegative
"projected Huber objective gap".

Usage example
-------------
From the `lc_huber` directory:

    python run_algos.py --scenario 5 --algo ADUCA_TORCH --beta 0.8 --gamma 0.2 --rho 1.2 \
        --block_size_u 100 --block_size_v 50 --device cuda:0 --dtype float64 \
        --maxiter 500000 --loggingfreq 20 --targetaccuracy 1e-8 \
        --outputdir output/traj/test.json

All results are written to the JSON file specified by `--outputdir`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from typing import Dict, Optional

from src.algorithms.aduca_torch import aduca_torch
from src.algorithms.coder import (
    coder,
    coder_linesearch,
    coder_linesearch_normalized,
    coder_normalized,
)
from src.algorithms.gr import gr, gr_normalized
from src.algorithms.pccm import pccm, pccm_normalized
from src.algorithms.utils.exitcriterion import ExitCriterion
from src.problems.lc_huber_instance import make_lc_huber_problem


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LC-Huber benchmark runner")

    # ------------------------------------------------------------------
    # Problem instance
    # ------------------------------------------------------------------
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # Regularization (elastic-net) on u
    parser.add_argument("--lambda1", type=float, default=0.0, help="L1 regularization weight on u")
    parser.add_argument("--lambda2", type=float, default=0.0, help="L2 regularization weight on u")

    # Optional scenario overrides (useful for quick parameter sweeps)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--A_std", type=float, default=None)

    parser.add_argument(
        "--A_kind",
        type=str,
        default=None,
        choices=[
            None,
            "dense",
            "sparse_mask",
            "sparse_degree",
            "banded",
            "block_dominant_dense",
            "banded_plus_noise_dense",
            "clustered_dense",
            "lowrank_plus_noise_dense",
        ],
        help="Override the A pattern (dense/sparse/banded or structured dense)",
    )
    parser.add_argument(
        "--A_density",
        type=float,
        default=None,
        help="For sparse_mask: Bernoulli density. For banded: approximate density (used if band_width not set).",
    )
    parser.add_argument(
        "--degree_per_col",
        type=int,
        default=None,
        help="For sparse_degree: nonzeros per column (if not set, derived from A_density*m).",
    )
    parser.add_argument(
        "--band_width",
        type=int,
        default=None,
        help="For banded: nonzeros per row in the band window.",
    )

    parser.add_argument(
        "--col_scaling_kind",
        type=str,
        default=None,
        choices=[None, "none", "lognormal", "powerlaw"],
    )
    parser.add_argument("--col_scale_strength", type=float, default=None)
    parser.add_argument(
        "--row_scaling_kind",
        type=str,
        default=None,
        choices=[None, "none", "lognormal"],
    )
    parser.add_argument("--row_scale_strength", type=float, default=None)

    # Optional stiff-but-inactive columns (Scenario F-style)
    parser.add_argument("--stiff_inactive_frac", type=float, default=None)
    parser.add_argument("--stiff_inactive_scale", type=float, default=None)

    # ------------------------------------------------------------------
    # Algorithm selection
    # ------------------------------------------------------------------
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=[
            "ADUCA_TORCH",
            "GR",
            "GR_normalized",
            "PCCM",
            "PCCM_normalized",
            "CODER",
            "CODER_normalized",
            "CODER_linesearch",
            "CODER_linesearch_normalized",
        ],
    )

    # ------------------------------------------------------------------
    # Common algorithm knobs
    # ------------------------------------------------------------------
    parser.add_argument("--block_size", type=int, default=1)
    parser.add_argument("--block_size_u", type=int, default=None)
    parser.add_argument("--block_size_v", type=int, default=None)

    parser.add_argument("--preconditioner", type=str, default="diag_lipschitz", choices=["identity", "diag_lipschitz"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--opt_measure",
        type=str,
        default="prox_residual",
        choices=["prox_residual", "projected_primal_gap"],
        help="Opt-measure used for logging and stopping.",
    )

    # GR / ADUCA parameters
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--a_max", type=float, default=1.0, help="ADUCA local backtracking initialization a_max")

    # Lipschitz settings for CODER/PCCM
    parser.add_argument("--lipschitz", type=float, default=None)
    parser.add_argument("--lipschitz_mult", type=float, default=1.0)
    parser.add_argument(
        "--lipschitz_method",
        type=str,
        default="auto",
        choices=["auto", "spectral", "one_inf", "fro"],
        help="How to estimate ||A|| for L (auto uses a cheap bound for large A).",
    )
    parser.add_argument(
        "--lipschitz_size_threshold",
        type=int,
        default=5_000_000,
        help="Max A.size for spectral norm when lipschitz_method=auto.",
    )

    # Problem construction controls
    parser.add_argument(
        "--compute_optval",
        type=str,
        default="auto",
        help="Compute optval_huber (auto skips for large A).",
    )
    parser.add_argument(
        "--optval_size_threshold",
        type=int,
        default=1_000_000,
        help="Max A.size for optval when compute_optval=auto.",
    )
    parser.add_argument(
        "--optval_device",
        type=str,
        default=None,
        help="Device for optval solve (e.g., cuda:0). Defaults to --device if unset.",
    )
    parser.add_argument(
        "--generate_device",
        type=str,
        default=None,
        help="Optional device for dense A generation (e.g., cuda:0).",
    )

    # Linesearch settings
    parser.add_argument("--L_init", type=float, default=1e-7)
    parser.add_argument("--min_step", type=float, default=1e-6)

    # Exit criteria
    parser.add_argument("--maxiter", type=int, default=1_000_000)
    parser.add_argument("--maxtime", type=float, default=200000.0)
    parser.add_argument("--targetaccuracy", type=float, default=1e-8)
    parser.add_argument("--loggingfreq", type=int, default=20)

    # Output
    parser.add_argument("--outputdir", type=str, required=True, help="Path to output JSON file")

    return parser.parse_args()


def _build_override(args: argparse.Namespace) -> Optional[Dict]:
    override: Dict = {}
    for name in (
        "n",
        "m",
        "delta",
        "sparsity",
        "A_std",
        "A_kind",
        "A_density",
        "degree_per_col",
        "band_width",
        "stiff_inactive_frac",
        "stiff_inactive_scale",
    ):
        val = getattr(args, name)
        if val is not None:
            override[name] = val

    if args.col_scaling_kind is not None:
        override["col_scaling_kind"] = None if args.col_scaling_kind == "none" else args.col_scaling_kind
    if args.col_scale_strength is not None:
        override["col_scale_strength"] = args.col_scale_strength

    if args.row_scaling_kind is not None:
        override["row_scaling_kind"] = None if args.row_scaling_kind == "none" else args.row_scaling_kind
    if args.row_scale_strength is not None:
        override["row_scale_strength"] = args.row_scale_strength

    return override if override else None


def _infer_lipschitz(L_est: float, args: argparse.Namespace) -> float:
    """Infer an L value for CODER/PCCM when the user does not specify one."""

    if args.lipschitz is not None:
        L = float(args.lipschitz)
        if L <= 0:
            raise ValueError("--lipschitz must be positive")
        return L

    mult = float(args.lipschitz_mult)
    if mult <= 0:
        raise ValueError("--lipschitz_mult must be positive")

    return float(L_est) * mult


def main() -> None:
    args = parse_commandline()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Output path handling
    output_path = os.path.abspath(args.outputdir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    logging.info("timestamp = %s", timestamp)

    # ------------------------------------------------------------------
    # Exit criterion
    # ------------------------------------------------------------------
    # We stop based on the projected primal Huber objective gap (>=0).
    exitcriterion = ExitCriterion(
        args.maxiter,
        args.maxtime,
        args.targetaccuracy,
        args.loggingfreq,
    )

    # ------------------------------------------------------------------
    # Build problem instance
    # ------------------------------------------------------------------
    override = _build_override(args)
    problem, x0, info = make_lc_huber_problem(
        scenario=args.scenario,
        seed=args.seed,
        override=override,
        lambda1=float(args.lambda1),
        lambda2=float(args.lambda2),
        lipschitz_method=str(args.lipschitz_method),
        lipschitz_size_threshold=int(args.lipschitz_size_threshold),
        compute_optval=str(args.compute_optval),
        optval_size_threshold=int(args.optval_size_threshold),
        optval_device=str(args.optval_device) if args.optval_device is not None else str(args.device),
        generate_device=str(args.generate_device) if args.generate_device is not None else None,
    )

    instance_summary = {
        k: info.get(k)
        for k in (
            "scenario",
            "seed",
            "n",
            "m",
            "delta",
            "sparsity",
            "A_std",
            "A_kind",
            "A_density",
            "degree_per_col",
            "band_width",
            "col_scaling_kind",
            "col_scale_strength",
            "row_scaling_kind",
            "row_scale_strength",
            "L_est",
            "optval_huber",
            "lambda1",
            "lambda2",
        )
        if k in info
    }
    instance_summary["measure_kind"] = str(args.opt_measure)

    logging.info(
        "Instance: n=%d, m=%d, delta=%.3g, sparsity=%.3g, A_kind=%s",
        instance_summary["n"],
        instance_summary["m"],
        instance_summary["delta"],
        instance_summary["sparsity"],
        instance_summary.get("A_kind", "dense"),
    )
    logging.info(
        "A extras: density=%s, degree_per_col=%s, band_width=%s",
        instance_summary.get("A_density", None),
        instance_summary.get("degree_per_col", None),
        instance_summary.get("band_width", None),
    )
    logging.info(
        "scaling: col=%s(%.3g), row=%s(%.3g)",
        instance_summary.get("col_scaling_kind", None),
        float(instance_summary.get("col_scale_strength", 0.0) or 0.0),
        instance_summary.get("row_scaling_kind", None),
        float(instance_summary.get("row_scale_strength", 0.0) or 0.0),
    )
    logging.info("optval_huber (for reference) = %.6e", float(instance_summary.get("optval_huber", 0.0)))

    # Lipschitz estimate for CODER/PCCM
    L_est = float(instance_summary.get("L_est", 0.0) or 0.0)
    L_used = _infer_lipschitz(L_est, args)
    instance_summary["L_used"] = float(L_used)
    logging.info("Using L_used = %.6e", L_used)

    # Block sizes (u/v separated)
    bs_u = int(args.block_size_u) if args.block_size_u is not None else int(args.block_size)
    bs_v = int(args.block_size_v) if args.block_size_v is not None else int(args.block_size)

    # ------------------------------------------------------------------
    # Dispatch algorithm
    # ------------------------------------------------------------------
    algo = args.algo

    if algo == "ADUCA_TORCH":
        for name in ("beta", "gamma", "rho"):
            if getattr(args, name) is None:
                raise ValueError(f"--{name} is required for ADUCA_TORCH")

        param = {
            "beta": float(args.beta),
            "gamma": float(args.gamma),
            "rho": float(args.rho),
            "a_max": float(args.a_max),
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "device": str(args.device),
            "dtype": str(args.dtype),
            "preconditioner": str(args.preconditioner),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = aduca_torch(problem, exitcriterion, param, u_0=x0)

    elif algo == "GR":
        if args.beta is None:
            raise ValueError("--beta is required for GR")
        param = {
            "beta": float(args.beta),
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "device": str(args.device),
            "dtype": str(args.dtype),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = gr(problem, exitcriterion, param, x0=x0)

    elif algo == "GR_normalized":
        if args.beta is None:
            raise ValueError("--beta is required for GR_normalized")
        param = {
            "beta": float(args.beta),
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "preconditioner": str(args.preconditioner),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = gr_normalized(problem, exitcriterion, param, x0=x0)

    elif algo == "PCCM":
        param = {
            "L": float(L_used),
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "device": str(args.device),
            "dtype": str(args.dtype),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = pccm(problem, exitcriterion, param, x0=x0)

    elif algo == "PCCM_normalized":
        # For diagonal preconditioning, a scaled Lipschitz is often O(1).
        L_scaled = float(args.lipschitz) if args.lipschitz is not None else 1.0
        L_scaled *= float(args.lipschitz_mult)
        param = {
            "L": float(L_scaled),
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "preconditioner": str(args.preconditioner),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = pccm_normalized(problem, exitcriterion, param, x0=x0)

    elif algo == "CODER":
        param = {
            "L": float(L_used),
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "device": str(args.device),
            "dtype": str(args.dtype),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = coder(problem, exitcriterion, param, x0=x0)

    elif algo == "CODER_normalized":
        L_scaled = float(args.lipschitz) if args.lipschitz is not None else 1.0
        L_scaled *= float(args.lipschitz_mult)
        param = {
            "L": float(L_scaled),
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "preconditioner": str(args.preconditioner),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = coder_normalized(problem, exitcriterion, param, x0=x0)

    elif algo == "CODER_linesearch":
        param = {
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "L_init": float(args.L_init),
            "min_step": float(args.min_step),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = coder_linesearch(problem, exitcriterion, param, x0=x0)

    elif algo == "CODER_linesearch_normalized":
        param = {
            "block_size_u": bs_u,
            "block_size_v": bs_v,
            "L_init": float(args.L_init),
            "min_step": float(args.min_step),
            "preconditioner": str(args.preconditioner),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "opt_measure": str(args.opt_measure),
        }
        output, output_x = coder_linesearch_normalized(problem, exitcriterion, param, x0=x0)

    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    with open(output_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "instance": instance_summary,
                "output_x": output_x.tolist(),
                "iterations": output.iterations,
                "times": output.times,
                "optmeasures": output.optmeasures,
                "L": output.L,
                "L_hat": output.L_hat,
            },
            f,
        )

    logging.info("output saved to %s", output_path)


if __name__ == "__main__":
    main()
