"""Command-line entry point for BPR traffic equilibrium experiments."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.data.datasets import available_dataset_sources, load_traffic_problem
from src.optimizers.aduca import run_aduca
from src.optimizers.coder import run_coder, run_coder_linesearch
from src.optimizers.graal import run_graal
from src.optimizers.pccm import run_pccm


def _none_or_float(x: str) -> float | None:
    if x.lower() in {"none", "auto", "null", ""}:
        return None
    return float(x)


def _none_or_int(x: str) -> int | None:
    if x.lower() in {"none", "all", "null", ""}:
        return None
    return int(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADUCA and baselines on nonlinear BPR traffic equilibrium.")

    # Dataset/path-set options.
    parser.add_argument("--dataset-source", default="synthetic_grid", choices=available_dataset_sources())
    parser.add_argument("--data-root", default="data", help="Root containing data/raw and data/processed.")
    parser.add_argument("--gmns-dir", default=None, help="Folder containing local GMNS link.csv and demand.csv when source=gmns_local.")
    parser.add_argument("--auto-download", action="store_true", help="Download registered TNTP files if missing.")
    parser.add_argument("--path-cache-dir", default=None, help="Optional directory for generated path-set NPZ caches.")
    parser.add_argument("--refresh-path-cache", action="store_true", help="Regenerate and overwrite path-set cache files.")
    parser.add_argument("--k-paths", type=int, default=5, help="Number of free-flow shortest paths per selected OD pair.")
    parser.add_argument("--max-od-pairs", type=_none_or_int, default=20, help="Maximum number of OD pairs; use 'all' for all.")
    parser.add_argument("--min-demand", type=float, default=0.0)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--od-strategy", choices=["top", "first", "random"], default="top")
    parser.add_argument("--max-path-hops", type=_none_or_int, default=None)
    parser.add_argument("--synthetic-grid-rows", type=int, default=4)
    parser.add_argument("--synthetic-grid-cols", type=int, default=4)
    parser.add_argument("--synthetic-demand-scale", type=float, default=100.0)
    parser.add_argument("--bpr-alpha-override", type=_none_or_float, default=None)
    parser.add_argument("--bpr-power-override", type=_none_or_float, default=None)
    parser.add_argument("--path-regularization", type=float, default=0.0)
    parser.add_argument("--lambda-mode", choices=["ones", "path_time", "sqrt_path_time", "inv_demand", "inv_sqrt_demand"], default="ones")
    parser.add_argument("--init-mode", choices=["uniform", "shortest", "random_simplex"], default="uniform")
    parser.add_argument("--seed", type=int, default=0)

    # Experiment options.
    parser.add_argument("--methods", default="aduca,coder,coder_linesearch,graal,pccm", help="Comma-separated methods.")
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default=None, help="Folder under output-root. Defaults to timestamp.")
    parser.add_argument("--trial-name", default=None, help="Subfolder name for this run. Defaults to dataset/method timestamp.")

    # ADUCA hyperparameters.
    parser.add_argument("--aduca-beta", type=float, default=0.8)
    parser.add_argument("--aduca-gamma", type=float, default=0.2)
    parser.add_argument("--aduca-rho", type=float, default=1.2)
    parser.add_argument("--aduca-mu", type=float, default=0.0)
    parser.add_argument("--aduca-a0", type=_none_or_float, default=None, help="Compatibility no-op; ADUCA initialization now chooses its own test-step backtracking start.")
    parser.add_argument("--aduca-max-stepsize", type=float, default=1e8, help="Safety cap for ADUCA stepsizes.")
    parser.add_argument("--aduca-safeguard-primary", action="store_true", help="Backtrack ADUCA steps that increase the traffic relative gap.")
    parser.add_argument("--aduca-safeguard-shrink", type=float, default=0.5)
    parser.add_argument("--aduca-safeguard-max-backtracks", type=int, default=8)
    parser.add_argument("--aduca-warmup-steps", type=int, default=0, help="Early ADUCA cycles that limit multiplicative stepsize changes.")
    parser.add_argument("--aduca-warmup-change-factor", type=float, default=2.0, help="Per-cycle ADUCA stepsize change cap at the start of warm-up.")

    # Baseline hyperparameters.
    parser.add_argument("--pccm-stepsize", type=_none_or_float, default=None)
    parser.add_argument("--coder-lhat", type=_none_or_float, default=None)
    parser.add_argument("--coder-gamma", type=float, default=0.0)
    parser.add_argument("--coder-ls-lhat0", type=_none_or_float, default=None)
    parser.add_argument("--graal-a0", type=_none_or_float, default=None)
    parser.add_argument("--graal-growth", type=float, default=1.15)
    parser.add_argument("--graal-lipschitz-coeff", type=float, default=0.45)
    parser.add_argument("--graal-phi", type=float, default=(1.0 + math.sqrt(5.0)) / 2.0)

    return parser.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _run_one_method(method: str, problem: Any, x0: np.ndarray, args: argparse.Namespace):
    method = method.lower().strip()
    if method == "aduca":
        return run_aduca(
            problem,
            x0,
            num_iterations=args.num_iterations,
            beta=args.aduca_beta,
            gamma=args.aduca_gamma,
            rho=args.aduca_rho,
            mu=args.aduca_mu,
            log_every=args.log_every,
            max_stepsize=args.aduca_max_stepsize,
            safeguard_primary=args.aduca_safeguard_primary,
            safeguard_shrink=args.aduca_safeguard_shrink,
            safeguard_max_backtracks=args.aduca_safeguard_max_backtracks,
            warmup_steps=args.aduca_warmup_steps,
            warmup_change_factor=args.aduca_warmup_change_factor,
        )
    if method == "pccm":
        return run_pccm(problem, x0, num_iterations=args.num_iterations, stepsize=args.pccm_stepsize, log_every=args.log_every)
    if method == "coder":
        return run_coder(problem, x0, num_iterations=args.num_iterations, lhat=args.coder_lhat, gamma=args.coder_gamma, log_every=args.log_every)
    if method in {"coder_linesearch", "coder-ls", "coder_ls"}:
        return run_coder_linesearch(
            problem,
            x0,
            num_iterations=args.num_iterations,
            lhat0=args.coder_ls_lhat0,
            gamma=args.coder_gamma,
            log_every=args.log_every,
        )
    if method == "graal":
        return run_graal(
            problem,
            x0,
            num_iterations=args.num_iterations,
            a0=args.graal_a0,
            phi=args.graal_phi,
            growth=args.graal_growth,
            lipschitz_coeff=args.graal_lipschitz_coeff,
            log_every=args.log_every,
        )
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"traffic_bpr_{timestamp}"
    root = Path(args.output_root) / run_name
    root.mkdir(parents=True, exist_ok=True)

    problem, dataset_meta = load_traffic_problem(
        source=args.dataset_source,
        data_root=args.data_root,
        gmns_dir=args.gmns_dir,
        auto_download=args.auto_download,
        k_paths=args.k_paths,
        max_od_pairs=args.max_od_pairs,
        min_demand=args.min_demand,
        demand_scale=args.demand_scale,
        od_strategy=args.od_strategy,
        seed=args.seed,
        synthetic_grid_rows=args.synthetic_grid_rows,
        synthetic_grid_cols=args.synthetic_grid_cols,
        synthetic_demand_scale=args.synthetic_demand_scale,
        bpr_alpha_override=args.bpr_alpha_override,
        bpr_power_override=args.bpr_power_override,
        path_regularization=args.path_regularization,
        lambda_mode=args.lambda_mode,
        max_path_hops=args.max_path_hops,
        path_cache_dir=args.path_cache_dir,
        refresh_path_cache=args.refresh_path_cache,
    )
    x0 = problem.initial_flow(args.init_mode, seed=args.seed)

    print("Problem summary:")
    print(json.dumps(dataset_meta, indent=2, default=_json_default))

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    combined_rows = []
    for method in methods:
        print(f"\nRunning {method} ...")
        result = _run_one_method(method, problem, x0, args)

        trial_base = args.trial_name or f"{args.dataset_source}_{result.method}_{timestamp}"
        trial = trial_base if len(methods) == 1 else f"{trial_base}_{result.method}"
        out_dir = root / trial
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = result.history.copy()
        metrics["dataset_source"] = args.dataset_source
        metrics["trial_name"] = trial
        metrics.to_csv(out_dir / "metrics.csv", index=False)
        np.save(out_dir / "x_final.npy", result.x_final)

        config = {
            "args": vars(args),
            "dataset_meta": dataset_meta,
            "optimizer_config": result.config,
            "method": result.method,
            "output_dir": str(out_dir),
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=_json_default))
        combined_rows.append(metrics)
        print(f"Saved {result.method} outputs to {out_dir}")

    if combined_rows:
        import pandas as pd

        all_metrics = pd.concat(combined_rows, ignore_index=True)
        all_metrics.to_csv(root / "combined_metrics.csv", index=False)
        (root / "problem_summary.json").write_text(json.dumps(dataset_meta, indent=2, default=_json_default))
        print(f"\nCombined metrics saved to {root / 'combined_metrics.csv'}")


if __name__ == "__main__":
    main()
