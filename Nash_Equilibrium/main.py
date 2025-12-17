from pathlib import Path
import time
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

# Scenarios and algorithms to run
scenarios = [1,2,3]

algorithms = [
              "PCCM", 
              "CODER", 
              "CODER_linesearch", 
              "GR", 
              "ADUCA"
              ]

# Output directories
output_root = Path("./output")
traj_dir = output_root / "traj"
log_dir = output_root / "log"
plot_dir = output_root / "plot"
for d in (traj_dir, log_dir, plot_dir):
    d.mkdir(parents=True, exist_ok=True)
# Timestamped subfolder for this run
output_run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
output_run_dir = traj_dir / output_run_stamp
output_run_dir.mkdir(parents=True, exist_ok=True)

# Base parameters (mirror those used in existing trajectory files)
base_params = {
    "maxiter": 2_000_000,
    "maxtime": 500_000,
    "targetaccuracy": 1e-14,
    "optval": 0.0,
    "loggingfreq": 20,
    "mu": 0.0,
    "block_size": 64,
}

# Per-scenario overrides; add/adjust values as needed.
# You can also supply algorithm-specific overrides under "algorithms".
# Example of algorithm-specific overrides:
# 1: {
#     "lipschitz": 15.0,
#     "algorithms": {
#         "ADUCA": {"rho": 1.10},
#         "GR": {"beta": 0.85},
#     },
# },
scenario_params = {
    1: {"lipschitz": 100, },
    2: {"lipschitz": 100, },
    3: {"lipschitz": 500, },
    4: {"lipschitz": 20, },
    5: {"lipschitz": 20, },
    6: {"lipschitz": 20, },
    7: {"lipschitz": 20, },
    8: {"lipschitz": 20, },
    9: {"lipschitz": 20, },
}

# Parameter sweeps for each algorithm (overrides applied on top of base + scenario settings)
algorithm_param_sets = {
    "GR": [
        {"beta": 0.7},
    ],
    "ADUCA": [
            {"beta": 0.7, "gamma": 0.1, "rho": 1.3},
            {"beta": 0.8, "gamma": 0.2, "rho": 1.2},
            {"beta": 0.9, "gamma": 0.3, "rho": 1.1},
            # {"beta": 0.95, "gamma": 0.43, "rho": 1.05},
            ],
}


def run_task(scenario: int, algo: str, variant_idx=None, variant_overrides=None):
    params = base_params.copy()
    scenario_cfg = scenario_params.get(scenario, {})
    algo_overrides = scenario_cfg.get("algorithms", {}).get(algo, {})

    # Merge overrides (scenario -> algo-specific)
    params.update({k: v for k, v in scenario_cfg.items() if k != "algorithms"})
    params.update(algo_overrides)
    if variant_overrides:
        params.update(variant_overrides)

    lips_tag = str(params["lipschitz"]).replace(".", "p")
    variant_suffix = f"-v{variant_idx}" if variant_idx is not None else ""
    beta = params.get("beta")
    if beta is None:
        beta = 'None'
    ### Name of the output trajectory file 
    params["outputdir"] = str(
        output_run_dir
        / f"scenario-{scenario}-{algo}{variant_suffix}-beta-{beta}-blocksize-{params['block_size']}-time-{output_run_stamp}.json"
    )

    cmd = ["python", "run_algos.py", "--scenario", str(scenario), "--algo", algo]
    for key, value in params.items():
        cmd += [f"--{key}", str(value)]

    log_path = log_dir / f"scenario{scenario}-{algo}{variant_suffix}-lips{lips_tag}.log"
    print(f"Launching scenario {scenario} | {algo}{variant_suffix}: {' '.join(shlex.quote(c) for c in cmd)}")
    with open(log_path, "w") as logf:
        logf.write("Command: " + " ".join(cmd) + "\n\n")
        result = subprocess.run(cmd, cwd=".", stdout=logf, stderr=subprocess.STDOUT)
    return scenario, algo, variant_suffix, result.returncode, log_path


if __name__ == "__main__":
    tasks = []
    for s in scenarios:
        for algo in algorithms:
            param_variants = algorithm_param_sets.get(algo)
            if param_variants:
                for idx, variant in enumerate(param_variants, start=1):
                    tasks.append((s, algo, idx, variant))
            else:
                tasks.append((s, algo, None, None))

    max_workers = len(tasks)*len(algorithms)*5  # Adjust as needed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_task, s, a, v_idx, variant): (s, a, v_idx)
            for s, a, v_idx, variant in tasks
        }
        for future in as_completed(futures):
            scenario, algo, variant_suffix, rc, log_path = future.result()
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"scenario {scenario} | {algo}{variant_suffix} finished: {status}; log: {log_path}")
