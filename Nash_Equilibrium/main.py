from pathlib import Path
import os
import socket
import time
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

# Scenarios and algorithms to run
scenarios = [1, 2, 3]

DIST_ALGO_NAME = "ADUCA_torch_dist"

algorithms = [
    # "PCCM",
    # "PCCM_normalized",
    # "CODER",
    # "CODER_normalized",
    # "CODER_linesearch",
    # "CODER_linesearch_normalized",
    # "GR",
    # "GR_normalized",
    # "ADUCA",
    DIST_ALGO_NAME,
]

ADUCA_ALGOS = {"ADUCA", DIST_ALGO_NAME}

# Distributed settings for ADUCA_torch_dist
nproc_per_node = 1
dist_backend = "nccl"
dtype = "float64"
reduce_dtype = None  # "float32" or "float64"
sync_step = False
strong_convexity = False

# GPU visibility (set to None to use the existing environment)
cuda_visible_devices = "0,2,3,4,5,6,7"

# Concurrency control (None -> auto)
max_workers = None

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
    # "maxiter": 10_000_000_000,
    "maxiter": 1_000_000,
    "maxtime": 500_000,
    "targetaccuracy": 0,
    "optval": 0.0,
    "loggingfreq": 20,
    "mu": 0.0,
    "block_size": 1000,
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
aduca_param_sets = [
    # {"beta": 0.7, "gamma": 0.1, "rho": 1.3},
    {"beta": 0.8, "gamma": 0.2, "rho": 1.2},
    # {"beta": 0.9, "gamma": 0.3, "rho": 1.1},
    # {"beta": 0.95, "gamma": 0.43, "rho": 1.05},
]

algorithm_param_sets = {
    "GR": [
        {"beta": 0.7},
    ],
    "ADUCA": aduca_param_sets,
    DIST_ALGO_NAME: aduca_param_sets,
}


def _find_free_port(default: int = 29500) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1] or default


def run_task(scenario: int, algo: str, variant_idx=None, variant_overrides=None):
    params = base_params.copy()
    scenario_cfg = scenario_params.get(scenario, {})
    algo_overrides = scenario_cfg.get("algorithms", {}).get(algo, {})

    # Merge overrides (scenario -> algo-specific)
    params.update({k: v for k, v in scenario_cfg.items() if k != "algorithms"})
    params.update(algo_overrides)
    if variant_overrides:
        params.update(variant_overrides)

    if algo in ADUCA_ALGOS:
        params.setdefault("gamma", aduca_param_sets[0]["gamma"])
        params.setdefault("rho", aduca_param_sets[0]["rho"])

    if algo == DIST_ALGO_NAME:
        params["dist_backend"] = dist_backend
        params["dtype"] = dtype
        if reduce_dtype is not None:
            params["reduce_dtype"] = reduce_dtype
        if sync_step:
            params["sync_step"] = True
        if strong_convexity:
            params["strong_convexity"] = True

    lips_tag = str(params["lipschitz"]).replace(".", "p")
    variant_suffix = f"-v{variant_idx}" if variant_idx is not None else ""
    name_parts = [f"scenario-{scenario}", f"{algo}{variant_suffix}"]
    if params.get("beta") is not None:
        name_parts.append(f"beta-{params['beta']}")
    if algo in ADUCA_ALGOS:
        name_parts.append(f"gamma-{params['gamma']}")
        name_parts.append(f"rho-{params['rho']}")
    name_parts.append(f"blocksize-{params['block_size']}")
    name_parts.append(f"time-{output_run_stamp}")
    # Name of the output trajectory file
    params["outputdir"] = str(output_run_dir / f"{'-'.join(name_parts)}.json")

    if algo == DIST_ALGO_NAME and nproc_per_node > 1:
        master_port = _find_free_port()
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            "--master-port",
            str(master_port),
            "run_algos.py",
            "--scenario",
            str(scenario),
            "--algo",
            algo,
        ]
    else:
        cmd = ["python", "run_algos.py", "--scenario", str(scenario), "--algo", algo]

    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd += [f"--{key}", str(value)]

    log_path = log_dir / f"scenario{scenario}-{algo}{variant_suffix}-lips{lips_tag}.log"
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        if isinstance(cuda_visible_devices, (list, tuple)):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(v) for v in cuda_visible_devices)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    print(f"Launching scenario {scenario} | {algo}{variant_suffix}: {' '.join(shlex.quote(c) for c in cmd)}")
    with open(log_path, "w") as logf:
        logf.write("Command: " + " ".join(cmd) + "\\n\\n")
        result = subprocess.run(cmd, cwd=".", env=env, stdout=logf, stderr=subprocess.STDOUT)
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

    auto_workers = len(tasks)
    # auto_workers = max_workers
    # if auto_workers is None:
    #     if DIST_ALGO_NAME in algorithms and nproc_per_node > 1:
    #         auto_workers = 1
    #     else:
    #         auto_workers = len(tasks)

    with ThreadPoolExecutor(max_workers=auto_workers) as executor:
        futures = {
            executor.submit(run_task, s, a, v_idx, variant): (s, a, v_idx)
            for s, a, v_idx, variant in tasks
        }
        for future in as_completed(futures):
            scenario, algo, variant_suffix, rc, log_path = future.result()
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"scenario {scenario} | {algo}{variant_suffix} finished: {status}; log: {log_path}")
