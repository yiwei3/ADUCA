from pathlib import Path
import socket
import os
import subprocess
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Datasets to run (files must exist in ./data)
datasets = [
    # 'a9a',
    # 'gisette_scale.bz2',
    # 'w8a',
    # 'real-sim',
    # 'epsilon_normalized.t.bz2',
    # 'rcv1_train.binary.bz2',
    # "news20.binary.bz2",
    # 'ijcnn1',
    # 'cod-rna',
    # 'phishing',
    # 'covtype.binary',
    'SUSY',
    # 'HIGGS',
]

# GPU visibility (set to None to use the existing environment)
cuda_visible_devices = "1,2,3,4,5,6,7"
nproc_per_node = 7  # number of GPUs to use
env = os.environ.copy()
if cuda_visible_devices is not None:
    if isinstance(cuda_visible_devices, (list, tuple)):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(v) for v in cuda_visible_devices)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

dtype = 'float32'  # 'float32' or 'float64'

# Strong convexity toggle for ADUCA_TORCH_DIST
strong_convexity = True   # Only for extrapolation term; do not change the stepsizes

# None => use as many simultaneous distributed jobs as the probed GPUs allow.
distributed_parallel_jobs = None

DIST_BOOL_PARAMS = {'use_dense', 'strong_convexity'}
SCRIPT_DIR = Path(__file__).resolve().parent
DISTRIBUTED_ALGOS = {'ADUCA_TORCH_DIST'}

algorithms = [
    # 'PCCM',
    # 'CODER',
    # 'CODER_linesearch',
    # 'GR',
    # 'PCCM_normalized',
    # 'CODER_normalized',
    # 'CODER_linesearch_normalized',
    # 'GR_normalized',
    'ADUCA_TORCH_DIST',
    # 'ADUCA',
]

# Output directories
output_root = Path('./output')
traj_dir = output_root / 'traj'
log_dir = output_root / 'log'
plot_dir = output_root / 'plot'
for d in (traj_dir, log_dir, plot_dir):
    d.mkdir(parents=True, exist_ok=True)
# Timestamped subfolder for this run's trajectories
output_run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
output_run_dir = traj_dir / output_run_stamp
output_run_dir.mkdir(parents=True, exist_ok=True)

# Base parameters shared by all runs (overridable per dataset)
base_params = {
    'outputdir': str(output_run_dir),
    'maxtime': 100_000,
    'targetaccuracy': 0,
    'lambda1': 1e-4,
    'lambda2': 1e-4,
    'mu': 0,
    'block_size': 64,
    'block_size_2': 512,
    'loggingfreq': 1,
}

# Per-dataset overrides (edit as needed)
dataset_params = {
    'a9a': {
        'maxiter': 100_100, 
        'lipschitz': 
        # 0.014,
        0.0007, # preconditioned
    },
    'gisette_scale.bz2': {
        'maxiter': 1_100, 
        'lipschitz': 
        # 0.75,
        0.01, # preconditioned
        'block_size': 500,
        'block_size_2': 60,
    },
    'SUSY': {
        'maxiter': 1_100,
        'lipschitz':
        # 0.003,
        1e-04, # preconditioned
        'block_size': 64,
        'block_size_2': 50_000,
    },
    'HIGGS': {
        'maxiter': 1_100,
        'lipschitz':
        # 0.002,
        5e-05, # preconditioned
        'block_size': 64,
        'block_size_2': 100_000,
    },
    'epsilon_normalized.t.bz2': {
        'maxiter': 1_000, 
        'lipschitz': 
        # 0.002,
        0.0007, # preconditioned
        'block_size': 1000,
        'block_size_2': 1000,
    },
        'w8a': {
        'maxiter': 5_000, 
        'lipschitz': 
        # 0.007, 
        0.0004, # preconditioned
    },
    'real-sim': {
        'maxiter': 1_000, 
        'lipschitz': 
        # 0.0004, 
        0.0002, # preconditioned
        'block_size': 5000,
        'block_size_2': 700,
    },
    'rcv1_train.binary.bz2': {
        'maxiter': 1_000, 
        'lipschitz':
        # 0.001,
        0.0006, # preconditioned
        'block_size': 256,
        'block_size_2': 2048,
    },
    'ijcnn1': {
        'maxiter': 5_000,
        'block_size': 64,
        'block_size_2': 512,
    },
    'cod-rna': {
        'maxiter': 5_000,
        'block_size': 64,
        'block_size_2': 512,
    },
    'phishing': {
        'maxiter': 5_000,
        'block_size': 64,
        'block_size_2': 128,
    },
    'covtype.binary': {
        'maxiter': 1_000,
        'block_size': 64,
        'block_size_2': 4096,
    },
    # "news20.binary.bz2": {
    #     'maxiter': 11_000,
    #     'lipschitz':
    #     0.0005,
    # },
}

# Per-algorithm parameter sets (each dict is one full set of overrides for a run)
algorithm_param_sets = {
    'GR': [
        {'beta': 0.7},
    ],
    'GR_normalized': [
        {'beta': 0.7},
    ],
    'ADUCA': [
        # {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2},
        # {'beta': 0.85, 'gamma': 0.3, 'rho': 1.1},
        # {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1},
    ],
    'ADUCA_TORCH_DIST': [
        # {'beta': 0.7, 'gamma': 0.05, 'rho': 1.2, 'mu': 0,},
        # {'beta': 0.7, 'gamma': 0.05, 'rho': 1.2, 'mu': 1e-1,},
        # {'beta': 0.7, 'gamma': 0.05, 'rho': 1.2, 'mu': 1e-2,},
        # {'beta': 0.7, 'gamma': 0.05, 'rho': 1.2, 'mu': 1e-3,},
        # {'beta': 0.7, 'gamma': 0.05, 'rho': 1.2, 'mu': 1e-4,},
        # {'beta': 0.7, 'gamma': 0.05, 'rho': 1.2, 'mu': 1e-5,},
        {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3, 'mu': 0,},
        {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3, 'mu': 1e-1,},
        {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3, 'mu': 1e-2,},
        {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3, 'mu': 1e-3,},
        {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3, 'mu': 1e-4,},
        {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3, 'mu': 1e-5,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'mu': 0,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'mu': 1e-1,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'mu': 1e-2,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'mu': 1e-3,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'mu': 1e-4},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'mu': 1e-5,},
        {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1, 'mu': 0,},
        {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1, 'mu': 1e-1,},
        {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1, 'mu': 1e-2,},
        {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1, 'mu': 1e-3,},
        {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1, 'mu': 1e-4,},
        {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1, 'mu': 1e-5,},
    ],
}

DEFAULT_SCRIPT = SCRIPT_DIR / 'run_algos.py'
DIST_PARAM_ORDER = [
    'outputdir',
    'maxiter',
    'maxtime',
    'targetaccuracy',
    'optval',
    'loggingfreq',
    'lambda1',
    'lambda2',
    'beta',
    'gamma',
    'rho',
    'mu',
    'strong_convexity',
    'block_size',
    'block_size_2',
    'backend',
    'dist_backend',
    'dtype',
    'use_dense',
    'dense_threshold',
]

# Build all (dataset, algo, variant) tasks
tasks = []
for ds in datasets:
    for algo in algorithms:
        param_variants = algorithm_param_sets.get(algo)
        if param_variants:
            if len(param_variants) == 1:
                tasks.append((ds, algo, None, param_variants[0]))
            else:
                for idx, variant in enumerate(param_variants, start=1):
                    tasks.append((ds, algo, idx, variant))
        else:
            tasks.append((ds, algo, None, None))

def _find_free_port():
    """Pick an open TCP port for torchrun rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _parse_visible_cuda_devices(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [entry.strip() for entry in text.split(",") if entry.strip()]


def _probe_cuda_device(device_id: str, base_env):
    probe_env = base_env.copy()
    probe_env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    probe_cmd = [
        sys.executable,
        "-c",
        (
            "import torch; "
            "assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'; "
            "torch.cuda.set_device(0); "
            "x = torch.empty(1, device='cuda:0'); "
            "torch.cuda.synchronize(); "
            "del x"
        ),
    ]
    result = subprocess.run(
        probe_cmd,
        cwd=str(SCRIPT_DIR),
        env=probe_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return True, ""
    detail = (result.stderr or result.stdout).strip()
    return False, detail or "CUDA probe failed without stderr output."


def _build_distributed_launch_specs(run_env, distributed_task_count: int):
    if nproc_per_node < 1:
        raise ValueError(f"nproc_per_node must be >= 1, got {nproc_per_node}.")
    if distributed_task_count < 1:
        return []

    requested_devices = _parse_visible_cuda_devices(run_env.get("CUDA_VISIBLE_DEVICES"))
    if requested_devices is None:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be set when running simultaneous distributed jobs "
            "so the launcher can partition GPUs across tasks."
        )
    if not requested_devices:
        raise ValueError("CUDA_VISIBLE_DEVICES is empty, so no GPUs are available for torchrun.")

    usable_devices = []
    failures = {}
    for device_id in requested_devices:
        ok, detail = _probe_cuda_device(device_id, run_env)
        if ok:
            usable_devices.append(device_id)
        else:
            failures[device_id] = detail

    if not usable_devices:
        failure_lines = [f"GPU {device_id}: {detail}" for device_id, detail in failures.items()]
        raise RuntimeError("No requested CUDA devices were usable.\n" + "\n".join(failure_lines))

    max_parallel_jobs = min(distributed_task_count, len(usable_devices))
    if distributed_parallel_jobs is not None:
        if distributed_parallel_jobs < 1:
            raise ValueError(
                f"distributed_parallel_jobs must be >= 1, got {distributed_parallel_jobs}."
            )
        max_parallel_jobs = min(max_parallel_jobs, distributed_parallel_jobs)

    gpu_groups = [[] for _ in range(max_parallel_jobs)]
    device_cursor = 0
    for group in gpu_groups:
        if device_cursor >= len(usable_devices):
            break
        group.append(usable_devices[device_cursor])
        device_cursor += 1

    while device_cursor < len(usable_devices):
        assigned = False
        for group in gpu_groups:
            if len(group) >= nproc_per_node:
                continue
            group.append(usable_devices[device_cursor])
            device_cursor += 1
            assigned = True
            if device_cursor >= len(usable_devices):
                break
        if not assigned:
            break

    launch_specs = []
    for slot_idx, devices in enumerate(gpu_groups, start=1):
        if not devices:
            continue
        launch_env = run_env.copy()
        launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        launch_specs.append(
            {
                "slot_idx": slot_idx,
                "devices": devices,
                "env": launch_env,
                "nproc_per_node": len(devices),
            }
        )

    failed_devices = [device_id for device_id in requested_devices if device_id in failures]
    unused_devices = usable_devices[device_cursor:]
    if failed_devices:
        print(f"Skipping unusable CUDA device(s): {','.join(failed_devices)}.")
    if unused_devices:
        print(
            f"Leaving usable CUDA device(s) idle due to task/gpu limits: {','.join(unused_devices)}."
        )

    print(
        f"Scheduling {distributed_task_count} distributed job(s) across {len(launch_specs)} "
        f"GPU group(s)."
    )
    for spec in launch_specs:
        print(
            f"  slot {spec['slot_idx']}: CUDA_VISIBLE_DEVICES={spec['env']['CUDA_VISIBLE_DEVICES']} "
            f"(nproc_per_node={spec['nproc_per_node']})"
        )

    return launch_specs


def _is_distributed_task(algo):
    return algo in DISTRIBUTED_ALGOS


def run_task(ds: str, algo: str, variant_idx=None, variant_overrides=None, dist_launch_spec=None):
    params = base_params.copy()
    params.update(dataset_params.get(ds, {}))
    if variant_overrides:
        params.update(variant_overrides)
    run_env = env.copy()

    # env = dict(os.environ)
    # env.setdefault("OMP_NUM_THREADS", "1")
    # env.setdefault("MKL_NUM_THREADS", "1")
    # env.setdefault("OPENBLAS_NUM_THREADS", "1")
    # env.setdefault("NUMEXPR_NUM_THREADS", "1")
    # env.setdefault("NCCL_DEBUG", "WARN")
    # env.setdefault("NCCL_IB_DISABLE", "1")

    if algo == 'ADUCA_TORCH_DIST':
        if dist_launch_spec is None:
            raise ValueError("dist_launch_spec is required for ADUCA_TORCH_DIST tasks.")
        run_env = dist_launch_spec["env"].copy()
        run_nproc = int(dist_launch_spec["nproc_per_node"])
        if strong_convexity:
            params['strong_convexity'] = True
        if dtype is not None:
            params['dtype'] = dtype
        params['backend'] = 'torch_dist'
        ### Distributed run with --nproc_per_node=j for using j GPUs
        master_port = _find_free_port()
        cmd = ['torchrun', 
               f'--nproc_per_node={run_nproc}',
                '--master-port', str(master_port),
                str(DEFAULT_SCRIPT),
                '--algo', 'ADUCA',
                '--dataset', ds]
        for key in DIST_PARAM_ORDER:
            if key not in params:
                continue
            val = params[key]
            if key in DIST_BOOL_PARAMS:
                if val:
                    cmd.append(f'--{key}')
                continue
            cmd += [f'--{key}', str(val)]
    else:
        cmd = [sys.executable, str(DEFAULT_SCRIPT), '--algo', algo, '--dataset', ds]
        for k, v in params.items():
            cmd += [f'--{k}', str(v)]

    variant_suffix = f"-v{variant_idx}" if variant_idx is not None else ""
    log_path = log_dir / f"{ds}-{algo}{variant_suffix}.log"
    launch_note = ""
    if dist_launch_spec is not None:
        launch_note = (
            f" [slot {dist_launch_spec['slot_idx']} | "
            f"CUDA_VISIBLE_DEVICES={run_env.get('CUDA_VISIBLE_DEVICES')}]"
        )
    print(
        f"Launching {ds} | {algo}{variant_suffix}{launch_note}:",
        ' '.join(shlex.quote(c) for c in cmd),
    )

    # Limit per-process threading to avoid CPU oversubscription when running many tasks in parallel

    with open(log_path, 'w') as logf:
        logf.write('Command: ' + ' '.join(cmd) + '\n\n')
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=run_env, stdout=logf, stderr=subprocess.STDOUT)
    return ds, algo, variant_suffix, result.returncode, log_path


def _run_distributed_slot(slot_spec, assigned_tasks):
    results = []
    for ds, algo, v_idx, variant in assigned_tasks:
        results.append(
            run_task(
                ds,
                algo,
                v_idx,
                variant,
                dist_launch_spec=slot_spec,
            )
        )
    return results


def _print_task_result(task_result):
    ds, algo, variant_suffix, rc, log_path = task_result
    status = 'OK' if rc == 0 else f'FAILED (rc=' + str(rc) + ')'
    print(f"{ds} | {algo}{variant_suffix} finished: {status}; log: {log_path}")

regular_tasks = [task for task in tasks if not _is_distributed_task(task[1])]
distributed_tasks = [task for task in tasks if _is_distributed_task(task[1])]

futures = {}
executors = []

if regular_tasks:
    regular_executor = ThreadPoolExecutor(max_workers=max(1, len(regular_tasks)))
    executors.append(regular_executor)
    for ds, algo, v_idx, variant in regular_tasks:
        futures[regular_executor.submit(run_task, ds, algo, v_idx, variant)] = "single"

if distributed_tasks:
    dist_launch_specs = _build_distributed_launch_specs(env, len(distributed_tasks))
    distributed_executor = ThreadPoolExecutor(max_workers=len(dist_launch_specs))
    executors.append(distributed_executor)
    task_groups = [[] for _ in dist_launch_specs]
    for task_idx, task in enumerate(distributed_tasks):
        task_groups[task_idx % len(dist_launch_specs)].append(task)
    for slot_spec, assigned_tasks in zip(dist_launch_specs, task_groups):
        if not assigned_tasks:
            continue
        futures[distributed_executor.submit(_run_distributed_slot, slot_spec, assigned_tasks)] = "batch"

try:
    for fut in as_completed(futures):
        result = fut.result()
        if futures[fut] == "single":
            _print_task_result(result)
        else:
            for task_result in result:
                _print_task_result(task_result)
finally:
    for executor in executors:
        executor.shutdown(wait=True)
