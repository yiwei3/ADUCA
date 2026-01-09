from pathlib import Path
import socket
import os
import subprocess
import shlex
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Datasets to run (files must exist in ./data)
datasets = [
    'a9a',
    'gisette_scale.bz2',
    # 'w8a',
    # 'real-sim',
    'epsilon_normalized.t.bz2',
    # 'rcv1_train.binary.bz2',
]

DIST_ALGO_NAME = 'ADUCA_TORCH_DIST'

algorithms = [
    # 'CODER',
    # 'CODER_linesearch',
    # 'CODER_normalized',
    # 'CODER_linesearch_normalized',
    # 'PCCM_normalized',
    # 'PCCM',
    # 'GR',
    # 'GR_normalized',
    # 'ADUCA',
    DIST_ALGO_NAME,
]

# Strong convexity toggle for ADUCA_TORCH_DIST
strong_convexity = True

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
    'maxtime': 500_000,
    'targetaccuracy': 1e-12,
    'lambda1': 1e-4,
    'lambda2': 1e-4,
    'mu': 1e-4,
    'block_size': 64,
    'block_size_2': 512,
    'loggingfreq': 10,
}

# Per-dataset overrides (edit as needed)
dataset_params = {
    'a9a': {
        'maxiter': 5_000_000, 
        'lipschitz': 
        # 0.014,
        0.0006, # preconditioned
    },
    'gisette_scale.bz2': {
        'maxiter': 5_000_000, 
        'lipschitz': 
        0.75,
        # 0.01, # preconditioned
    },
    'w8a': {
        'maxiter': 3_000_000, 
        'lipschitz': 
        # 0.007, 
        0.0004, # preconditioned
    },
    'real-sim': {
        'maxiter': 1_000_000, 
        'lipschitz': 
        # 0.0004, 
        0.0002, # preconditioned
    },
    'epsilon_normalized.t.bz2': {
        'maxiter': 5_000_000, 
        'lipschitz': 
        0.002,
        # 0.0007, # preconditioned
    },
    'rcv1_train.binary.bz2': {
        'maxiter': 50_000_000, 
        'lipschitz':
        # 0.001,
        0.0006, # preconditioned
    },
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
        {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2},
        # {'beta': 0.85, 'gamma': 0.3, 'rho': 1.1},
        {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1},
    ],
    DIST_ALGO_NAME: [
        # {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3, 'dist_backend': 'nccl'},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'dist_backend': 'nccl', 'dtype': 'float32', 'mu': 0,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'dist_backend': 'nccl', 'dtype': 'float32', 'mu': 1e-1,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'dist_backend': 'nccl', 'dtype': 'float32', 'mu': 1e-2,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'dist_backend': 'nccl', 'dtype': 'float32', 'mu': 1e-3,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'dist_backend': 'nccl', 'dtype': 'float32', 'mu': 1e-4,},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2, 'dist_backend': 'nccl', 'dtype': 'float32', 'mu': 1e-5,},
        # {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1, 'dist_backend': 'nccl'},
    ],
}

DIST_SCRIPT = 'run_algos_torch_dist.py'
DEFAULT_SCRIPT = 'run_algos.py'
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
    'dist_backend',
    'dtype',
    'dense_threshold',
]
DIST_BOOL_PARAMS = {'use_dense', 'strong_convexity'}

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

max_workers = len(tasks)  # tweak concurrency as needed

def _find_free_port():
    """Pick an open TCP port for torchrun rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_task(ds: str, algo: str, variant_idx=None, variant_overrides=None):
    params = base_params.copy()
    params.update(dataset_params.get(ds, {}))
    if variant_overrides:
        params.update(variant_overrides)

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("NCCL_IB_DISABLE", "1")

    if algo == DIST_ALGO_NAME:
        if strong_convexity:
            params['strong_convexity'] = True
        ### Distributed run with --nproc_per_node=j for using j GPUs
        master_port = _find_free_port()
        cmd = ['torchrun', '--nproc_per_node=8', '--master-port', str(master_port), DIST_SCRIPT, '--dataset', ds]
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
        cmd = ['python', DEFAULT_SCRIPT, '--algo', algo, '--dataset', ds]
        for k, v in params.items():
            cmd += [f'--{k}', str(v)]

    variant_suffix = f"-v{variant_idx}" if variant_idx is not None else ""
    log_path = log_dir / f"{ds}-{algo}{variant_suffix}.log"
    print(f"Launching {ds} | {algo}{variant_suffix}:", ' '.join(shlex.quote(c) for c in cmd))

    # Limit per-process threading to avoid CPU oversubscription when running many tasks in parallel

    with open(log_path, 'w') as logf:
        logf.write('Command: ' + ' '.join(cmd) + '\n\n')
        result = subprocess.run(cmd, cwd='.', env=env, stdout=logf, stderr=subprocess.STDOUT)
    return ds, algo, variant_suffix, result.returncode, log_path


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(run_task, ds, algo, v_idx, variant): (ds, algo, v_idx) for ds, algo, v_idx, variant in tasks}
    for fut in as_completed(futures):
        ds, algo, variant_suffix, rc, log_path = fut.result()
        status = 'OK' if rc == 0 else f'FAILED (rc=' + str(rc) + ')'
        print(f"{ds} | {algo}{variant_suffix} finished: {status}; log: {log_path}")
