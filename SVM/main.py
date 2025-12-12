from pathlib import Path
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

# Datasets to run (files must exist in ./data)
datasets = [
    # 'a9a',
    # 'gisette_scale.bz2',
    # 'w8a',
    # 'real-sim',
    'epsilon_normalized.t.bz2',
    # 'rcv1_train.binary.bz2',
]

algorithms = [
    'CODER',
    'CODER_linesearch',
    'PCCM',
    'GR',
    'ADUCA',
]

# Output directories
output_root = Path('./output')
traj_dir = output_root / 'traj'
log_dir = output_root / 'log'
plot_dir = output_root / 'plot'
for d in (traj_dir, log_dir, plot_dir):
    d.mkdir(parents=True, exist_ok=True)

# Base parameters shared by all runs (overridable per dataset)
base_params = {
    'outputdir': str(output_root),
    'maxtime': 500_000,
    'targetaccuracy': 1e-12,
    'lambda1': 1e-4,
    'lambda2': 1e-4,
    'mu': 0.0,
    'block_size': 64,
    'block_size_2': 1024,
    'loggingfreq': 10,
}

# Per-dataset overrides (edit as needed)
dataset_params = {
    'a9a': {
        'maxiter': 3_000_000, 
        'lipschitz': 
        0.02, 
        # 0.0002, # CODER
    },
    'gisette_scale': {
        'maxiter': 1_000_000, 
        'lipschitz': 
        # 0.89, 
        0.56, # CODER
    },
    'w8a': {
        'maxiter': 3_000_000, 
        'lipschitz': 
        0.05, 
        # 0.00005 # CODER
    },
    'real-sim': {
        'maxiter': 1_000_000, 
        'lipschitz': 
        0.004, 
        # 0.000002 # CODER
    },
    'epsilon_normalized.t.bz2': {
        'maxiter': 50_000, 
        'lipschitz': 
        0.003, 
        # 0.000004 # CODER
    },
    'rcv1_train.binary.bz2': {
        'maxiter': 2_000_000, 
        'lipschitz':
        0.007, 
        # 0.000001 # CODER
    },
}

# Per-algorithm overrides (edit as needed)
algorithm_params = {
    'CODER': {
            'lipschitz': 0.000004,
    },
    'CODER_linesearch': {
            'lipschitz': 0.000004,
    },
    'PCCM': {
            'lipschitz': 0.003,
    },
    'GR': {
            'beta': 0.7
    },
}

# Per-algorithm parameter sets (each dict is one full set of overrides for a run)
algorithm_param_sets = {
    'ADUCA': [
        {'beta': 0.7, 'gamma': 0.05, 'rho': 1.3},
        {'beta': 0.8, 'gamma': 0.2, 'rho': 1.2},
        {'beta': 0.85, 'gamma': 0.3, 'rho': 1.1},
        {'beta': 0.9, 'gamma': 0.3, 'rho': 1.1},
    ],
}

# Build all (dataset, algo, variant) tasks
tasks = []
for ds in datasets:
    for algo in algorithms:
        param_variants = algorithm_param_sets.get(algo)
        if param_variants:
            for idx, variant in enumerate(param_variants, start=1):
                tasks.append((ds, algo, idx, variant))
        else:
            tasks.append((ds, algo, None, None))

max_workers = len(tasks)*len(algorithms)*5  # tweak concurrency as needed


def run_task(ds: str, algo: str, variant_idx=None, variant_overrides=None):
    params = base_params.copy()
    params.update(dataset_params.get(ds, {}))
    params.update(algorithm_params.get(algo, {}))
    if variant_overrides:
        params.update(variant_overrides)
    cmd = ['python', 'run_algos.py', '--algo', algo, '--dataset', ds]
    for k, v in params.items():
        cmd += [f'--{k}', str(v)]
    variant_suffix = f"-v{variant_idx}" if variant_idx is not None else ""
    log_path = log_dir / f"{ds}-{algo}{variant_suffix}.log"
    print(f"Launching {ds} | {algo}{variant_suffix}:", ' '.join(shlex.quote(c) for c in cmd))
    with open(log_path, 'w') as logf:
        logf.write('Command: ' + ' '.join(cmd) + '\n\n')
        result = subprocess.run(cmd, cwd='.', stdout=logf, stderr=subprocess.STDOUT)
    return ds, algo, variant_suffix, result.returncode, log_path


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(run_task, ds, algo, v_idx, variant): (ds, algo, v_idx) for ds, algo, v_idx, variant in tasks}
    for fut in as_completed(futures):
        ds, algo, variant_suffix, rc, log_path = fut.result()
        status = 'OK' if rc == 0 else f'FAILED (rc=' + str(rc) + ')'
        print(f"{ds} | {algo}{variant_suffix} finished: {status}; log: {log_path}")
