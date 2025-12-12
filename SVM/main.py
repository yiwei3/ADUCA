from pathlib import Path
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

# Datasets to run (files must exist in ./data)
datasets = [
    # 'a9a',
    # 'gisette_scale.bz2',
    'w8a',
    # 'real-sim',
    # 'epsilon_normalized.t.bz2',
    # 'rcv1_train.binary.bz2',
]

algorithms = [
    # 'CODER',
    # 'PCCM',
    # 'CODER_linesearch',
    # 'GR',
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
    'maxiter': 3_000_000,
    'maxtime': 500_000,
    'targetaccuracy': 1e-12,
    'lambda1': 1e-4,
    'lambda2': 1e-4,
    'mu': 0.0,
    'beta': 
    # 0.9,
    # 0.95, 
    # 0.8,
    0.7,
    'gamma': 
    # 0.3,
    # 0.43,
    # 0.2,
    0.1,
    'rho': 
    # 1.1,
    # 1.05,
    # 1.2,
    1.3,
    'block_size': 64,
    'block_size_2': 512,
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
        0.89, 
        # 0.56,
    },
    'w8a': {
        'maxiter': 3_000_000, 
        'lipschitz': 
        0.05, 
        # 0.00005
    },
    'real-sim': {
        'maxiter': 3_000_000, 
        'lipschitz': 
        0.004, 
        # 0.000002
    },
    'epsilon_normalized.t.bz2': {
        'maxiter': 3_000_000, 
        'lipschitz': 
        0.003, 
        # 0.000004
    },
    'rcv1_train.binary.bz2': {
        'maxiter': 3_000_000, 
        'lipschitz':
        0.007, 
        # 0.000001
    },
}

# Build all (dataset, algo) tasks
tasks = [(ds, algo) for ds in datasets for algo in algorithms]
max_workers = len(tasks)*len(algorithms)  # tweak concurrency as needed


def run_task(ds: str, algo: str):
    params = base_params.copy()
    params.update(dataset_params.get(ds, {}))
    cmd = ['python', 'run_algos.py', '--algo', algo, '--dataset', ds]
    for k, v in params.items():
        cmd += [f'--{k}', str(v)]
    log_path = log_dir / f"{ds}-{algo}.log"
    print(f"Launching {ds} | {algo}:", ' '.join(shlex.quote(c) for c in cmd))
    with open(log_path, 'w') as logf:
        logf.write('Command: ' + ' '.join(cmd) + '\n\n')
        result = subprocess.run(cmd, cwd='.', stdout=logf, stderr=subprocess.STDOUT)
    return ds, algo, result.returncode, log_path


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(run_task, ds, algo): (ds, algo) for ds, algo in tasks}
    for fut in as_completed(futures):
        ds, algo, rc, log_path = fut.result()
        status = 'OK' if rc == 0 else f'FAILED (rc=' + str(rc) + ')'
        print(f"{ds} | {algo} finished: {status}; log: {log_path}")
