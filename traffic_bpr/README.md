# ADUCA experiments for nonlinear traffic network equilibrium with BPR costs

This folder implements a path-flow traffic assignment variational inequality with nonlinear Bureau of Public Roads (BPR) congestion costs, together with the optimizer family used in the ADUCA comparison: ADUCA, PCCM, CODER, CODER-LineSearch, and GRAAL.

The model is

$$
\begin{aligned}
X_r &= \{x_r\in\mathbb{R}^{P_r}_+ : \mathbf 1^\top x_r = D_r\},\qquad X=\prod_r X_r,\\
f &= A x,\\
t_e(f_e) &= t^0_e\left(1+\alpha_e\left(\frac{f_e}{c_e}\right)^{\beta_e}\right),\\
F(x) &= A^\top t(Ax)+\lambda_{\mathrm{path}}x,\\
\text{find }x^\star\in X\quad &\text{s.t.}\quad \langle F(x^\star),x-x^\star\rangle\ge 0,\qquad \forall x\in X.
\end{aligned}
$$

Here each block is one OD-pair simplex. This is a nonlinear monotone VI because the BPR derivative depends on the current link flow.

## Quick start

```bash
cd aduca_traffic_bpr_experiments
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Small synthetic run; no external data required.
bash scripts/run_tuning.sh
```

The default command writes results to `outputs/<timestamped_run_name>/`. To plot, open `plot.ipynb`, set

```python
RUN_FOLDER = Path("outputs/<timestamped_run_name>")
```

and run the notebook.

## One explicit experiment command

```bash
PYTHONPATH=. python -m src.experiments.run_experiment \
  --dataset-source synthetic_grid \
  --methods aduca,coder,coder_linesearch,graal,pccm \
  --num-iterations 100 \
  --k-paths 5 \
  --max-od-pairs 20 \
  --init-mode random_simplex \
  --seed 0 \
  --output-root outputs \
  --run-name synthetic_grid_demo
```

## TNTP benchmark run

```bash
# Download Sioux Falls, Anaheim, Chicago Sketch, and Braess TNTP files.
python scripts/download_tntp.py --source all

# Run a small Sioux Falls comparison.
PYTHONPATH=. python -m src.experiments.run_experiment \
  --dataset-source tntp_siouxfalls \
  --methods aduca,coder_linesearch,graal \
  --num-iterations 200 \
  --k-paths 5 \
  --max-od-pairs 24 \
  --output-root outputs \
  --run-name siouxfalls_demo
```

You can also let the experiment command download registered TNTP files:

```bash
PYTHONPATH=. python -m src.experiments.run_experiment \
  --dataset-source tntp_anaheim \
  --auto-download \
  --methods aduca,coder_linesearch,graal \
  --num-iterations 100 \
  --k-paths 4 \
  --max-od-pairs 30
```

## Hyperparameter tuning

Edit the environment variables in `scripts/run_tuning.sh`, or override them from the shell:

```bash
DATASET_SOURCES="synthetic_grid tntp_siouxfalls" \
METHODS="aduca coder_linesearch graal" \
AUTO_DOWNLOAD=1 \
NUM_ITERATIONS=200 \
K_PATHS=5 \
MAX_OD_PAIRS=24 \
bash scripts/run_tuning.sh
```

Important grids in the shell script:

```bash
ADUCA_GRID="0.8,0.2,1.2,auto,0.0 0.7,0.05,1.3,auto,0.0"
CODER_LHAT_GRID="auto 0.1 1.0 10.0"
CODER_LS_LHAT0_GRID="auto 0.01 0.1 1.0"
GRAAL_GRID="auto,1.15,0.45 auto,1.3,0.45"
PCCM_STEPSIZE_GRID="auto 1e-4 1e-3 1e-2"
```

The ADUCA entry format is `beta,gamma,rho,a0,mu`. The default ADUCA setting is `0.8,0.2,1.2,auto,0.0`.

`--init-mode` supports `uniform`, `shortest`, and `random_simplex`. The random simplex initializer draws one feasible random point per OD block and is reproducible from `--seed`.

<!-- ## Output structure

A run folder contains one subfolder per method/trial:

```text
outputs/<run_name>/
├── combined_metrics.csv
├── problem_summary.json
├── <trial_name>/
│   ├── config.json
│   ├── metrics.csv
│   └── x_final.npy
└── ...
``` -->

<!-- `metrics.csv` contains, among other columns:

- `relative_gap`: restricted path-set Wardrop relative gap.
- `wardrop_gap`: absolute Wardrop gap.
- `beckmann_potential`: Beckmann potential for BPR costs.
- `operator_evals`: literal operator-evaluation counter.
- `logical_passes`: outer iteration/cycle count.
- `Lk`, `Lhat`: ADUCA local Lipschitz diagnostics when method is ADUCA. -->

<!-- ## Recommended first runs

For debugging:

```bash
DATASET_SOURCES="synthetic_braess synthetic_grid" METHODS="aduca graal pccm" NUM_ITERATIONS=50 bash scripts/run_tuning.sh
```

For the paper-style nonlinear traffic experiment:

```bash
DATASET_SOURCES="tntp_siouxfalls tntp_anaheim" \
METHODS="aduca coder coder_linesearch graal pccm" \
AUTO_DOWNLOAD=1 \
NUM_ITERATIONS=300 \
K_PATHS=5 \
MAX_OD_PAIRS=40 \
bash scripts/run_tuning.sh
```

For a larger stress test:

```bash
DATASET_SOURCES="tntp_chicago_sketch" \
METHODS="aduca coder_linesearch graal" \
AUTO_DOWNLOAD=1 \
NUM_ITERATIONS=200 \
K_PATHS=3 \
MAX_OD_PAIRS=80 \
DEMAND_SCALE=2.0 \
bash scripts/run_tuning.sh
```

Chicago Sketch can be slow because path generation uses k-shortest simple paths. Start with small `MAX_OD_PAIRS` and `K_PATHS`, then scale up. -->
