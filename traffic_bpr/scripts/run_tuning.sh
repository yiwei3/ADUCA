#!/usr/bin/env bash
# Hyperparameter tuning launcher for nonlinear BPR traffic equilibrium experiments.
#
# Usage examples from the repository root:
#
#   bash scripts/run_tuning.sh
#
# Edit the configuration values below to tune datasets, methods, and
# hyperparameters.
#
# Dataset choices supported by the code:
#   synthetic_grid          fast controlled grid, no download needed
#   synthetic_braess        tiny nonlinear Braess-style network, no download needed
#   tntp_braess            TNTP Braess example
#   tntp_siouxfalls        TNTP Sioux Falls, good debugging benchmark
#   tntp_anaheim           TNTP Anaheim, medium benchmark
#   tntp_austin            TNTP Austin AM benchmark
#   tntp_chicago_sketch    TNTP Chicago Sketch, larger benchmark
#   tntp_eastern_massachusetts
#   tntp_winnipeg
#   tntp_winnipeg_asymmetric
#   tntp_barcelona
#   tntp_berlin_center
#   tntp_berlin_friedrichshain
#   tntp_berlin_mitte_center
#   tntp_berlin_mitte_prenzlauerberg_friedrichshain_center
#   tntp_berlin_prenzlauerberg_center
#   tntp_berlin_tiergarten
#   tntp_birmingham
#   tntp_sydney
#   tntp_terrassa_asymmetric
#   tntp_chicago_regional
#   tntp_goldcoast
#   tntp_hessen_asymmetric
#   tntp_munich
#   tntp_philadelphia
#   gmns_local             a local GMNS folder with link.csv and demand.csv

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
if [[ -v PYTHONPATH && -n "${PYTHONPATH}" ]]; then
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${ROOT_DIR}"
fi

# ---------------------------------------------------------------------------
# Output and data-root configuration.
#
# OUTPUT_ROOT   : experiment outputs are written here.
# RUN_NAME      : output subfolder name; empty means traffic_bpr_tuning_timestamp.
# DATA_ROOT     : root containing data/raw and data/processed.
# PATH_CACHE_DIR: optional generated path-set cache directory. Set empty to
#                 disable; default saves reusable paths in data/generated_paths.
# REFRESH_PATH_CACHE:
#                 1 regenerates path caches even if a matching file exists.
# GMNS_DIR      : required only when DATASET_SOURCES contains gmns_local.
# AUTO_DOWNLOAD : 0 disables downloads, 1 downloads registered TNTP files when
#                 missing.
# ---------------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_NAME="${RUN_NAME-}"                       # empty uses traffic_bpr_tuning plus timestamp
DATA_ROOT="${DATA_ROOT:-data}"
PATH_CACHE_DIR="${PATH_CACHE_DIR:-data/generated_paths}"
REFRESH_PATH_CACHE="${REFRESH_PATH_CACHE:-0}"
GMNS_DIR="${GMNS_DIR-}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"

# Choose one or more space-separated data sources.
#
# Allowed DATASET_SOURCES values:
#   synthetic_grid          : generated grid network; no download.
#   synthetic_braess        : generated Braess network; no download.
#   tntp_braess            : TNTP Braess example.
#   tntp_siouxfalls        : TNTP Sioux Falls benchmark.
#   tntp_anaheim           : TNTP Anaheim benchmark.
#   tntp_austin            : TNTP Austin AM benchmark.
#   tntp_chicago_sketch    : TNTP Chicago Sketch benchmark.
#   tntp_eastern_massachusetts
#   tntp_winnipeg
#   tntp_winnipeg_asymmetric
#   tntp_barcelona
#   tntp_berlin_center
#   tntp_berlin_friedrichshain
#   tntp_berlin_mitte_center
#   tntp_berlin_mitte_prenzlauerberg_friedrichshain_center
#   tntp_berlin_prenzlauerberg_center
#   tntp_berlin_tiergarten
#   tntp_birmingham
#   tntp_sydney
#   tntp_terrassa_asymmetric
#   tntp_chicago_regional
#   tntp_goldcoast
#   tntp_hessen_asymmetric
#   tntp_munich
#   tntp_philadelphia
#   gmns_local             : local GMNS folder; set GMNS_DIR.
DATASET_SOURCES="${DATASET_SOURCES:-tntp_anaheim tntp_chicago_sketch}"
read -r -a DATASET_ARRAY <<< "${DATASET_SOURCES}"

# Choose one or more space-separated methods.
#
# Allowed METHODS values:
#   aduca
#   coder
#   coder_linesearch      (aliases accepted: coder_ls, coder-ls)
#   graal
#   pccm
METHODS="${METHODS:-aduca coder coder_linesearch graal pccm}"
read -r -a METHOD_ARRAY <<< "${METHODS}"

# Common problem/path-set controls.
#
# NUM_ITERATIONS      : positive integer number of outer iterations/cycles.
# LOG_EVERY           : positive integer logging interval.
# K_PATHS             : positive integer number of shortest paths per OD pair.
# MAX_OD_PAIRS        : positive integer, or all/none/null/empty for all OD pairs.
# MIN_DEMAND          : nonnegative float; OD demands below this are filtered.
# DEMAND_SCALE        : positive float multiplier for TNTP/GMNS demands.
# OD_STRATEGY         : top | first | random.
# INIT_MODE           : uniform | shortest | random_simplex.
# LAMBDA_MODE         : ones | path_time | sqrt_path_time | inv_demand |
#                       inv_sqrt_demand.
# PATH_REGULARIZATION : nonnegative path-flow Tikhonov regularization.
# SYNTHETIC_*         : synthetic grid/Braess controls; row/col counts are
#                       positive integers and demand scale is positive.
# BPR_*_OVERRIDE      : auto/none/null/empty to use data defaults, or a
#                       nonnegative float override.
# MAX_PATH_HOPS       : none/all/null/empty for unrestricted, or positive int.
# SEED                : integer random seed for random OD selection/synthetic data.
NUM_ITERATIONS="${NUM_ITERATIONS:-1000}"
LOG_EVERY="${LOG_EVERY:-1}"
K_PATHS="${K_PATHS:-50}"
MAX_OD_PAIRS="${MAX_OD_PAIRS:-200}"
MIN_DEMAND="${MIN_DEMAND:-0}"
DEMAND_SCALE="${DEMAND_SCALE:-1.0}"
OD_STRATEGY="${OD_STRATEGY:-top}"
INIT_MODE="${INIT_MODE:-uniform}"
LAMBDA_MODE="${LAMBDA_MODE:-ones}"
PATH_REGULARIZATION="${PATH_REGULARIZATION:-0.0}"
SYNTHETIC_GRID_ROWS="${SYNTHETIC_GRID_ROWS:-4}"
SYNTHETIC_GRID_COLS="${SYNTHETIC_GRID_COLS:-4}"
SYNTHETIC_DEMAND_SCALE="${SYNTHETIC_DEMAND_SCALE:-100.0}"
BPR_ALPHA_OVERRIDE="${BPR_ALPHA_OVERRIDE:-auto}"
BPR_POWER_OVERRIDE="${BPR_POWER_OVERRIDE:-auto}"
MAX_PATH_HOPS="${MAX_PATH_HOPS:-none}"
SEED="${SEED:-0}"

# Method hyperparameters.
#
# ADUCA_BETA/GAMMA/RHO/MU:
#   beta  must be in ((sqrt(5)-1)/2, 1), approximately (0.618034, 1).
#   gamma must be in (0, 1 - 1/(beta * (1 + beta))).
#   rho   must be in (1, 1/beta).
#   mu    is a nonnegative strong-monotonicity estimate; 0.0 is standard.
# ADUCA_MAX_STEPSIZE is a positive upper cap on the adaptive stepsize.
# ADUCA_WARMUP_STEPS controls warm-up; 0 disables warm-up, positive values
# enable that many early ADUCA cycles with capped multiplicative step changes.
# ADUCA_WARMUP_CHANGE_FACTOR is the per-cycle step-size change cap during warm-up.
ADUCA_BETA="${ADUCA_BETA:-0.8}"
ADUCA_GAMMA="${ADUCA_GAMMA:-0.2}"
ADUCA_RHO="${ADUCA_RHO:-1.2}"
ADUCA_MU="${ADUCA_MU:-0.0}"
ADUCA_MAX_STEPSIZE="${ADUCA_MAX_STEPSIZE:-1e8}"
ADUCA_WARMUP_STEPS="${ADUCA_WARMUP_STEPS:-0}"
ADUCA_WARMUP_CHANGE_FACTOR="${ADUCA_WARMUP_CHANGE_FACTOR:-2.0}"

# CODER_LHAT may be auto/none/null/empty to estimate Lhat at the initial point,
# or a positive float. CODER_GAMMA is a float acceleration/regularization
# parameter; 0.0 is the baseline.
CODER_LHAT="${CODER_LHAT:-auto}"
CODER_GAMMA="${CODER_GAMMA:-0.0}"

# CODER-LineSearch initial Lhat. May be auto/none/null/empty or a positive
# float. Small values force more line-search work.
CODER_LS_LHAT0="${CODER_LS_LHAT0:-0.00001}"

# GRAAL settings:
#   a0              : auto/none/null/empty to estimate, or positive float.
#   growth          : positive step-growth multiplier; values > 1 permit growth.
#   lipschitz_coeff : positive local-Lipschitz coefficient.
# GRAAL_PHI must be > 1; the default is the golden ratio.
GRAAL_A0="${GRAAL_A0:-auto}"
GRAAL_GROWTH="${GRAAL_GROWTH:-1.15}"
GRAAL_LIPSCHITZ_COEFF="${GRAAL_LIPSCHITZ_COEFF:-0.45}"
GRAAL_PHI="${GRAAL_PHI:-1.618033988749895}"

# PCCM fixed stepsize. May be auto/none/null/empty to estimate an initial step,
# or a positive float.
PCCM_STEPSIZE="${PCCM_STEPSIZE:-auto}"

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="traffic_bpr_tuning_$(date +%Y%m%d_%H%M%S)"
fi

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --num-iterations "${NUM_ITERATIONS}"
  --log-every "${LOG_EVERY}"
  --k-paths "${K_PATHS}"
  --max-od-pairs "${MAX_OD_PAIRS}"
  --min-demand "${MIN_DEMAND}"
  --demand-scale "${DEMAND_SCALE}"
  --od-strategy "${OD_STRATEGY}"
  --init-mode "${INIT_MODE}"
  --lambda-mode "${LAMBDA_MODE}"
  --path-regularization "${PATH_REGULARIZATION}"
  --synthetic-grid-rows "${SYNTHETIC_GRID_ROWS}"
  --synthetic-grid-cols "${SYNTHETIC_GRID_COLS}"
  --synthetic-demand-scale "${SYNTHETIC_DEMAND_SCALE}"
  --bpr-alpha-override "${BPR_ALPHA_OVERRIDE}"
  --bpr-power-override "${BPR_POWER_OVERRIDE}"
  --max-path-hops "${MAX_PATH_HOPS}"
  --seed "${SEED}"
)

if [[ -n "${PATH_CACHE_DIR}" ]]; then
  COMMON_ARGS+=(--path-cache-dir "${PATH_CACHE_DIR}")
fi
if [[ "${REFRESH_PATH_CACHE}" == "1" ]]; then
  COMMON_ARGS+=(--refresh-path-cache)
fi
if [[ "${AUTO_DOWNLOAD}" == "1" ]]; then
  COMMON_ARGS+=(--auto-download)
fi
if [[ -n "${GMNS_DIR}" ]]; then
  COMMON_ARGS+=(--gmns-dir "${GMNS_DIR}")
fi

run_one() {
  local dataset="$1"
  local method="$2"
  local trial="$3"
  shift 3
  echo "============================================================"
  echo "Dataset: ${dataset} | Method: ${method} | Trial: ${trial}"
  echo "============================================================"
  python -m src.experiments.run_experiment \
    --dataset-source "${dataset}" \
    --methods "${method}" \
    --trial-name "${trial}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

for dataset in "${DATASET_ARRAY[@]}"; do
  for method in "${METHOD_ARRAY[@]}"; do
    path_tag="k${K_PATHS}_od${MAX_OD_PAIRS}_init${INIT_MODE}_lam${LAMBDA_MODE}"
    case "${method}" in
      aduca)
        trial="${dataset}_${path_tag}_aduca_b${ADUCA_BETA}_g${ADUCA_GAMMA}_r${ADUCA_RHO}_mu${ADUCA_MU}_warmup${ADUCA_WARMUP_STEPS}x${ADUCA_WARMUP_CHANGE_FACTOR}"
        run_one "${dataset}" "aduca" "${trial}" \
          --aduca-beta "${ADUCA_BETA}" \
          --aduca-gamma "${ADUCA_GAMMA}" \
          --aduca-rho "${ADUCA_RHO}" \
          --aduca-mu "${ADUCA_MU}" \
          --aduca-max-stepsize "${ADUCA_MAX_STEPSIZE}" \
          --aduca-warmup-steps "${ADUCA_WARMUP_STEPS}" \
          --aduca-warmup-change-factor "${ADUCA_WARMUP_CHANGE_FACTOR}"
        ;;
      coder)
        trial="${dataset}_${path_tag}_coder_lhat${CODER_LHAT}"
        run_one "${dataset}" "coder" "${trial}" \
          --coder-lhat "${CODER_LHAT}" \
          --coder-gamma "${CODER_GAMMA}"
        ;;
      coder_linesearch|coder_ls|coder-ls)
        trial="${dataset}_${path_tag}_coderls_lhat0${CODER_LS_LHAT0}"
        run_one "${dataset}" "coder_linesearch" "${trial}" \
          --coder-ls-lhat0 "${CODER_LS_LHAT0}" \
          --coder-gamma "${CODER_GAMMA}"
        ;;
      graal)
        trial="${dataset}_${path_tag}_graal_a${GRAAL_A0}_grow${GRAAL_GROWTH}_c${GRAAL_LIPSCHITZ_COEFF}"
        run_one "${dataset}" "graal" "${trial}" \
          --graal-a0 "${GRAAL_A0}" \
          --graal-growth "${GRAAL_GROWTH}" \
          --graal-lipschitz-coeff "${GRAAL_LIPSCHITZ_COEFF}" \
          --graal-phi "${GRAAL_PHI}"
        ;;
      pccm)
        trial="${dataset}_${path_tag}_pccm_step${PCCM_STEPSIZE}"
        run_one "${dataset}" "pccm" "${trial}" --pccm-stepsize "${PCCM_STEPSIZE}"
        ;;
      *)
        echo "Unknown method '${method}'." >&2
        exit 1
        ;;
    esac
  done
done

echo "All runs complete. Root output folder: ${OUTPUT_ROOT}/${RUN_NAME}"
