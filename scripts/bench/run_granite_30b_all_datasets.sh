#!/usr/bin/env bash
set -euo pipefail
# ForensicBench paper model: Granite-30B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/pkg_paths.sh"
forensicbench_init_paths "${SCRIPT_DIR}"
source "${SCRIPTS_DIR}/lib/llm_wait.sh"
source "${SCRIPTS_DIR}/lib/run_model_bench.sh"

DATASETS=(energy healthcare manufacturing luxurygoods transport)
OUT_ROOT="${FORENSIC_ROOT}/experiments/results"

MODEL_NAME="${MODEL_NAME:-ibm-granite/granite-4.1-30b-fp8}"
BASE_URL="${BASE_URL:-http://localhost:8030/v1}"
API_KEY="${API_KEY:-dummy}"
MODEL_SUBDIR="${MODEL_SUBDIR:-granite_30b}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"

_run_bench "$@"
