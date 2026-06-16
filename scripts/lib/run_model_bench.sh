# Shared runner that executes ForensicBench across all datasets.
# Source pkg_paths.sh and llm_wait.sh before this file.
#
# Required configuration (set by consumer):
#   MODEL_NAME, BASE_URL, MODEL_SUBDIR, DATASETS, OUT_ROOT
#   TEMPERATURE, TOP_P
#
# Optional:
#   API_KEY (default: dummy), PROVIDER (default: openai_compatible)
#   ENABLE_THINKING, DISABLE_THINKING, PARALLEL_WORKERS, EVALUATE, TOOLS
#   FORENSIC_PARALLEL_BENCH=1 for concurrent datasets

EXTRA_ARGS=()
API_KEY="${API_KEY:-dummy}"
PROVIDER="${PROVIDER:-openai_compatible}"
EVALUATE="${EVALUATE:-1}"
TOOLS="${TOOLS:-}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
STAGGER_SECS="${STAGGER_SECS:-1}"

export FORENSIC_LLM_MAX_RETRIES="${FORENSIC_LLM_MAX_RETRIES:-8}"
export FORENSIC_MODEL_CONTEXT_WINDOW="${FORENSIC_MODEL_CONTEXT_WINDOW:-128000}"
export FORENSIC_LLM_MAX_TOKENS_PER_STEP="${FORENSIC_LLM_MAX_TOKENS_PER_STEP:-16384}"
export FORENSIC_LLM_MAX_TOKENS_PLANNING="${FORENSIC_LLM_MAX_TOKENS_PLANNING:-16384}"
export FORENSIC_SLOT_PLAN_TOKENS="${FORENSIC_SLOT_PLAN_TOKENS:-8000}"
export FORENSIC_SLOT_PAST_TOKENS="${FORENSIC_SLOT_PAST_TOKENS:-8192}"
export FORENSIC_SLOT_SCRATCHPAD_TOKENS="${FORENSIC_SLOT_SCRATCHPAD_TOKENS:-20480}"
export FORENSIC_SLOT_RECENT_TOKENS="${FORENSIC_SLOT_RECENT_TOKENS:-61600}"
export FORENSIC_SLOT_INPUT_TOKENS="${FORENSIC_SLOT_INPUT_TOKENS:-8192}"
export FORENSIC_PACK_SYSTEM_RESERVE_TOKENS="${FORENSIC_PACK_SYSTEM_RESERVE_TOKENS:-8192}"
export FORENSIC_PACK_BUFFER_TOKENS="${FORENSIC_PACK_BUFFER_TOKENS:-2048}"
export FORENSIC_TRUNC_ORIENTATION_MEMO_SYNTHESIS_INPUT_TOKENS="${FORENSIC_TRUNC_ORIENTATION_MEMO_SYNTHESIS_INPUT_TOKENS:-110000}"
export FORENSIC_TRUNC_ORIENTATION_MEMO_SYNTHESIS_OUTPUT_TOKENS="${FORENSIC_TRUNC_ORIENTATION_MEMO_SYNTHESIS_OUTPUT_TOKENS:-16000}"
export FORENSIC_TRUNC_PLANNING_ORIENTATION_PROMPT_TOKENS="${FORENSIC_TRUNC_PLANNING_ORIENTATION_PROMPT_TOKENS:-16000}"
export FORENSIC_TRUNC_ORIENTATION_SUMMARY_STORE_TOKENS="${FORENSIC_TRUNC_ORIENTATION_SUMMARY_STORE_TOKENS:-16000}"

DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-20000000}"

_build_args() {
  local dataset="$1"
  local out_dir="${OUT_ROOT}/${dataset}/${MODEL_SUBDIR}"

  _ARGS=(
    --provider "${PROVIDER}"
    --base-url "${BASE_URL}"
    --api-key "${API_KEY}"
    --model "${MODEL_NAME}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --task "${TASK:-full}"
    --max-tokens "${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"
    --db-name "datasynth_forensic_public__${dataset}"
    --output-dir "${out_dir}"
    --max-parallel-workers "${PARALLEL_WORKERS}"
  )
  [[ "${ENABLE_THINKING:-}" == "1" ]] && _ARGS+=(--enable-thinking)
  [[ "${DISABLE_THINKING:-}" == "1" ]] && _ARGS+=(--disable-thinking)
  [[ "${EVALUATE}" == "1" ]] && _ARGS+=(--evaluate)
  if [[ -n "${TOOLS:-}" ]]; then
    read -ra _t <<< "${TOOLS}"
    _ARGS+=(--tools "${_t[@]}")
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    _ARGS+=("${EXTRA_ARGS[@]}")
  fi
}

_print_banner() {
  local dataset="$1"
  echo "====================================================================="
  echo "MODEL   : ${MODEL_NAME}"
  [[ "${ENABLE_THINKING:-}" == "1" ]] && echo "THINKING: on (temp=${TEMPERATURE})"
  echo "DATASET : ${dataset}"
  echo "DB      : datasynth_forensic_public__${dataset}"
  echo "OUTPUT  : ${OUT_ROOT}/${dataset}/${MODEL_SUBDIR}"
  echo "====================================================================="
}

_run_bench_sequential() {
  forensic_llm_scan_wait_flags_from_argv "$@"
  forensic_llm_wait_until_available
  mkdir -p "${OUT_ROOT}"

  for dataset in "${DATASETS[@]}"; do
    _build_args "${dataset}"
    mkdir -p "${OUT_ROOT}/${dataset}/${MODEL_SUBDIR}"
    _print_banner "${dataset}"
    (cd "${PKG_ROOT}" && python -m researchpkg.forensic_llm.run "${_ARGS[@]}")
  done
}

_run_bench_parallel() {
  forensic_llm_scan_wait_flags_from_argv "$@"
  forensic_llm_wait_until_available
  mkdir -p "${OUT_ROOT}"

  declare -a PIDS=()
  declare -a LOGS=()
  echo "Starting parallel runs for ${MODEL_NAME} on ${#DATASETS[@]} datasets..."

  local LAST_DATASET="${DATASETS[-1]}"

  for dataset in "${DATASETS[@]}"; do
    _build_args "${dataset}"
    local out_dir="${OUT_ROOT}/${dataset}/${MODEL_SUBDIR}"
    mkdir -p "${out_dir}"
    local log="${out_dir}/parallel.log"
    LOGS+=("${log}")

    echo "  [LAUNCH] dataset=${dataset}  log=${log}"
    if [[ "${dataset}" == "${LAST_DATASET}" ]]; then
      (cd "${PKG_ROOT}" && python -m researchpkg.forensic_llm.run "${_ARGS[@]}" 2>&1 | tee "${log}") &
    else
      (cd "${PKG_ROOT}" && python -m researchpkg.forensic_llm.run "${_ARGS[@]}" >"${log}" 2>&1) &
    fi
    PIDS+=($!)
    sleep "${STAGGER_SECS}"
  done

  echo "All ${#PIDS[@]} jobs launched. Waiting..."

  local FAILED=0
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "  [OK]   ${DATASETS[$i]} (pid=${PIDS[$i]})"
    else
      echo "  [FAIL] ${DATASETS[$i]} (pid=${PIDS[$i]}) - see ${LOGS[$i]}"
      FAILED=$((FAILED + 1))
    fi
  done

  echo ""
  echo "======================================================="
  echo "  ${MODEL_NAME} - parallel run complete"
  echo "  Datasets: ${#DATASETS[@]}  |  Failed: ${FAILED}"
  echo "======================================================="
  exit "${FAILED}"
}

_run_bench() {
  case "${FORENSIC_PARALLEL_BENCH:-0}" in
    1 | true | True | TRUE | yes | Yes | YES) _run_bench_parallel "$@" ;;
    *) _run_bench_sequential "$@" ;;
  esac
}
