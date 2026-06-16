# Shared helpers: wait until an OpenAI-compatible LLM endpoint is ready.
#
# Usage (from any script under scripts/<sub>/):
#   forensicbench_init_paths "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
#   source "${SCRIPTS_DIR}/lib/llm_wait.sh"
#
# After BASE_URL / API_KEY / (optional) MODEL_NAME are set:
#   forensic_llm_scan_wait_flags_from_argv "$@"
#   forensic_llm_wait_until_available
#
# Enable via:
#   WAIT_FOR_LLM=1 ./your_benchmark.sh
#   ./your_benchmark.sh --wait-for-llm

forensic_llm_models_endpoint_url() {
  local b="${BASE_URL%/}"
  if [[ "${b}" == */v1 ]]; then
    printf '%s/models\n' "${b}"
  else
    printf '%s/v1/models\n' "${b}"
  fi
}

forensic_llm_probe_models_once() {
  local url body
  url="$(forensic_llm_models_endpoint_url)"
  if ! body="$(curl -fsS --connect-timeout 15 --max-time 60 \
    -H "Authorization: Bearer ${API_KEY:-dummy}" \
    "${url}" 2>/dev/null)"; then
    return 1
  fi
  if ! printf '%s' "${body}" | MODEL_NAME="${MODEL_NAME:-}" python3 -c '
import json, os, sys
want = (os.environ.get("MODEL_NAME") or "").strip()
j = json.load(sys.stdin)
data = j.get("data")
if not isinstance(data, list) or len(data) < 1:
    sys.exit(1)
if want:
    ids = [x.get("id") for x in data if isinstance(x, dict)]
    sys.exit(0 if want in ids else 1)
sys.exit(0)
'; then
    return 1
  fi
  return 0
}

forensic_llm_scan_wait_flags_from_argv() {
  local a
  for a in "$@"; do
    case "${a}" in
      --wait-for-llm) WAIT_FOR_LLM=1 ;;
      --wait-for-llm-interval=*) WAIT_FOR_LLM_INTERVAL_SEC="${a#*=}" ;;
    esac
  done
}

forensic_llm_wait_until_available() {
  case "${WAIT_FOR_LLM:-0}" in
    1 | true | True | TRUE | yes | Yes | YES) ;;
    *) return 0 ;;
  esac

  local interval="${WAIT_FOR_LLM_INTERVAL_SEC:-60}"
  local url
  url="$(forensic_llm_models_endpoint_url)"

  echo "[wait-for-llm] Polling ${url} every ${interval}s until the API responds."
  if [[ -n "${MODEL_NAME:-}" ]]; then
    echo "[wait-for-llm] Also requiring model id: ${MODEL_NAME}"
  fi

  while true; do
    if forensic_llm_probe_models_once; then
      echo "[wait-for-llm] Ready; continuing."
      return 0
    fi
    echo "[wait-for-llm] Not available yet; sleeping ${interval}s..."
    sleep "${interval}"
  done
}
