#!/usr/bin/env bash
# Verify ForensicBench is installable and runnable (smoke test).
#
# Checks: Python deps, Docker Postgres, dataset extract (one sector),
# DB load (one sector), agent dry-run, optional LLM endpoint probe.
#
# Usage:
#   ./scripts/smoke_test.sh
#   ./scripts/smoke_test.sh --with-llm http://localhost:8027/v1 Qwen/Qwen3.6-27B-FP8
#
# A full benchmark run takes hours; this script confirms the stack works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/pkg_paths.sh"
forensicbench_init_paths "${SCRIPT_DIR}"

WITH_LLM=0
LLM_BASE_URL=""
LLM_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-llm)
      WITH_LLM=1
      LLM_BASE_URL="${2:-}"
      LLM_MODEL="${3:-}"
      shift 3
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

pass() { echo "[OK]   $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

SECTOR=energy
ARCHIVE="${PKG_ROOT}/datasets/${SECTOR}.tar.zst"
OUT_DIR="${FORENSIC_ROOT}/experiments/datasets/${SECTOR}"

step "System tools"
command -v python3 >/dev/null || fail "python3 not found"
command -v docker >/dev/null || fail "docker not found"
command -v zstd >/dev/null || fail "zstd not found (needed to extract datasets)"
command -v tar >/dev/null || fail "tar not found"
command -v psql >/dev/null || fail "psql not found (postgresql-client)"
docker compose version >/dev/null 2>&1 || fail "docker compose not found"
pass "python3, docker, zstd, tar, psql"

step "Python package import"
export PYTHONPATH="${PKG_ROOT}:${PYTHONPATH:-}"
python3 -c "from researchpkg.forensic_llm import run; from researchpkg.config import DATASET_DIR" \
  || fail "researchpkg not importable (run: pip install -e .)"
pass "researchpkg imports"

step "Dataset archive"
[[ -f "${ARCHIVE}" ]] || fail "missing ${ARCHIVE}"
pass "${SECTOR}.tar.zst present"

step "Docker Postgres"
cd "${PKG_ROOT}"
"${SCRIPTS_DIR}/datasets/docker_db_up.sh" >/dev/null
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/datasets/docker_db_env.sh"
pass "Postgres on ${FORENSIC_PG_HOST}:${FORENSIC_PG_PORT}"

step "Extract one sector (${SECTOR})"
mkdir -p "${FORENSIC_ROOT}/experiments/datasets"
if [[ ! -d "${OUT_DIR}/forensic_llm" ]]; then
  zstd -d -c "${ARCHIVE}" | tar -C "${FORENSIC_ROOT}/experiments/datasets" -xf -
fi
[[ -f "${OUT_DIR}/forensic_llm/forensic_llm.sql" ]] || fail "extract failed"
pass "extracted to ${OUT_DIR}"

step "Load ${SECTOR} into Postgres"
DB_PUBLIC="datasynth_forensic_public__${SECTOR}"
if PGPASSWORD="${FORENSIC_PG_PASS}" psql -h "${FORENSIC_PG_HOST}" -p "${FORENSIC_PG_PORT}" -U "${FORENSIC_PG_USER}" \
  -d "${DB_PUBLIC}" -tAc "SELECT 1 FROM je_header LIMIT 1" 2>/dev/null | grep -q 1; then
  pass "DB ${DB_PUBLIC} already loaded (skipping reload)"
else
  "${SCRIPTS_DIR}/datasets/populate_psql.sh" \
    --input "${OUT_DIR}" \
    --dataset "${SECTOR}" >/dev/null
  pass "DB ${DB_PUBLIC}"
fi

step "Agent dry-run (DB schema)"
python3 -m researchpkg.forensic_llm.run \
  --dry-run \
  --db-host "${FORENSIC_DB_HOST}" \
  --db-port "${FORENSIC_DB_PORT}" \
  --db-user "${FORENSIC_DB_USER}" \
  --db-password "${FORENSIC_DB_PASSWORD}" \
  --db-name "datasynth_forensic_public__${SECTOR}" \
  | grep -q "je_header" || fail "dry-run did not list tables"
pass "agent sees journal-entry tables"

if [[ "${WITH_LLM}" == "1" ]]; then
  [[ -n "${LLM_BASE_URL}" && -n "${LLM_MODEL}" ]] || fail "--with-llm needs BASE_URL and MODEL"
  step "LLM endpoint probe (${LLM_BASE_URL})"
  # shellcheck disable=SC1091
  source "${SCRIPTS_DIR}/lib/llm_wait.sh"
  BASE_URL="${LLM_BASE_URL}"
  API_KEY=dummy
  MODEL_NAME="${LLM_MODEL}"
  export BASE_URL API_KEY MODEL_NAME
  forensic_llm_probe_models_once || fail "LLM endpoint unreachable or model missing"
  pass "model ${LLM_MODEL} available"
fi

echo
echo "Smoke test passed."
echo "Next: start vLLM, then run the agent (see README Quick start)."
if [[ "${WITH_LLM}" != "1" ]]; then
  echo "Optional: ./scripts/smoke_test.sh --with-llm http://localhost:8027/v1 Qwen/Qwen3.6-27B-FP8"
fi
