#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Populate Postgres DBs from a Forensic Ledger sector export.

Creates two databases per dataset:
  - labelled: keeps fraud-label columns
  - public  : strips fraud-label + procedurally generated text columns (LLM-visible)

Usage:
  populate_psql.sh --input <dataset_outdir> --dataset <dataset_key>

Args:
  --input    Directory that contains forensic_llm/ (e.g. .../datasets/energy)
  --dataset  Dataset key used to name DBs (e.g. energy)

Env (optional):
  FORENSIC_PG_HOST (default: localhost)
  FORENSIC_PG_PORT (default: 5432)
  FORENSIC_PG_USER (default: postgres)
  FORENSIC_PG_PASS (if unset, script will prompt)
EOF
}

INPUT_DIR=""
DATASET_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_DIR="${2:-}"; shift 2 ;;
    --dataset)
      DATASET_KEY="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${INPUT_DIR}" || -z "${DATASET_KEY}" ]]; then
  usage
  exit 2
fi

FORENSIC_DIR="${INPUT_DIR%/}/forensic_llm"
SQL_FILE="${FORENSIC_DIR}/forensic_llm.sql"

if [[ ! -d "${FORENSIC_DIR}" ]]; then
  echo "Missing directory: ${FORENSIC_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SQL_FILE}" ]]; then
  echo "Missing file: ${SQL_FILE}" >&2
  exit 1
fi

PG_USER="${FORENSIC_PG_USER:-postgres}"
PG_HOST="${FORENSIC_PG_HOST:-localhost}"
PG_PORT="${FORENSIC_PG_PORT:-5432}"
PG_PASS="${FORENSIC_PG_PASS:-}"

if [[ -z "${PG_PASS}" ]]; then
  read -r -s -p "Postgres password for user '${PG_USER}' (leave empty for none): " PG_PASS
  echo
fi

export PGPASSWORD="${PG_PASS}"

DB_NAME_LABELLED="datasynth_forensic__${DATASET_KEY}"
DB_NAME_PUBLIC="datasynth_forensic_public__${DATASET_KEY}"

run_psql () {
  psql -v ON_ERROR_STOP=1 -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" "$@"
}

echo "Dataset      : ${DATASET_KEY}"
echo "Input        : ${INPUT_DIR}"
echo "SQL schema   : ${SQL_FILE}"
echo "PG endpoint  : ${PG_USER}@${PG_HOST}:${PG_PORT}"
echo

echo "Dropping and recreating labelled database '${DB_NAME_LABELLED}'..."
run_psql -c "DROP DATABASE IF EXISTS \"${DB_NAME_LABELLED}\";" postgres
run_psql -c "CREATE DATABASE \"${DB_NAME_LABELLED}\";" postgres

echo "Dropping and recreating sanitised database '${DB_NAME_PUBLIC}'..."
run_psql -c "DROP DATABASE IF EXISTS \"${DB_NAME_PUBLIC}\";" postgres
run_psql -c "CREATE DATABASE \"${DB_NAME_PUBLIC}\";" postgres

echo "Loading forensic_llm.sql into labelled database '${DB_NAME_LABELLED}'..."
(cd "${FORENSIC_DIR}" && run_psql -d "${DB_NAME_LABELLED}" -f "forensic_llm.sql")

echo "Loading forensic_llm.sql into sanitised database '${DB_NAME_PUBLIC}'..."
(cd "${FORENSIC_DIR}" && run_psql -d "${DB_NAME_PUBLIC}" -f "forensic_llm.sql")

echo "Stripping fraud label columns from sanitised database '${DB_NAME_PUBLIC}'..."
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE je_header  DROP COLUMN IF EXISTS is_fraud, DROP COLUMN IF EXISTS is_anomaly;"
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE je_line    DROP COLUMN IF EXISTS is_fraud, DROP COLUMN IF EXISTS is_anomaly;"
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE employees  DROP COLUMN IF EXISTS is_fraud_actor;"
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE vendors    DROP COLUMN IF EXISTS is_fraud_actor;"
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE customers  DROP COLUMN IF EXISTS is_fraud_actor;"

echo "Stripping procedurally generated text columns from sanitised database '${DB_NAME_PUBLIC}'..."
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE je_header  DROP COLUMN IF EXISTS header_text;"
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE je_line    DROP COLUMN IF EXISTS line_text;"
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE je_header  DROP COLUMN IF EXISTS je_libelle;"
run_psql -d "${DB_NAME_PUBLIC}" -c "ALTER TABLE je_line    DROP COLUMN IF EXISTS je_libelle;"

echo "Removing ground-truth table from sanitised database '${DB_NAME_PUBLIC}'..."
run_psql -d "${DB_NAME_PUBLIC}" -c "DROP TABLE IF EXISTS public.anomaly_labels CASCADE;"

echo
echo "Done."
echo "  - Labelled DB     : ${DB_NAME_LABELLED}"
echo "  - LLM-visible DB  : ${DB_NAME_PUBLIC}"
