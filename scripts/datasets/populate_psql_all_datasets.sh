#!/usr/bin/env bash
set -euo pipefail
# Load all extracted Forensic Ledger sectors into PostgreSQL.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/pkg_paths.sh"
forensicbench_init_paths "${SCRIPT_DIR}"

POP_SCRIPT="${SCRIPT_DIR}/populate_psql.sh"
OUT_ROOT="${FORENSIC_ROOT}/experiments/datasets"
SECTORS=(energy healthcare luxurygoods manufacturing transport)

if [[ ! -x "${POP_SCRIPT}" ]]; then
  echo "Expected executable script: ${POP_SCRIPT}" >&2
  exit 1
fi

for sector in "${SECTORS[@]}"; do
  dataset_dir="${OUT_ROOT}/${sector}"
  if [[ ! -d "${dataset_dir}/forensic_llm" ]]; then
    echo "Missing extracted dataset: ${dataset_dir}" >&2
    echo "Run: ${SCRIPT_DIR}/extract_datasets.sh" >&2
    exit 1
  fi
  echo "====================================================================="
  echo "Populating Postgres for dataset: ${sector}"
  echo "Input: ${dataset_dir}"
  echo "====================================================================="
  "${POP_SCRIPT}" --input "${dataset_dir}" --dataset "${sector}"
done

echo
echo "All datasets populated into Postgres."
