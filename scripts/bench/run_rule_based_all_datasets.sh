#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/pkg_paths.sh"
forensicbench_init_paths "${SCRIPT_DIR}"

DATASETS=(energy healthcare manufacturing luxurygoods transport)
OUT_ROOT="${FORENSIC_ROOT}/experiments/results"
MODEL_SUBDIR="${MODEL_SUBDIR:-rule_based_oracle}"

mkdir -p "${OUT_ROOT}"
for dataset in "${DATASETS[@]}"; do
  out_dir="${OUT_ROOT}/${dataset}/${MODEL_SUBDIR}"
  mkdir -p "${out_dir}"
  echo "Rule-based baseline: ${dataset} -> ${out_dir}"
  (cd "${PKG_ROOT}" && python -m researchpkg.forensic_llm.run_rule_based \
    --db-name "datasynth_forensic_public__${dataset}" \
    --output-dir "${out_dir}" \
    --evaluate)
done
