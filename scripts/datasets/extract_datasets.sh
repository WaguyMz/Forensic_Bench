#!/usr/bin/env bash
set -euo pipefail
# Extract bundled Forensic Ledger archives into the agent dataset path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/pkg_paths.sh"
forensicbench_init_paths "${SCRIPT_DIR}"

ARCHIVE_DIR="${PKG_ROOT}/datasets"
OUT_ROOT="${FORENSIC_ROOT}/experiments/datasets"
SECTORS=(energy healthcare luxurygoods manufacturing transport)

if [[ ! -d "${ARCHIVE_DIR}" ]]; then
  echo "Missing archive directory: ${ARCHIVE_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

for sector in "${SECTORS[@]}"; do
  archive="${ARCHIVE_DIR}/${sector}.tar.zst"
  if [[ ! -f "${archive}" ]]; then
    echo "Missing archive: ${archive}" >&2
    exit 1
  fi
  echo "Extracting ${sector} -> ${OUT_ROOT}"
  zstd -d -c "${archive}" | tar -C "${OUT_ROOT}" -xf -
done

echo "All sectors extracted to ${OUT_ROOT}"
