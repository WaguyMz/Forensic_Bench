#!/usr/bin/env bash
set -euo pipefail
# Start the dedicated ForensicBench Postgres container.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/pkg_paths.sh"
forensicbench_init_paths "${SCRIPT_DIR}"

cd "${PKG_ROOT}"
docker compose up -d forensicbench-postgres

echo "Waiting for Postgres healthcheck..."
for _ in $(seq 1 60); do
  if docker compose exec -T forensicbench-postgres pg_isready -U postgres -d postgres >/dev/null 2>&1; then
    echo "Postgres ready on localhost:55432 (user=postgres, password=forensicbench)"
    exit 0
  fi
  sleep 2
done
echo "Postgres did not become ready in time." >&2
exit 1
