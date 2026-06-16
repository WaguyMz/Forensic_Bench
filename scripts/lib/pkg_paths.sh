# Resolve ForensicBench package paths.
# Source from any script under scripts/<subdir>/.
#
# Optional override:
#   export FORENSICBENCH_ROOT=/path/to/ForensicBench

forensicbench_init_paths() {
  local script_dir="$1"
  if [[ -n "${FORENSICBENCH_ROOT:-}" ]]; then
    PKG_ROOT="${FORENSICBENCH_ROOT}"
  else
    PKG_ROOT="$(cd "${script_dir}/../.." && pwd)"
  fi
  SCRIPTS_DIR="${PKG_ROOT}/scripts"
  FORENSIC_ROOT="${PKG_ROOT}/researchpkg/forensic_llm"
  export PYTHONPATH="${PKG_ROOT}:${PYTHONPATH:-}"
}
