"""Path constants for the ForensicBench package."""
from pathlib import Path

HERE = Path(__file__).parent
FORENSIC_ROOT_DIR = HERE / "forensic_llm"
FORENSIC_OUTPUT_DIR = FORENSIC_ROOT_DIR / "forensic_output"
DATASET_DIR = FORENSIC_ROOT_DIR / "experiments" / "datasets"
# Default --dataset-dir for run.py (extract archives into DATASET_DIR first).
SYNTH_DATA_OUTPUT_DIR = DATASET_DIR
