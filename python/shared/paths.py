"""Folder layout. Every step keeps its own output files in <step folder>/results."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUICK = bool(os.environ.get("QUICK"))  # QUICK=1 runs a small smoke test into results_quick folders
STEPS = {"data": "01_data_understanding", "prep": "02_preprocessing", "tune": "03_tuning", "final": "04_final_test", "errors": "05_error_analysis", "explorer": "06_explorer", "extra": "07_extra_experiments"}


def results(step):
    folder = ROOT / "python" / STEPS[step] / ("results_quick" if QUICK else "results")
    folder.mkdir(parents=True, exist_ok=True)
    return folder
