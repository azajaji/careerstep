"""Benchmark experiments backing the CareerStep Systems paper's Tables 2-10."""

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
