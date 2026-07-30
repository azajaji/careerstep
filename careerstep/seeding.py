"""Global seed control, loaded from reproducibility/seeds.txt."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import numpy as np

_SEEDS_FILE = Path(__file__).resolve().parent.parent / "reproducibility" / "seeds.txt"


def load_seeds(path: Path = _SEEDS_FILE) -> Dict[str, int]:
    seeds: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        seeds[key.strip()] = int(value.strip())
    return seeds


def set_global_seeds(override: int | None = None) -> Dict[str, int]:
    """Set every relevant RNG. Returns the seed dict actually used."""
    seeds = load_seeds()
    if override is not None:
        seeds = {k: override for k in seeds}

    random.seed(seeds["python_random_seed"])
    np.random.seed(seeds["numpy_seed"])
    os.environ["PYTHONHASHSEED"] = str(seeds["python_random_seed"])

    try:
        import torch
        torch.manual_seed(seeds["torch_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds["torch_seed"])
    except Exception:
        pass

    return seeds
