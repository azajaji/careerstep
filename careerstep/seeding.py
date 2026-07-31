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


def force_offline_backends() -> bool:
    """Remove OPENAI_API_KEY from this process so backends stay deterministic.

    ``ModuleBackends`` picks the language-model path whenever OPENAI_API_KEY is
    set, which changes the interview generator's output and its coverage score.
    The reported benchmarks are the offline path, so a key left in the caller's
    environment must not silently change them. Returns True if one was removed.
    """
    return os.environ.pop("OPENAI_API_KEY", None) is not None


def set_global_seeds(override: int | None = None) -> Dict[str, int]:
    """Fix every RNG and force offline backends. Returns the seeds used.

    Every experiment calls this first, so the reported numbers reproduce
    regardless of what is set in the caller's environment.
    """
    if force_offline_backends():
        print("  [seeding] OPENAI_API_KEY ignored; benchmarks run offline")

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
