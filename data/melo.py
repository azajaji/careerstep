"""MELO benchmark fetcher.

MELO (Retyk et al., 2024) links occupation mentions to the ESCO occupation
taxonomy. It is external to this project, MIT-licensed, and is downloaded on
demand rather than vendored here.

Source: https://github.com/Avature/melo-benchmark
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple

from data import CACHE_DIR

BASE = ("https://raw.githubusercontent.com/Avature/melo-benchmark/main/"
        "data/processed/melo")
FILES = ("queries.tsv", "corpus_elements.tsv", "annotations.tsv")


def fetch(config: str) -> Path:
    """Download one MELO configuration into the cache; return its directory."""
    out = CACHE_DIR / "melo" / config
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = out / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        urllib.request.urlretrieve(f"{BASE}/{config}/{name}", dst)
    return out


def _read_tsv(path: Path) -> List[List[str]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                rows.append(line.split("\t"))
    return rows


def load(config: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Return (queries, corpus, qrels) keyed by MELO identifiers."""
    d = fetch(config)
    queries = {r[0]: r[1] for r in _read_tsv(d / "queries.tsv") if len(r) >= 2}
    corpus = {r[0]: r[1] for r in _read_tsv(d / "corpus_elements.tsv") if len(r) >= 2}
    qrels: Dict[str, Set[str]] = {}
    for r in _read_tsv(d / "annotations.tsv"):
        if len(r) >= 4 and r[3] != "0":
            qrels.setdefault(r[0], set()).add(r[2])
    return queries, corpus, qrels
