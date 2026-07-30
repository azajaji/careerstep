"""Corpus construction and loaders.

Curated inputs live in ``data/``; generated corpora are cached in
``data/cache/`` and are not committed."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
