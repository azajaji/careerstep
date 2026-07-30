"""Corpus construction and loaders.

Loaders return ``pandas.DataFrame`` with a fixed schema:

- resumes: ``[resume_id, text, category]``
- job descriptions: ``[jd_id, text, title, required_skills]``
- resume/JD pairs: ``[resume_id, jd_id, label]``
- occupations: ``[role, skills]``
- interview questions: ``[qid, question, role, kind]``

Curated inputs live in ``data/``; generated corpora are cached in
``data/cache/`` and are not committed.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
