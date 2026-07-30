"""DataFrame loaders over the cached corpora."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pandas as pd

from data import CACHE_DIR
from data import download as dl


def _load(name: str, downloader) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.parquet"
    if not path.exists():
        downloader(False)
    return pd.read_parquet(path)


def load_resumes() -> pd.DataFrame:
    return _load("resumes", dl.download_resumes)


def load_jds() -> pd.DataFrame:
    if not (CACHE_DIR / "jds.parquet").exists():
        dl.download_resume_jd_pairs(False)
    return pd.read_parquet(CACHE_DIR / "jds.parquet")


def load_resume_jd_pairs() -> pd.DataFrame:
    return _load("resume_jd_pairs", dl.download_resume_jd_pairs)


def load_occupations() -> pd.DataFrame:
    df = _load("occupations", dl.download_occupations)
    # The ``skills`` column is stored as a Python list; pandas/parquet may
    # round-trip it as a numpy array. Normalise to list[str].
    df["skills"] = df["skills"].apply(lambda v: list(v) if not isinstance(v, list) else v)
    return df


def load_interview_questions() -> pd.DataFrame:
    return _load("interview_questions", dl.download_interview_questions)


def load_learning_resources() -> pd.DataFrame:
    return _load("learning_resources", dl.download_learning_resources)


def load_feedback_references() -> pd.DataFrame:
    return _load("feedback_refs", dl.download_feedback_references)


def load_resume_jd_pairs_ar() -> pd.DataFrame:
    return _load("resume_jd_pairs_ar", dl.download_resume_jd_pairs_ar)


# -- career positioning (CSMQ + O*NET WV + Saudi cyber roles) -------------


def load_csmq_questionnaire():
    """Return the parsed CSMQ questionnaire (25 items, 5 orientations)."""
    from careerstep.career_positioning import CSMQQuestionnaire
    from data import DATA_DIR

    return CSMQQuestionnaire.from_path(DATA_DIR / "csmq_items.json")


def load_onet_work_values() -> pd.DataFrame:
    """O*NET Work Values for cyber-relevant SOC codes (6-dim Extent scores)."""
    return _load("onet_work_values", dl.download_onet_work_values)


def load_saudi_cyber_roles() -> pd.DataFrame:
    """Saudi cybersecurity role taxonomy with O*NET-SOC + sector tags."""
    return _load("saudi_cyber_roles", dl.download_saudi_cyber_roles)


# -- convenience -----------------------------------------------------------


def role_to_skills() -> dict[str, List[str]]:
    df = load_occupations()
    return {row["role"]: list(row["skills"]) for _, row in df.iterrows()}
