"""Corpus construction.

Only ``download_onet_work_values`` fetches an external public dataset.
Every other corpus here is generated or hand-authored for evaluation.

Run: ``python -m data.download --all``"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import string
import sys
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from data import CACHE_DIR, DATA_DIR


# Cache helpers
def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(df: pd.DataFrame, name: str) -> Path:
    p = _cache_path(name)
    df.to_parquet(p, index=False)
    return p


def _is_fresh(name: str) -> bool:
    return _cache_path(name).exists()


# Synthetic fallback (small but realistic)
_CATEGORIES = [
    "Data Scientist", "Software Engineer", "DevOps Engineer", "ML Engineer",
    "Frontend Developer", "Backend Developer", "Cybersecurity Analyst",
    "Product Manager", "UX Designer", "Business Analyst",
]

_SKILLS_BY_ROLE = {
    "Data Scientist": ["python", "pandas", "scikit-learn", "sql", "statistics", "machine learning"],
    "Software Engineer": ["python", "java", "git", "rest", "sql", "testing"],
    "DevOps Engineer": ["aws", "docker", "kubernetes", "linux", "ci/cd", "terraform"],
    "ML Engineer": ["python", "pytorch", "tensorflow", "mlops", "docker", "machine learning"],
    "Frontend Developer": ["javascript", "typescript", "react", "css", "html", "accessibility"],
    "Backend Developer": ["python", "node", "sql", "rest", "microservices", "docker"],
    "Cybersecurity Analyst": ["network security", "penetration testing", "siem", "iso 27001"],
    "Product Manager": ["roadmaps", "stakeholder management", "user research", "agile"],
    "UX Designer": ["figma", "user research", "wireframing", "prototyping", "accessibility"],
    "Business Analyst": ["sql", "tableau", "stakeholder management", "process modeling"],
}


def _synth_resume(role: str, rng: random.Random) -> str:
    skills = _SKILLS_BY_ROLE.get(role, ["communication", "teamwork"])
    chosen = ", ".join(rng.sample(skills, k=min(len(skills), 4)))
    years = rng.randint(0, 8)
    return textwrap.dedent(
        f"""
        SUMMARY
        {role} with {years}+ years of experience.

        EXPERIENCE
        - Built and shipped {role.lower()} systems with measurable impact.
        - Collaborated cross-functionally with product, design, and engineering.

        EDUCATION
        BSc Computer Science, 2018-2022.

        SKILLS
        {chosen}
        """
    ).strip()


def _synth_jd(role: str, rng: random.Random) -> str:
    skills = _SKILLS_BY_ROLE.get(role, ["communication"])
    must = ", ".join(rng.sample(skills, k=min(len(skills), 4)))
    return textwrap.dedent(
        f"""
        Position: {role}
        Responsibilities:
        - Deliver high-quality {role.lower()} outcomes.
        - Mentor junior team members.

        Required skills: {must}.
        Nice to have: {rng.choice(skills)}, {rng.choice(skills)}.
        """
    ).strip()


# Resume corpus (Kaggle / Innovatiana)
def download_resumes(force: bool = False) -> Path:
    name = "resumes"
    if _is_fresh(name) and not force:
        return _cache_path(name)

    # 1. Try Kaggle if creds present.
    try:
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

            api = KaggleApi()
            api.authenticate()
            target = CACHE_DIR / "kaggle_resume"
            target.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(
                "snehaanbhawal/resume-dataset",
                path=str(target),
                unzip=True,
                quiet=False,
            )
            csv = next(target.rglob("*.csv"), None)
            if csv:
                df = pd.read_csv(csv)
                # Normalize to the standard schema.
                cols = {c.lower(): c for c in df.columns}
                if "resume_str" in cols and "category" in cols:
                    out = pd.DataFrame({
                        "resume_id": range(len(df)),
                        "text": df[cols["resume_str"]].astype(str),
                        "category": df[cols["category"]].astype(str),
                    })
                    return _save(out, name)
    except Exception as exc:  # noqa: BLE001
        print(f"[resumes] Kaggle download failed ({exc}); falling back to synthetic.")

    # 2. Synthetic fallback.
    rng = random.Random(20260101)
    rows = []
    for i in range(400):
        cat = rng.choice(_CATEGORIES)
        rows.append({"resume_id": i, "text": _synth_resume(cat, rng), "category": cat})
    return _save(pd.DataFrame(rows), name)


# Resume <-> JD pairs (for retrieval/alignment)
def download_resume_jd_pairs(force: bool = False) -> Path:
    name = "resume_jd_pairs"
    if _is_fresh(name) and not force:
        return _cache_path(name)

    rng = random.Random(20260102)
    resumes_path = download_resumes(force=False)
    resumes = pd.read_parquet(resumes_path)

    # Build a small JD corpus and label pairs by category match.
    jds: List[Dict] = []
    jd_id = 0
    for role in _CATEGORIES:
        for _ in range(8):
            jds.append({
                "jd_id": jd_id,
                "title": role,
                "text": _synth_jd(role, rng),
                "required_skills": ",".join(_SKILLS_BY_ROLE[role]),
            })
            jd_id += 1
    jds_df = pd.DataFrame(jds)
    _save(jds_df, "jds")

    pairs: List[Dict] = []
    for _, r in resumes.iterrows():
        same_role = jds_df[jds_df["title"] == r["category"]]
        if same_role.empty:
            continue
        pos = same_role.sample(1, random_state=rng.randint(0, 2**31)).iloc[0]
        neg_pool = jds_df[jds_df["title"] != r["category"]]
        neg = neg_pool.sample(3, random_state=rng.randint(0, 2**31))
        pairs.append({"resume_id": r["resume_id"], "jd_id": int(pos["jd_id"]), "label": 1})
        for _, n in neg.iterrows():
            pairs.append({"resume_id": r["resume_id"], "jd_id": int(n["jd_id"]), "label": 0})
    return _save(pd.DataFrame(pairs), name)


# O*NET / ESCO occupations
def download_occupations(force: bool = False) -> Path:
    name = "occupations"
    if _is_fresh(name) and not force:
        return _cache_path(name)

    # The official O*NET excel + ESCO CSV downloads are large and require
    # interactive license acceptance. We emit a small curated taxonomy here
    # that mirrors the schema, and document the full download path in the
    # reproducibility appendix.
    rows = [{"role": role, "skills": skills} for role, skills in _SKILLS_BY_ROLE.items()]
    df = pd.DataFrame(rows)
    return _save(df, name)


# HR interview question corpora
def download_interview_questions(force: bool = False) -> Path:
    name = "interview_questions"
    if _is_fresh(name) and not force:
        return _cache_path(name)

    # Curated seed of representative questions (kept small but role-balanced
    # so the metric tables can be reproduced offline).
    behavioral = [
        ("Tell me about yourself.", "behavioral"),
        ("Describe a time you led a project under tight deadlines.", "behavioral"),
        ("Tell me about a conflict you resolved on a team.", "behavioral"),
        ("Describe a failure and what you learned.", "behavioral"),
        ("How do you prioritize competing tasks?", "behavioral"),
    ]
    technical = {
        "Data Scientist": [
            "Walk through how you would build a churn prediction model.",
            "How do you handle imbalanced data?",
            "When would you choose XGBoost over a neural network?",
        ],
        "Software Engineer": [
            "How would you design a URL shortener?",
            "Explain the difference between processes and threads.",
            "How would you debug a memory leak in production?",
        ],
        "DevOps Engineer": [
            "Walk through an incident-response runbook you have used.",
            "How do you secure a Kubernetes cluster?",
            "Describe how you set up CI/CD for a microservice.",
        ],
        "ML Engineer": [
            "How do you monitor model drift?",
            "Walk through an MLOps pipeline you built.",
            "How do you decide between batch and online inference?",
        ],
        "Frontend Developer": [
            "How do you make a web app accessible?",
            "Explain React reconciliation.",
            "How would you debug a layout shift?",
        ],
        "Backend Developer": [
            "How would you design a rate limiter?",
            "Compare REST and gRPC.",
            "How do you avoid N+1 queries in an ORM?",
        ],
        "Cybersecurity Analyst": [
            "Walk through a SOC alert triage workflow.",
            "How do you investigate a suspected phishing email?",
            "Explain the principle of least privilege with an example.",
        ],
        "Product Manager": [
            "How do you measure feature success?",
            "Walk through a product trade-off you have made.",
            "How do you write a PRD?",
        ],
        "UX Designer": [
            "How do you design for accessibility?",
            "Walk through a usability study you ran.",
            "How do you balance business and user needs?",
        ],
        "Business Analyst": [
            "How do you translate requirements into stories?",
            "Walk through a process you have improved.",
            "How do you measure project ROI?",
        ],
    }

    rows: List[Dict] = []
    qid = 0
    for q, kind in behavioral:
        for role in _CATEGORIES:
            rows.append({"qid": qid, "question": q, "role": role, "kind": kind})
            qid += 1
    for role, qs in technical.items():
        for q in qs:
            rows.append({"qid": qid, "question": q, "role": role, "kind": "technical"})
            qid += 1
    return _save(pd.DataFrame(rows), name)


# Learning resource bank (for roadmap quality)
def download_learning_resources(force: bool = False) -> Path:
    name = "learning_resources"
    if _is_fresh(name) and not force:
        return _cache_path(name)

    rows: List[Dict] = []
    rid = 0
    providers = ["Coursera", "edX", "Udemy", "LinkedIn Learning", "O'Reilly", "DataCamp"]
    rng = random.Random(20260103)
    for role, skills in _SKILLS_BY_ROLE.items():
        for skill in skills:
            for level in ("beginner", "intermediate", "advanced"):
                rows.append({
                    "resource_id": rid,
                    "skill": skill,
                    "resource": f"{level.capitalize()} {skill} for {role}",
                    "provider": rng.choice(providers),
                    "estimated_hours": rng.choice([8, 16, 24, 40]),
                    "level": level,
                    "certification": rng.random() < 0.4,
                })
                rid += 1
    return _save(pd.DataFrame(rows), name)


# Annotated feedback references (for Experiment 6)
def download_feedback_references(force: bool = False) -> Path:
    name = "feedback_refs"
    if _is_fresh(name) and not force:
        return _cache_path(name)

    rng = random.Random(20260104)
    rows: List[Dict] = []
    for i in range(60):
        role = rng.choice(_CATEGORIES)
        rows.append({
            "case_id": i,
            "role": role,
            "cv_text": _synth_resume(role, rng),
            "reference_feedback": "; ".join([
                f"Add a quantified bullet that mentions {rng.choice(_SKILLS_BY_ROLE[role])}.",
                "Move the Skills section above Experience.",
                "Remove the photograph from the CV.",
                f"Practice STAR-format answers for {role} behavioral questions.",
            ]),
        })
    return _save(pd.DataFrame(rows), name)


# Driver
def download_resumes_ar(force: bool = False) -> Path:
    """Arabic resume seed corpus."""
    name = "resumes_ar"
    if _is_fresh(name) and not force:
        return _cache_path(name)
    from data.seeds_ar import resumes_ar

    return _save(pd.DataFrame(resumes_ar()), name)


def download_resume_jd_pairs_ar(force: bool = False) -> Path:
    """Arabic resume / JD paired retrieval set."""
    name = "resume_jd_pairs_ar"
    if _is_fresh(name) and not force:
        return _cache_path(name)
    from data.seeds_ar import resume_jd_pairs_ar

    return _save(pd.DataFrame(resume_jd_pairs_ar()), name)


# Career positioning: O*NET Work Values + Saudi cyber roles
ONET_WV_URL = (
    # O*NET 28.x text release; the file we want is ``Work Values.txt`` inside
    # the database archive. Refresh against
    # https://www.onetcenter.org/database.html for the current release.
    "https://www.onetcenter.org/dl_files/database/db_28_0_text/Work%20Values.txt"
)


def download_onet_work_values(force: bool = False) -> Path:
    """O*NET Work Values for the catalogue's SOC codes.

    Uses the curated snapshot ``data/onet_work_values.csv``. With
    ``ONET_REFRESH=1`` it instead fetches the live release and reshapes it
    into the same schema, restricted to the SOC codes in use."""
    name = "onet_work_values"
    if _is_fresh(name) and not force:
        return _cache_path(name)

    curated = DATA_DIR / "onet_work_values.csv"
    refresh = os.environ.get("ONET_REFRESH") == "1"

    if refresh:
        try:
            import requests  # local import: only required when refreshing

            resp = requests.get(ONET_WV_URL, timeout=60)
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text), sep="\t")
            # Filter to the Extent scale (Scale ID "EX").
            raw = raw[raw["Scale ID"] == "EX"]
            wide = raw.pivot_table(
                index=["O*NET-SOC Code"],
                columns="Element Name",
                values="Data Value",
                aggfunc="first",
            ).reset_index()
            wide.columns = [
                "onet_soc" if c == "O*NET-SOC Code" else c.lower().replace(" ", "_")
                for c in wide.columns
            ]
            # Restrict to the SOC codes we actually use.
            roles_path = download_saudi_cyber_roles(force=False)
            roles = pd.read_parquet(roles_path)
            wide = wide[wide["onet_soc"].isin(roles["onet_soc"].unique())]
            wide["source"] = "onet_live_refresh"
            return _save(wide, name)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[onet_work_values] live refresh failed ({exc}); "
                f"falling back to curated snapshot."
            )

    if not curated.exists():
        raise FileNotFoundError(f"missing curated snapshot at {curated}")
    df = pd.read_csv(curated)
    return _save(df, name)


def download_saudi_cyber_roles(force: bool = False) -> Path:
    """Curated Saudi cybersecurity role taxonomy."""
    name = "saudi_cyber_roles"
    if _is_fresh(name) and not force:
        return _cache_path(name)
    curated = DATA_DIR / "saudi_cyber_roles.csv"
    if not curated.exists():
        raise FileNotFoundError(f"missing curated role table at {curated}")
    df = pd.read_csv(curated)
    return _save(df, name)


REGISTRY: Dict[str, Callable[[bool], Path]] = {
    "resumes": download_resumes,
    "resume_jd_pairs": download_resume_jd_pairs,
    "occupations": download_occupations,
    "interview_questions": download_interview_questions,
    "learning_resources": download_learning_resources,
    "feedback_references": download_feedback_references,
    "resumes_ar": download_resumes_ar,
    "resume_jd_pairs_ar": download_resume_jd_pairs_ar,
    "onet_work_values": download_onet_work_values,
    "saudi_cyber_roles": download_saudi_cyber_roles,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Khutwa dataset acquisition")
    parser.add_argument("--all", action="store_true", help="download every dataset")
    parser.add_argument("--force", action="store_true", help="ignore cache")
    parser.add_argument("--record-checksums", action="store_true")
    parser.add_argument(
        "datasets", nargs="*", help="optional list of dataset names to download"
    )
    args = parser.parse_args()

    targets = args.datasets or (list(REGISTRY) if args.all else [])
    if not targets:
        parser.print_help()
        sys.exit(1)

    checksums: Dict[str, str] = {}
    for name in targets:
        if name not in REGISTRY:
            print(f"[skip] unknown dataset: {name}", file=sys.stderr)
            continue
        path = REGISTRY[name](args.force)
        if args.record_checksums:
            checksums[name] = _sha256(path)
        print(f"[ok] {name} -> {path}")

    if checksums:
        (CACHE_DIR / "checksums.json").write_text(json.dumps(checksums, indent=2))


if __name__ == "__main__":
    main()
