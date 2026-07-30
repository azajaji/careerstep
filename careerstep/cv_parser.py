"""Section-aware CV parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
URL_RE = re.compile(r"https?://\S+")
SECTION_HEADERS = [
    "education", "experience", "work experience", "professional experience",
    "skills", "technical skills", "projects", "certifications",
    "summary", "objective", "publications", "awards",
]

# A compact but workable skill vocabulary. Replaced by ESCO/O*NET vocabularies
# inside the experiments — see ``careerstep.skill_gap.SkillVocabulary``.
_DEFAULT_SKILLS = [
    "python", "java", "c++", "javascript", "typescript", "sql", "nosql",
    "react", "node", "flutter", "django", "flask", "fastapi", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "spark", "hadoop", "aws",
    "azure", "gcp", "docker", "kubernetes", "linux", "git", "rest", "graphql",
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "agile", "scrum", "data analysis", "machine learning",
    "deep learning", "nlp", "computer vision", "cybersecurity",
]


@dataclass
class ParsedCV:
    raw: str
    name: Optional[str] = None
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    years_experience: Optional[float] = None

    def as_text(self) -> str:
        """Return canonical text used for downstream embeddings."""
        parts: List[str] = []
        for header in ("summary", "experience", "education", "skills",
                       "projects", "certifications"):
            chunk = self.sections.get(header)
            if chunk:
                parts.append(f"{header.upper()}:\n{chunk}")
        if not parts:
            return self.raw
        return "\n\n".join(parts)


def _split_sections(text: str) -> dict:
    pattern = re.compile(
        r"^\s*(?P<header>" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    sections = {}
    matches = list(pattern.finditer(text))
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        header = m.group("header").lower()
        canonical = "experience" if "experience" in header else header
        sections[canonical] = text[start:end].strip()
    return sections


def _extract_skills(text: str, vocab: Sequence[str] = _DEFAULT_SKILLS) -> List[str]:
    haystack = text.lower()
    found = []
    for skill in vocab:
        if re.search(rf"\b{re.escape(skill.lower())}\b", haystack):
            found.append(skill.lower())
    return sorted(set(found))


def _extract_years(text: str) -> Optional[float]:
    matches = re.findall(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*years?", text.lower())
    if not matches:
        return None
    try:
        return max(float(m) for m in matches)
    except ValueError:
        return None


def parse_cv(text: str, *, skill_vocab: Optional[Sequence[str]] = None) -> ParsedCV:
    cv = ParsedCV(raw=text or "")
    if not text:
        return cv
    cv.emails = EMAIL_RE.findall(text)
    cv.phones = PHONE_RE.findall(text)
    cv.urls = URL_RE.findall(text)
    cv.sections = _split_sections(text)
    cv.skills = _extract_skills(text, skill_vocab or _DEFAULT_SKILLS)
    cv.years_experience = _extract_years(text)

    # Best-effort name guess: first non-empty line that is not contact info.
    for line in (text.splitlines()[:5]):
        line = line.strip()
        if not line:
            continue
        if EMAIL_RE.search(line) or PHONE_RE.search(line) or URL_RE.search(line):
            continue
        if len(line.split()) <= 5 and line.replace(" ", "").isalpha():
            cv.name = line
            break
    return cv


def parse_pdf(path: str | Path) -> ParsedCV:
    """Parse a CV stored as PDF using ``pypdf``."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return parse_cv(text)


def parse_many(texts: Iterable[str]) -> List[ParsedCV]:
    return [parse_cv(t) for t in texts]
