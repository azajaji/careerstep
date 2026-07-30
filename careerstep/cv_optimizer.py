"""Rule-based CV scoring with a per-component breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from careerstep.backends import LLMBackend, get_backend
from careerstep.cv_parser import ParsedCV, parse_cv

REQUIRED_SECTIONS = ("summary", "experience", "education", "skills")
RECOMMENDED_SECTIONS = ("projects", "certifications")


@dataclass
class CVScoreReport:
    overall: float
    components: Dict[str, float] = field(default_factory=dict)
    missing_sections: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "components": self.components,
            "missing_sections": self.missing_sections,
            "missing_keywords": self.missing_keywords,
            "suggestions": self.suggestions,
        }


class CVOptimizer:
    """Computes an ATS-style readiness score and concrete rewrite suggestions."""

    def __init__(self, backend=None) -> None:
        self.backend = backend or get_backend()

    # -- scoring -----------------------------------------------------------

    def score(self, cv: ParsedCV, target_keywords: Optional[List[str]] = None) -> CVScoreReport:
        target_keywords = [k.lower() for k in (target_keywords or [])]
        components: Dict[str, float] = {}

        # Structural completeness
        missing_required = [s for s in REQUIRED_SECTIONS if s not in cv.sections]
        missing_recommended = [s for s in RECOMMENDED_SECTIONS if s not in cv.sections]
        structure = 1.0 - (
            0.18 * len(missing_required) + 0.06 * len(missing_recommended)
        )
        components["structure"] = max(0.0, structure)

        # Contact completeness
        contact = 0.0
        if cv.emails:
            contact += 0.5
        if cv.phones:
            contact += 0.3
        if cv.urls:
            contact += 0.2
        components["contact"] = min(contact, 1.0)

        # Keyword coverage
        if target_keywords:
            present = sum(1 for k in target_keywords if k in cv.raw.lower())
            components["keyword_coverage"] = present / len(target_keywords)
            missing_keywords = [k for k in target_keywords if k not in cv.raw.lower()]
        else:
            components["keyword_coverage"] = 0.5  # neutral when no JD provided
            missing_keywords = []

        # Length
        word_count = len(cv.raw.split())
        if word_count < 200:
            length = word_count / 200.0
        elif word_count <= 900:
            length = 1.0
        else:
            length = max(0.4, 1.0 - (word_count - 900) / 900.0)
        components["length"] = length

        # Years-of-experience signal (rewards clear quantification)
        components["experience_signal"] = 1.0 if cv.years_experience is not None else 0.4

        overall = (
            0.35 * components["structure"]
            + 0.15 * components["contact"]
            + 0.30 * components["keyword_coverage"]
            + 0.10 * components["length"]
            + 0.10 * components["experience_signal"]
        )

        report = CVScoreReport(
            overall=round(overall, 4),
            components={k: round(v, 4) for k, v in components.items()},
            missing_sections=missing_required + missing_recommended,
            missing_keywords=missing_keywords[:20],
        )
        report.suggestions = self._heuristic_suggestions(report)
        return report

    # -- suggestions -------------------------------------------------------

    @staticmethod
    def _heuristic_suggestions(report: CVScoreReport) -> List[str]:
        out: List[str] = []
        for section in report.missing_sections:
            out.append(f"Add a `{section}` section.")
        if report.components.get("contact", 1.0) < 0.8:
            out.append("Include a professional email and a phone number at the top of the CV.")
        if report.components.get("length", 1.0) < 0.6:
            out.append(
                "Extend the CV with quantified bullet points (impact, scale, metrics)."
            )
        if report.missing_keywords:
            preview = ", ".join(report.missing_keywords[:6])
            out.append(
                f"Incorporate role-relevant keywords (e.g., {preview}) into the Skills and "
                "Experience sections."
            )
        if report.components.get("experience_signal", 1.0) < 0.7:
            out.append("State the number of years of experience explicitly (e.g., '3+ years ...').")
        return out

    # -- LLM rewrite (optional) -------------------------------------------

    def suggest_rewrite(self, cv_text: str, jd_text: str) -> str:
        if not isinstance(self.backend, LLMBackend) or not self.backend.is_live():
            cv = parse_cv(cv_text)
            report = self.score(cv, target_keywords=_extract_jd_keywords(jd_text))
            return "\n".join(f"- {s}" for s in report.suggestions)
        system = (
            "You are a careful ATS-aware CV reviewer. Rewrite or suggest specific edits "
            "to the user's CV so it aligns better with the target job description. Output "
            "structured bullet points, no fluff."
        )
        user = f"JOB DESCRIPTION:\n{jd_text}\n\nCV:\n{cv_text}\n\nSuggested edits:"
        return self.backend.chat(system, user)


def _extract_jd_keywords(jd_text: str, top_k: int = 20) -> List[str]:
    """Quick keyword extraction from a JD via TF-IDF over its sentences."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentences = [s for s in jd_text.split(".") if s.strip()]
    if not sentences:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=200, stop_words="english")
    matrix = vec.fit_transform(sentences)
    scores = matrix.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    order = scores.argsort()[::-1][:top_k]
    return [terms[i] for i in order]
