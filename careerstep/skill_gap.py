"""Multi-label skill-gap matching in lexical and semantic modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from careerstep.backends import EmbeddingBackend, get_backend


@dataclass
class SkillVocabulary:
    """A flat skill taxonomy keyed by canonical name with synonyms."""

    canonical: List[str] = field(default_factory=list)
    synonyms: Dict[str, List[str]] = field(default_factory=dict)

    def all_terms(self) -> List[str]:
        terms: List[str] = []
        for c in self.canonical:
            terms.append(c)
            terms.extend(self.synonyms.get(c, []))
        return terms

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, Sequence[str]]]) -> "SkillVocabulary":
        canonical: List[str] = []
        synonyms: Dict[str, List[str]] = {}
        for c, syns in pairs:
            canonical.append(c)
            synonyms[c] = list(syns)
        return cls(canonical=canonical, synonyms=synonyms)


@dataclass
class SkillGapReport:
    role: str
    required: List[str]
    covered: List[str]
    missing: List[str]
    coverage: float
    jaccard: float

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "required": self.required,
            "covered": self.covered,
            "missing": self.missing,
            "coverage": self.coverage,
            "jaccard": self.jaccard,
        }


class SkillGapDetector:
    """Detect missing skills relative to a role's required competencies."""

    def __init__(
        self,
        vocabulary: SkillVocabulary,
        *,
        mode: str = "semantic",
        threshold: float = 0.55,
        backend=None,
    ) -> None:
        self.vocabulary = vocabulary
        self.mode = mode
        self.threshold = threshold
        if mode == "semantic":
            self.backend = backend or EmbeddingBackend()
        else:
            self.backend = backend or get_backend("lexical")

    # -- public API --------------------------------------------------------

    def detect(self, cv_text: str, role: str, required_skills: Sequence[str]) -> SkillGapReport:
        covered = self._covered(cv_text, required_skills)
        missing = [s for s in required_skills if s not in covered]
        jaccard = (
            len(covered) / len(set(required_skills) | set(covered))
            if (required_skills or covered)
            else 0.0
        )
        coverage = len(covered) / max(len(required_skills), 1)
        return SkillGapReport(
            role=role,
            required=list(required_skills),
            covered=covered,
            missing=missing,
            coverage=round(coverage, 4),
            jaccard=round(jaccard, 4),
        )

    # -- matching ----------------------------------------------------------

    def _covered(self, cv_text: str, required_skills: Sequence[str]) -> List[str]:
        text = cv_text.lower()
        if self.mode in ("exact", "lemma"):
            return [s for s in required_skills if self._lexical_hit(text, s)]
        return self._semantic_hits(cv_text, required_skills)

    def _lexical_hit(self, haystack: str, skill: str) -> bool:
        synonyms = self.vocabulary.synonyms.get(skill, []) + [skill]
        return any(s.lower() in haystack for s in synonyms)

    def _semantic_hits(self, cv_text: str, required_skills: Sequence[str]) -> List[str]:
        # Compare each required skill (+ its synonyms) against CV sentences.
        sentences = [s.strip() for s in cv_text.split(".") if s.strip()]
        if not sentences:
            return []
        cv_emb = self.backend.encode(sentences)
        covered: List[str] = []
        for skill in required_skills:
            phrases = [skill] + self.vocabulary.synonyms.get(skill, [])
            sk_emb = self.backend.encode(phrases)
            sim = cv_emb @ sk_emb.T
            if sim.size and sim.max() >= self.threshold:
                covered.append(skill)
        return covered

    # -- evaluation hook ---------------------------------------------------

    def predict_set(self, cv_text: str, candidate_skills: Sequence[str]) -> List[str]:
        """Return the predicted set of skills present in the CV.

        Used by Experiment 3 (skill-gap precision/recall) where the predicted
        set is compared against an annotated ground-truth skill list.
        """
        covered = self._covered(cv_text, candidate_skills)
        return covered
