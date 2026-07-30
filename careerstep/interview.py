"""Role-conditioned interview questions and answer scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from careerstep.backends import LLMBackend, get_backend


# Question generation
_QUESTION_TEMPLATES_GENERIC = [
    "Walk me through a {role} project you led from start to finish.",
    "Tell me about a time you handled a difficult stakeholder while doing {role} work.",
    "How do you keep your {role} skills up to date?",
    "Describe a {role} problem you solved that you are most proud of.",
    "What is the most important quality of a successful {role}?",
]

_QUESTION_TEMPLATES_TECHNICAL = [
    "Explain how you would approach {skill} in a production setting.",
    "What are the trade-offs of using {skill} versus a simpler alternative?",
    "Walk through how you debug a failing {skill} pipeline.",
    "How would you teach a junior engineer {skill}?",
    "Describe a real situation where {skill} did not work as expected, and what you did.",
]


@dataclass
class InterviewQuestion:
    text: str
    kind: str = "behavioral"  # behavioral / technical
    role: str = ""
    targeted_skill: Optional[str] = None


class InterviewQuestionGenerator:
    def __init__(self, backend=None) -> None:
        self.backend = backend or get_backend()

    def generate(
        self,
        role: str,
        skills: Sequence[str] = (),
        *,
        n_behavioral: int = 4,
        n_technical: int = 6,
    ) -> List[InterviewQuestion]:
        if isinstance(self.backend, LLMBackend) and self.backend.is_live():
            return self._generate_llm(role, skills, n_behavioral, n_technical)
        return self._generate_template(role, skills, n_behavioral, n_technical)

    # -- template fallback -------------------------------------------------

    @staticmethod
    def _generate_template(
        role: str,
        skills: Sequence[str],
        n_behavioral: int,
        n_technical: int,
    ) -> List[InterviewQuestion]:
        questions: List[InterviewQuestion] = []
        for tpl in _QUESTION_TEMPLATES_GENERIC[:n_behavioral]:
            questions.append(
                InterviewQuestion(text=tpl.format(role=role), kind="behavioral", role=role)
            )
        skills_iter = list(skills) or ["the core technical area"]
        for i in range(n_technical):
            tpl = _QUESTION_TEMPLATES_TECHNICAL[i % len(_QUESTION_TEMPLATES_TECHNICAL)]
            skill = skills_iter[i % len(skills_iter)]
            questions.append(
                InterviewQuestion(
                    text=tpl.format(skill=skill),
                    kind="technical",
                    role=role,
                    targeted_skill=skill,
                )
            )
        return questions

    # -- LLM path ----------------------------------------------------------

    def _generate_llm(
        self,
        role: str,
        skills: Sequence[str],
        n_behavioral: int,
        n_technical: int,
    ) -> List[InterviewQuestion]:
        system = (
            "You are an experienced technical interviewer. Generate concise, "
            "non-repetitive interview questions for the given role. Output one "
            "question per line, with no numbering."
        )
        user = (
            f"Role: {role}\n"
            f"Required skills: {', '.join(skills) or 'unspecified'}\n"
            f"Generate {n_behavioral} behavioral questions, then {n_technical} "
            f"technical questions."
        )
        response = self.backend.chat(system, user)
        lines = [ln.strip("- ").strip() for ln in response.splitlines() if ln.strip()]
        out: List[InterviewQuestion] = []
        for idx, ln in enumerate(lines):
            kind = "behavioral" if idx < n_behavioral else "technical"
            out.append(InterviewQuestion(text=ln, kind=kind, role=role))
        if not out:
            return self._generate_template(role, skills, n_behavioral, n_technical)
        return out


# Answer scoring
@dataclass
class AnswerScore:
    relevance: float
    technical_accuracy: float
    communication: float
    coverage: float
    overall: float
    rubric_text: str = ""

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "technical_accuracy": self.technical_accuracy,
            "communication": self.communication,
            "coverage": self.coverage,
            "overall": self.overall,
            "rubric_text": self.rubric_text,
        }

