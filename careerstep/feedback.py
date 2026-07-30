"""Feedback aggregation across modules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

ACTION_VERB_RE = re.compile(
    r"\b(add|include|quantify|remove|rewrite|expand|reduce|highlight|clarify|"
    r"emphasize|move|reorder|practice|study|learn|complete|enrol|prepare)\b",
    re.IGNORECASE,
)


@dataclass
class FeedbackBullet:
    text: str

    def is_actionable(self) -> bool:
        return bool(ACTION_VERB_RE.search(self.text)) and len(self.text.split()) >= 3


@dataclass
class FeedbackReport:
    cv_suggestions: List[FeedbackBullet] = field(default_factory=list)
    skill_gap_suggestions: List[FeedbackBullet] = field(default_factory=list)
    interview_suggestions: List[FeedbackBullet] = field(default_factory=list)

    def all_bullets(self) -> List[FeedbackBullet]:
        return self.cv_suggestions + self.skill_gap_suggestions + self.interview_suggestions

    def actionability_rate(self) -> float:
        bullets = self.all_bullets()
        if not bullets:
            return 0.0
        return sum(b.is_actionable() for b in bullets) / len(bullets)

    def to_text(self) -> str:
        lines: List[str] = []
        for label, items in (
            ("CV", self.cv_suggestions),
            ("Skills", self.skill_gap_suggestions),
            ("Interview", self.interview_suggestions),
        ):
            if not items:
                continue
            lines.append(f"## {label}")
            for b in items:
                lines.append(f"- {b.text}")
        return "\n".join(lines)


def bulletize(texts: Sequence[str]) -> List[FeedbackBullet]:
    return [FeedbackBullet(text=t) for t in texts if t.strip()]
