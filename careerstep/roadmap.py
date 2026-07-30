"""Diversity-aware greedy learning-roadmap generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from careerstep.backends import EmbeddingBackend, get_backend


@dataclass
class LearningItem:
    skill: str
    resource: str
    provider: str = ""
    estimated_hours: int = 0
    level: str = "intermediate"
    certification: bool = False

    def as_text(self) -> str:
        bits = [self.skill, self.resource]
        if self.provider:
            bits.append(self.provider)
        if self.level:
            bits.append(self.level)
        return " - ".join(bits)


@dataclass
class Roadmap:
    role: str
    items: List[LearningItem] = field(default_factory=list)

    def as_text_list(self) -> List[str]:
        return [it.as_text() for it in self.items]


class RoadmapGenerator:
    """Greedy diversity-aware roadmap generator over a learning resource bank."""

    def __init__(
        self,
        resource_bank: Sequence[LearningItem],
        *,
        diversity_lambda: float = 0.4,
        backend=None,
    ) -> None:
        self.resource_bank = list(resource_bank)
        self.diversity_lambda = diversity_lambda
        self.backend = backend or EmbeddingBackend()
        self._bank_emb: Optional[np.ndarray] = None

    # -- public ------------------------------------------------------------

    def generate(
        self,
        role: str,
        missing_skills: Sequence[str],
        *,
        max_items: int = 8,
    ) -> Roadmap:
        if not missing_skills or not self.resource_bank:
            return Roadmap(role=role, items=[])
        bank_emb = self._encode_bank()
        gap_emb = self.backend.encode([f"learn {s}" for s in missing_skills])
        relevance = gap_emb @ bank_emb.T  # [#gaps, #resources]
        score = relevance.max(axis=0)     # best fit per resource

        chosen: List[int] = []
        for _ in range(min(max_items, len(self.resource_bank))):
            best_i, best_v = -1, -np.inf
            for i in range(len(self.resource_bank)):
                if i in chosen:
                    continue
                redundancy = 0.0
                if chosen:
                    redundancy = float((bank_emb[chosen] @ bank_emb[i]).max())
                value = score[i] - self.diversity_lambda * redundancy
                if value > best_v:
                    best_v, best_i = value, i
            if best_i < 0:
                break
            chosen.append(best_i)

        return Roadmap(role=role, items=[self.resource_bank[i] for i in chosen])

    # -- helpers -----------------------------------------------------------

    def _encode_bank(self) -> np.ndarray:
        if self._bank_emb is None:
            texts = [it.as_text() for it in self.resource_bank]
            self._bank_emb = self.backend.encode(texts)
        return self._bank_emb
