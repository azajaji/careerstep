"""Baseline roadmap generators."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

from careerstep.roadmap import LearningItem


def _tokens(text: str) -> set:
    return {t for t in text.lower().split() if t}


@dataclass
class RandomRoadmap:
    seed: int = 20260101

    def generate(
        self,
        *,
        role: str,
        missing_skills: Sequence[str],
        resource_bank: Sequence[LearningItem],
        k: int = 8,
    ) -> List[LearningItem]:
        rng = random.Random(self.seed + hash(role) % 10_000)
        if not resource_bank:
            return []
        return rng.sample(list(resource_bank), k=min(k, len(resource_bank)))


@dataclass
class PopularityRoadmap:
    def generate(
        self,
        *,
        role: str,
        missing_skills: Sequence[str],
        resource_bank: Sequence[LearningItem],
        k: int = 8,
    ) -> List[LearningItem]:
        if not resource_bank:
            return []
        counter: Counter = Counter()
        for item in resource_bank:
            counter[item.skill.lower()] += 1
        ranked = sorted(
            resource_bank,
            key=lambda it: (-counter[it.skill.lower()], it.skill),
        )
        return ranked[:k]


@dataclass
class LexicalRoadmap:
    """Per-gap lexical lookup baseline."""

    def generate(
        self,
        *,
        role: str,
        missing_skills: Sequence[str],
        resource_bank: Sequence[LearningItem],
        k: int = 8,
    ) -> List[LearningItem]:
        if not resource_bank or not missing_skills:
            return list(resource_bank[:k])

        chosen: List[LearningItem] = []
        used: set = set()
        for skill in missing_skills:
            skill_tokens = _tokens(skill)
            best = None
            best_score = -1.0
            for idx, item in enumerate(resource_bank):
                if idx in used:
                    continue
                cand_tokens = _tokens(item.skill) | _tokens(item.resource)
                if not cand_tokens:
                    continue
                jacc = len(skill_tokens & cand_tokens) / max(
                    len(skill_tokens | cand_tokens), 1
                )
                if jacc > best_score:
                    best, best_score = idx, jacc
            if best is not None:
                chosen.append(resource_bank[best])
                used.add(best)
            if len(chosen) == k:
                break
        # Fill out to k with un-used items if necessary.
        if len(chosen) < k:
            for idx, item in enumerate(resource_bank):
                if idx in used:
                    continue
                chosen.append(item)
                if len(chosen) == k:
                    break
        return chosen
