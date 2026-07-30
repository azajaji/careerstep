"""CV / job-description ranking over lexical and dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from careerstep.backends import get_backend


@dataclass
class AlignmentResult:
    query_id: int
    ranked_doc_ids: List[int]
    scores: List[float]


class CVJobAligner:
    def __init__(self, backend=None) -> None:
        self.backend = backend or get_backend()

    def rank(
        self,
        cv_texts: Sequence[str],
        jd_texts: Sequence[str],
        top_k: int = 10,
    ) -> List[AlignmentResult]:
        if not cv_texts or not jd_texts:
            return []
        sims = self.backend.similarity_matrix(cv_texts, jd_texts)
        results: List[AlignmentResult] = []
        for qi, row in enumerate(sims):
            order = np.argsort(row)[::-1][:top_k]
            results.append(
                AlignmentResult(
                    query_id=qi,
                    ranked_doc_ids=[int(i) for i in order],
                    scores=[float(row[i]) for i in order],
                )
            )
        return results

    def score_pair(self, cv_text: str, jd_text: str) -> float:
        sim = self.backend.similarity_matrix([cv_text], [jd_text])
        return float(sim[0, 0])
