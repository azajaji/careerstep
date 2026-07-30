"""Career positioning: CSMQ responses -> ranked role recommendations.

A 25-item Likert questionnaire is scored into a five-dimensional orientation
profile. Each role is anchored on O*NET Work Values (six dimensions) and
projected into the same five-dimensional space, then ranked by cosine
similarity to the respondent's profile.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ORIENTATIONS: Tuple[str, ...] = (
    "getting_ahead",
    "getting_secure",
    "getting_free",
    "getting_high",
    "getting_balanced",
)

ONET_WORK_VALUES: Tuple[str, ...] = (
    "achievement",
    "independence",
    "recognition",
    "relationships",
    "support",
    "working_conditions",
)

# Linear projection matrix W (5 x 6) mapping a role's O*NET work-value
# vector to its CSMQ orientation centroid. Each row is L1-normalized, so the
# projection stays on the input scale. Row weights follow the Derr typology;
# the derivation is given in the paper.
_PROJECTION = np.array(
    [
        # ach,  ind,   rec,   rel,   sup,   wc
        [0.60, 0.00, 0.40, 0.00, 0.00, 0.00],  # getting_ahead
        [0.00, -0.20, 0.00, 0.00, 0.50, 0.30],  # getting_secure
        [0.00, 0.80, 0.00, 0.00, 0.00, 0.20],  # getting_free
        [0.50, 0.30, 0.00, 0.00, 0.00, 0.20],  # getting_high
        [0.00, 0.00, 0.00, 0.40, 0.20, 0.40],  # getting_balanced
    ],
    dtype=float,
)


# Data classes
@dataclass(frozen=True)
class CSMQItem:
    item_id: str
    orientation: str
    reverse: bool
    en: str
    ar: str

    @classmethod
    def from_json(cls, obj: Mapping) -> "CSMQItem":
        return cls(
            item_id=obj["id"],
            orientation=obj["orientation"],
            reverse=bool(obj["reverse"]),
            en=obj["en"],
            ar=obj["ar"],
        )


@dataclass(frozen=True)
class CSMQQuestionnaire:
    items: Tuple[CSMQItem, ...]
    scale_min: int = 1
    scale_max: int = 5

    @classmethod
    def from_path(cls, path: Path) -> "CSMQQuestionnaire":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        items = tuple(CSMQItem.from_json(it) for it in payload["items"])
        return cls(
            items=items,
            scale_min=int(payload["scale"]["min"]),
            scale_max=int(payload["scale"]["max"]),
        )

    def items_by_orientation(self) -> Dict[str, List[CSMQItem]]:
        out: Dict[str, List[CSMQItem]] = {o: [] for o in ORIENTATIONS}
        for it in self.items:
            out[it.orientation].append(it)
        return out


@dataclass(frozen=True)
class OrientationProfile:
    """A respondent's five-dimensional CSMQ profile (0-1 normalized)."""

    scores: Dict[str, float]

    def as_vector(self) -> np.ndarray:
        return np.array([self.scores[o] for o in ORIENTATIONS], dtype=float)

    def dominant(self) -> str:
        return max(self.scores, key=self.scores.get)

    def secondary(self) -> str:
        ordered = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[1][0]


@dataclass(frozen=True)
class Recommendation:
    role_id: str
    title_en: str
    title_ar: str
    specialty: str
    seniority: str
    sector_tag: str
    vision_2030_anchor: str
    employer_examples: str
    nca_alignment: str
    onet_soc: str
    score: float
    role_centroid: Dict[str, float]
    dominant_orientation: str
    rationale: str


# Scoring
def score_csmq(
    responses: Mapping[str, int],
    questionnaire: CSMQQuestionnaire,
    *,
    strict: bool = True,
) -> OrientationProfile:
    """Convert raw Likert responses to a five-dim orientation profile.

    Each item is reverse-scored where flagged, then normalized to [0, 1] by
    ``(value - min) / (max - min)``. An orientation score is the arithmetic
    mean of its five items. Missing items raise under ``strict``, otherwise
    take the scale midpoint.
    """
    lo, hi = questionnaire.scale_min, questionnaire.scale_max
    mid = (lo + hi) / 2.0
    by_orientation: Dict[str, List[float]] = {o: [] for o in ORIENTATIONS}

    for item in questionnaire.items:
        raw = responses.get(item.item_id)
        if raw is None:
            if strict:
                raise KeyError(f"missing response for {item.item_id}")
            raw = mid
        if not (lo <= raw <= hi):
            raise ValueError(
                f"response for {item.item_id} = {raw!r} outside [{lo}, {hi}]"
            )
        adjusted = (lo + hi) - raw if item.reverse else raw
        normalized = (adjusted - lo) / (hi - lo)
        by_orientation[item.orientation].append(normalized)

    scores = {o: float(np.mean(vals)) if vals else 0.5 for o, vals in by_orientation.items()}
    return OrientationProfile(scores=scores)


# O*NET WV -> CSMQ projection
def project_work_values_to_csmq(
    work_values: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    onet_scale_max: float = 7.0,
) -> OrientationProfile:
    """Project a six-dim O*NET WV vector onto the five-dim CSMQ space.

    Output values are clipped to [0, 1] so they live on the same scale
    as :func:`score_csmq`. This makes user profile and role centroid
    directly comparable.
    """
    if isinstance(work_values, Mapping):
        vec = np.array([work_values[w] for w in ONET_WORK_VALUES], dtype=float)
    else:
        vec = np.asarray(work_values, dtype=float)
    if vec.shape != (6,):
        raise ValueError(f"expected 6-dim O*NET WV vector, got shape {vec.shape}")

    # Map to 0..1 first (O*NET Extent ratings are on a 0..7 scale).
    normalized = vec / float(onet_scale_max)

    # Apply projection. Some rows have negative weights (e.g. -0.2 on
    # Independence for getting_secure); we clip to [0,1] after.
    projected = _PROJECTION @ normalized
    projected = np.clip(projected, 0.0, 1.0)

    return OrientationProfile(scores={o: float(v) for o, v in zip(ORIENTATIONS, projected)})


# Role recommender
def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


@dataclass
class RoleRecommender:
    """Rank roles by cosine similarity to a CSMQ orientation profile.

    A CSMQ centroid is precomputed per role by projecting its O*NET Work
    Values vector. ``ml_index`` exposes the same centroids as a fitted
    ``NearestNeighbors`` index."""

    roles_df: pd.DataFrame
    wv_df: pd.DataFrame
    centroids: np.ndarray = field(init=False)
    centroid_df: pd.DataFrame = field(init=False)
    ml_index: NearestNeighbors = field(init=False)

    def __post_init__(self) -> None:
        joined = self.roles_df.merge(
            self.wv_df.add_prefix("wv_"),
            left_on="onet_soc",
            right_on="wv_onet_soc",
            how="left",
            validate="m:1",
        )
        missing = joined[joined["wv_achievement"].isna()]["onet_soc"].unique()
        if len(missing):
            raise ValueError(
                "O*NET Work Values missing for SOC codes: " + ", ".join(missing)
            )

        wv_cols = [f"wv_{w}" for w in ONET_WORK_VALUES]
        wv_matrix = joined[wv_cols].to_numpy(dtype=float) / 7.0

        # Centroids = (n_roles, 5)
        centroids = wv_matrix @ _PROJECTION.T
        centroids = np.clip(centroids, 0.0, 1.0)

        self.centroids = centroids
        self.centroid_df = pd.DataFrame(centroids, columns=list(ORIENTATIONS))
        self.centroid_df["role_id"] = joined["role_id"].values
        self.centroid_df = self.centroid_df.set_index("role_id")

        # Keep the joined metadata accessible.
        self._meta = joined.set_index("role_id")

        # ML index over the centroids. ``metric='cosine'`` gives the
        # same ordering as our cosine ranker but exposes a sklearn API
        # so the same fit can drive UMAP/clustering downstream.
        self.ml_index = NearestNeighbors(
            n_neighbors=min(10, len(centroids)),
            metric="cosine",
            algorithm="brute",
        ).fit(centroids)

    # -- public API ---------------------------------------------------------

    def recommend(
        self,
        profile: OrientationProfile,
        *,
        top_k: int = 5,
        sector_tag: Optional[str] = None,
        seniority: Optional[Iterable[str]] = None,
        method: str = "cosine",
    ) -> List[Recommendation]:
        """Return the top-k roles for ``profile``.

        ``method="cosine"`` scores against the precomputed centroids;
        ``"knn"`` uses the fitted ``NearestNeighbors`` index and gives the
        same ordering.
        """
        user_vec = profile.as_vector()

        if method == "knn":
            n_query = min(len(self.centroids), max(top_k * 4, 10))
            dist, idx = self.ml_index.kneighbors(
                user_vec.reshape(1, -1), n_neighbors=n_query
            )
            scores = 1.0 - dist[0]
            order = list(idx[0])
        elif method == "cosine":
            scores_full = np.array(
                [_cosine(user_vec, self.centroids[i]) for i in range(len(self.centroids))]
            )
            order = list(np.argsort(-scores_full))
            scores = scores_full[order]
        else:
            raise ValueError(f"unknown method: {method!r}")

        results: List[Recommendation] = []
        for rank_pos, role_idx in enumerate(order):
            row = self._meta.iloc[role_idx]
            if sector_tag is not None and row["sector_tag"] != sector_tag:
                continue
            if seniority is not None and row["seniority"] not in set(seniority):
                continue
            centroid_vec = self.centroids[role_idx]
            centroid_dict = {o: float(v) for o, v in zip(ORIENTATIONS, centroid_vec)}
            dom = max(centroid_dict, key=centroid_dict.get)
            results.append(
                Recommendation(
                    role_id=str(row.name),
                    title_en=str(row["title_en"]),
                    title_ar=str(row["title_ar"]),
                    specialty=str(row["specialty"]),
                    seniority=str(row["seniority"]),
                    sector_tag=str(row["sector_tag"]),
                    vision_2030_anchor=str(row["vision_2030_anchor"]),
                    employer_examples=str(row["employer_examples"]),
                    nca_alignment=str(row["nca_alignment"]),
                    onet_soc=str(row["onet_soc"]),
                    score=float(scores[rank_pos]),
                    role_centroid=centroid_dict,
                    dominant_orientation=dom,
                    rationale=_rationale(profile, centroid_dict),
                )
            )
            if len(results) >= top_k:
                break
        return results


def _rationale(profile: OrientationProfile, centroid: Mapping[str, float]) -> str:
    """One-line explanation of why this role matched the profile."""
    user_top = profile.dominant()
    role_top = max(centroid, key=centroid.get)
    if user_top == role_top:
        return (
            f"Your dominant orientation is {user_top.replace('_', ' ')}, "
            f"which is also this role's strongest pull."
        )
    return (
        f"Your dominant orientation is {user_top.replace('_', ' ')}; "
        f"this role's strongest pull is {role_top.replace('_', ' ')} - "
        f"the match is on the next strongest fit."
    )


# Convenience: end-to-end
