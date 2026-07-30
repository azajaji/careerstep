"""Synthetic profiles and baseline rankers for the positioning benchmark."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from careerstep.career_positioning import (
    ORIENTATIONS,
    OrientationProfile,
    RoleRecommender,
)


# ---------------------------------------------------------------------------
# Specialty -> required-skill bank
# ---------------------------------------------------------------------------

# One-line vocabulary per specialty. The bank is small on purpose: the
# experiment compares rankers, not lexicons. The skill strings are
# normalised lowercase tokens so the lexical scorer can use substring /
# token overlap without preprocessing tricks.

SPECIALTY_SKILLS: Dict[str, List[str]] = {
    "defensive_soc": ["siem", "incident triage", "network logs", "linux", "splunk"],
    "dfir": ["digital forensics", "memory analysis", "volatility", "chain of custody", "yara"],
    "insider_threat": ["ueba", "data loss prevention", "behavioural analytics", "policy review"],
    "network_security": ["firewall", "ids", "ips", "vpn", "tcp/ip"],
    "mobile_security": ["android security", "ios security", "frida", "static analysis", "mobsf"],
    "ai_security": ["ml security", "adversarial examples", "model robustness", "ai red teaming"],
    "cyber_range": ["scenario design", "ctf", "purple teaming", "tabletop exercises"],
    "awareness_training": ["phishing simulation", "security awareness", "training delivery"],
    "ciso": ["security strategy", "governance", "risk management", "board reporting"],
    "security_architect": ["zero trust", "enterprise architecture", "threat modelling", "iam design"],
    "policy": ["regulation drafting", "policy writing", "standards mapping", "compliance"],
    "cyber_risk_quant": ["fair model", "monte carlo", "loss exceedance", "risk quantification"],
    "grc_compliance": ["iso 27001", "nca ecc", "audit documentation", "control mapping"],
    "reverse_engineering": ["ghidra", "ida pro", "malware analysis", "x86 assembly"],
    "cti": ["threat intel", "mitre att&ck", "stix taxii", "indicator pivoting"],
    "cryptography": ["aes", "rsa", "key management", "pkcs", "hsm"],
    "iam_pam": ["identity management", "okta", "active directory", "privileged access"],
    "critical_infra": ["ot security", "scada", "iec 62443", "asset inventory"],
    "ot_ics": ["scada", "plc", "modbus", "industrial control"],
    "cloud_sec": ["aws security", "azure security", "kubernetes", "cspm"],
    "appsec": ["sast", "dast", "owasp top 10", "secure code review"],
    "offensive_redteam": ["red team operations", "c2 frameworks", "evasion", "active directory abuse"],
    "offensive_pentest": ["metasploit", "burp suite", "exploitation", "nmap"],
    "audit_assurance": ["audit planning", "evidence collection", "iso 27001", "control testing"],
}


SENIORITY_TO_LEVEL: Dict[str, str] = {
    "entry": "entry",
    "mid": "mid",
    "senior": "senior",
    "executive": "executive",
}


# Feasibility lookup for a student at a given training stage applying
# for a role of a given seniority. Numbers are intentionally simple and
# documented in the manuscript so the weighting is auditable.
FEASIBILITY: Dict[Tuple[str, str], float] = {
    ("student", "entry"):     1.00,
    ("student", "mid"):       0.70,
    ("student", "senior"):    0.25,
    ("student", "executive"): 0.05,
    ("graduate", "entry"):    0.95,
    ("graduate", "mid"):      0.85,
    ("graduate", "senior"):   0.45,
    ("graduate", "executive"): 0.15,
}


# Weights for the full Khutwa suitability score.
W_VALUE = 0.60
W_SKILL = 0.25
W_FEAS  = 0.15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticProfile:
    profile_id: int
    latent_role_id: str
    acceptable_role_ids: Tuple[str, ...]
    csmq_vector: np.ndarray          # shape (5,), in [0, 1]
    known_skills: Tuple[str, ...]
    level: str                       # "student" or "graduate"


@dataclass
class RankerResult:
    ranked_role_ids: List[str]


# ---------------------------------------------------------------------------
# Required-skills derivation
# ---------------------------------------------------------------------------


def required_skills_for_role(specialty: str) -> List[str]:
    """Return the required-skill list for a role's specialty.

    Specialties without a curated bank fall back to a generic
    cybersecurity vocabulary; this keeps the experiment robust if the
    catalogue grows.
    """
    return list(SPECIALTY_SKILLS.get(
        specialty,
        ["cybersecurity fundamentals", "linux", "scripting", "incident response"],
    ))


def role_to_required_skills(roles_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map each role_id to its derived required-skills list."""
    return {
        str(row["role_id"]): required_skills_for_role(str(row["specialty"]))
        for _, row in roles_df.iterrows()
    }


# ---------------------------------------------------------------------------
# Profile simulation
# ---------------------------------------------------------------------------


def _nearest_neighbours(
    centroids: np.ndarray,
    role_ids: Sequence[str],
    target_idx: int,
    k: int,
) -> List[str]:
    target = centroids[target_idx]
    # Cosine distance to all roles.
    norms = np.linalg.norm(centroids, axis=1)
    target_norm = np.linalg.norm(target)
    if target_norm == 0:
        return []
    cos_sim = (centroids @ target) / (norms * target_norm + 1e-12)
    order = np.argsort(-cos_sim)
    out: List[str] = []
    for idx in order:
        if int(idx) == target_idx:
            continue
        out.append(str(role_ids[int(idx)]))
        if len(out) == k:
            break
    return out


def simulate_profiles(
    recommender: RoleRecommender,
    roles_df: pd.DataFrame,
    *,
    n_profiles: int = 120,
    noise_sigma: float = 0.08,
    skill_coverage: float = 0.55,
    distractor_skills_per_profile: int = 2,
    n_neighbours_acceptable: int = 2,
    student_share: float = 0.7,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> List[SyntheticProfile]:
    """Generate synthetic student profiles for the ranking benchmark.

    For each profile:
      * Pick a latent role uniformly from the catalogue.
      * Set the CSMQ vector to the role's orientation centroid plus
        Gaussian noise (sigma = ``noise_sigma``), clipped to [0, 1].
      * Sample a partial skill set: keep ``skill_coverage`` of the
        role's required skills + ``distractor_skills_per_profile``
        random skills drawn from other roles.
      * Acceptable target roles = the latent role plus its
        ``n_neighbours_acceptable`` nearest neighbours in the
        orientation-centroid space.
      * Student level = "student" with probability ``student_share``
        else "graduate".
    """
    role_ids = list(recommender.centroid_df.index)
    centroids = recommender.centroids
    required = role_to_required_skills(roles_df)
    universe: List[str] = sorted({s for skills in required.values() for s in skills})

    profiles: List[SyntheticProfile] = []
    for pid in range(n_profiles):
        latent_idx = rng.randrange(len(role_ids))
        latent_id = role_ids[latent_idx]
        latent_vec = centroids[latent_idx]
        noisy = latent_vec + np_rng.normal(0.0, noise_sigma, size=latent_vec.shape)
        noisy = np.clip(noisy, 0.0, 1.0)

        # Sampled known-skills set.
        role_skills = required[latent_id]
        n_keep = max(1, int(round(skill_coverage * len(role_skills))))
        kept = rng.sample(role_skills, k=min(n_keep, len(role_skills)))
        # Add distractors from the cross-role universe (excluding role's skills).
        distractor_pool = [s for s in universe if s not in role_skills]
        if distractor_pool:
            n_distract = min(distractor_skills_per_profile, len(distractor_pool))
            kept += rng.sample(distractor_pool, k=n_distract)

        neighbours = _nearest_neighbours(
            centroids, role_ids, latent_idx, k=n_neighbours_acceptable
        )
        acceptable = tuple([latent_id] + neighbours)

        level = "student" if rng.random() < student_share else "graduate"

        profiles.append(SyntheticProfile(
            profile_id=pid,
            latent_role_id=latent_id,
            acceptable_role_ids=acceptable,
            csmq_vector=noisy,
            known_skills=tuple(kept),
            level=level,
        ))
    return profiles


# ---------------------------------------------------------------------------
# Skill-readiness scorers
# ---------------------------------------------------------------------------


def lexical_skill_readiness(
    user_skills: Sequence[str], required: Sequence[str]
) -> float:
    if not required:
        return 0.0
    user_set = {s.lower() for s in user_skills}
    req_set = {s.lower() for s in required}
    return len(user_set & req_set) / len(req_set)


def semantic_skill_readiness(
    user_skills: Sequence[str],
    required: Sequence[str],
    *,
    encode: Callable[[Sequence[str]], np.ndarray],
    threshold: float = 0.55,
) -> float:
    """Embedding-based skill readiness: a required skill is "covered" if
    any user skill embeds within cosine distance of ``threshold``.
    """
    if not required:
        return 0.0
    if not user_skills:
        return 0.0
    req_emb = encode(list(required))
    usr_emb = encode(list(user_skills))
    sims = req_emb @ usr_emb.T  # both rows already L2-normalised
    covered = (sims.max(axis=1) >= threshold).sum()
    return float(covered) / len(required)


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------


def _role_centroid_vec(recommender: RoleRecommender, role_id: str) -> np.ndarray:
    return recommender.centroids[recommender.centroid_df.index.get_loc(role_id)]


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def rank_random(profile: SyntheticProfile, role_ids: Sequence[str],
                rng: random.Random) -> List[str]:
    order = list(role_ids)
    rng.shuffle(order)
    return order


def rank_level_only(profile: SyntheticProfile, recommender: RoleRecommender,
                    roles_df: pd.DataFrame) -> List[str]:
    role_ids = list(recommender.centroid_df.index)
    metas = roles_df.set_index("role_id")
    scores = []
    for rid in role_ids:
        seniority = str(metas.loc[rid, "seniority"]).lower()
        feas = FEASIBILITY.get((profile.level, seniority), 0.5)
        scores.append(feas)
    order = np.argsort(-np.asarray(scores))
    return [role_ids[int(i)] for i in order]


def rank_skills_only(profile: SyntheticProfile, recommender: RoleRecommender,
                     required: Dict[str, List[str]],
                     skill_scorer: Callable[[Sequence[str], Sequence[str]], float],
                     ) -> List[str]:
    role_ids = list(recommender.centroid_df.index)
    scores = [skill_scorer(profile.known_skills, required[rid]) for rid in role_ids]
    order = np.argsort(-np.asarray(scores))
    return [role_ids[int(i)] for i in order]


def rank_csmq_only(profile: SyntheticProfile, recommender: RoleRecommender) -> List[str]:
    role_ids = list(recommender.centroid_df.index)
    scores = [_cosine(profile.csmq_vector, recommender.centroids[i])
              for i in range(len(role_ids))]
    order = np.argsort(-np.asarray(scores))
    return [role_ids[int(i)] for i in order]


def rank_khutwa(profile: SyntheticProfile, recommender: RoleRecommender,
                roles_df: pd.DataFrame, required: Dict[str, List[str]],
                skill_scorer: Callable[[Sequence[str], Sequence[str]], float],
                ) -> List[str]:
    role_ids = list(recommender.centroid_df.index)
    metas = roles_df.set_index("role_id")
    scores = []
    for i, rid in enumerate(role_ids):
        value_fit = _cosine(profile.csmq_vector, recommender.centroids[i])
        skill = skill_scorer(profile.known_skills, required[rid])
        seniority = str(metas.loc[rid, "seniority"]).lower()
        feas = FEASIBILITY.get((profile.level, seniority), 0.5)
        scores.append(W_VALUE * value_fit + W_SKILL * skill + W_FEAS * feas)
    order = np.argsort(-np.asarray(scores))
    return [role_ids[int(i)] for i in order]


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def recall_at_k(ranked: Sequence[str], acceptable: Sequence[str], k: int) -> float:
    top = set(ranked[:k])
    acc = set(acceptable)
    if not acc:
        return 0.0
    return float(len(top & acc) > 0)


def mrr(ranked: Sequence[str], acceptable: Sequence[str]) -> float:
    acc = set(acceptable)
    for i, rid in enumerate(ranked, start=1):
        if rid in acc:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], acceptable: Sequence[str], k: int) -> float:
    """Binary relevance nDCG@k (relevance = 1 if in acceptable set)."""
    acc = set(acceptable)
    dcg = 0.0
    for i, rid in enumerate(ranked[:k], start=1):
        rel = 1.0 if rid in acc else 0.0
        if rel:
            dcg += rel / math.log2(i + 1)
    # Ideal DCG: best case puts up to min(k, |acceptable|) hits at the top.
    n_hits = min(k, len(acc))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0
