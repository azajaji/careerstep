"""The one profile cohort every benchmark table is computed on.

Seven experiments previously built their own 120-profile cohort. Two of them
did so from a NumPy generator that had already been advanced by an earlier
step in the same script, so the categorical attributes of the profiles agreed
while the Gaussian CSMQ vectors did not. Tables computed in different scripts
were therefore not describing the same 120 students, which made the ranking
table and the calibration table incomparable.

This module builds the cohort once, from generators used for nothing else,
and every experiment imports it. The parameters are the ones the manuscript
reports; changing any of them changes every table, which is the intended
coupling.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd

from careerstep.career_positioning import RoleRecommender
from careerstep.career_positioning_benchmark import (
    SyntheticProfile, simulate_profiles,
)
from careerstep.seeding import load_seeds

N_PROFILES = 120
NOISE_SIGMA = 0.08
SKILL_COVERAGE = 0.35
DISTRACTORS = 3
NEIGHBOURS_ACCEPTABLE = 2

_CACHE: dict = {}


def build_cohort(recommender: RoleRecommender, roles_df: pd.DataFrame,
                 *, seed_offset: int = 0) -> List[SyntheticProfile]:
    """Return the frozen benchmark cohort.

    ``seed_offset`` is used only by the seed-sensitivity sweep; it is 0 for
    every reported table.
    """
    key = seed_offset
    if key in _CACHE:
        return _CACHE[key]

    seeds = load_seeds()
    # Dedicated generators. Nothing else may draw from these, which is the
    # whole point of routing every experiment through this function.
    py_rng = random.Random(seeds["python_random_seed"] + seed_offset)
    np_rng = np.random.default_rng(seeds["numpy_seed"] + seed_offset)

    profiles = simulate_profiles(
        recommender, roles_df,
        n_profiles=N_PROFILES,
        noise_sigma=NOISE_SIGMA,
        skill_coverage=SKILL_COVERAGE,
        distractor_skills_per_profile=DISTRACTORS,
        n_neighbours_acceptable=NEIGHBOURS_ACCEPTABLE,
        rng=py_rng, np_rng=np_rng,
    )
    _CACHE[key] = profiles
    return profiles


def cohort_fingerprint(profiles: List[SyntheticProfile]) -> str:
    """Short hash of the cohort, so every report can prove it used the same one."""
    import hashlib
    h = hashlib.sha256()
    for p in profiles:
        h.update(np.asarray(p.csmq_vector, dtype=float).tobytes())
        h.update("|".join(sorted(p.known_skills)).encode())
        h.update(str(p.level).encode())
        h.update("|".join(p.acceptable_role_ids).encode())
    return h.hexdigest()[:16]
