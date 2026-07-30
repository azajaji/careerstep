"""Roadmap generation against random, popularity, and lexical baselines."""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List

import numpy as np

from data.loaders import load_learning_resources, load_occupations
from eval.metrics_generation import (
    coverage as coverage_metric,
    diversity,
    list_overlap,
    semantic_similarity,
)
from eval.stats import summarize, wilcoxon_signed_rank
from experiments._io import print_header, save_report
from careerstep.backends import EmbeddingBackend
from careerstep.roadmap import LearningItem, RoadmapGenerator
from careerstep.roadmap_baselines import (
    LexicalRoadmap,
    PopularityRoadmap,
    RandomRoadmap,
)
from careerstep.seeding import set_global_seeds


def _bank_from_df(df) -> List[LearningItem]:
    items = []
    for _, row in df.iterrows():
        items.append(LearningItem(
            skill=row["skill"],
            resource=row["resource"],
            provider=row.get("provider", ""),
            estimated_hours=int(row.get("estimated_hours", 0)),
            level=row.get("level", "intermediate"),
            certification=bool(row.get("certification", False)),
        ))
    return items


def _popularity_quartile(bank: List[LearningItem]) -> set:
    counter = Counter(item.skill.lower() for item in bank)
    if not counter:
        return set()
    sorted_skills = sorted(counter.values())
    cutoff = sorted_skills[max(0, int(len(sorted_skills) * 0.75))]
    return {s for s, c in counter.items() if c >= cutoff}


def _novelty(items: List[LearningItem], popular: set) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if it.skill.lower() not in popular) / len(items)


def _evaluate_one(
    items: List[LearningItem],
    missing: List[str],
    reference: List[LearningItem],
    popular: set,
    backend: EmbeddingBackend,
) -> Dict[str, float]:
    pred_texts = [it.as_text() for it in items]
    pred_skills = [it.skill for it in items]
    ref_texts = [it.as_text() for it in reference]
    ref_skills = [it.skill for it in reference]
    return {
        "coverage": coverage_metric(pred_texts, missing),
        "overlap": list_overlap(pred_skills, ref_skills),
        "semantic_similarity": (
            semantic_similarity(pred_texts, ref_texts, backend=backend)
            if pred_texts and ref_texts else 0.0
        ),
        "diversity": diversity(pred_texts, backend=backend) if pred_texts else 0.0,
        "novelty": _novelty(items, popular),
    }


def run() -> dict:
    seeds = set_global_seeds()
    rng = random.Random(seeds["python_random_seed"])
    resources = load_learning_resources()
    occ = load_occupations()

    bank = _bank_from_df(resources)
    backend = EmbeddingBackend()
    popular = _popularity_quartile(bank)

    baseline_systems = {
        "random": RandomRoadmap(),
        "popularity": PopularityRoadmap(),
        "lexical": LexicalRoadmap(),
    }
    khutwa_gen = RoadmapGenerator(bank, backend=backend)
    all_names = list(baseline_systems) + ["khutwa"]

    per_system: Dict[str, List[Dict[str, float]]] = {s: [] for s in all_names}
    per_role_log: List[Dict] = []

    for _, row in occ.iterrows():
        role = row["role"]
        role_skills = list(row["skills"])
        if len(role_skills) < 2:
            continue
        rng.shuffle(role_skills)
        cut = max(1, len(role_skills) // 2)
        missing = role_skills[:cut]

        reference = [it for it in bank if it.skill.lower() in {m.lower() for m in missing}]
        log_entry: Dict = {"role": role, "missing_skills": missing}

        # Baselines.
        for name, baseline in baseline_systems.items():
            items = baseline.generate(
                role=role, missing_skills=missing, resource_bank=bank, k=8
            )
            metrics = _evaluate_one(items, missing, reference, popular, backend)
            per_system[name].append(metrics)
            log_entry[name] = metrics

        # Khutwa.
        rm = khutwa_gen.generate(role=role, missing_skills=missing, max_items=8)
        metrics = _evaluate_one(rm.items, missing, reference, popular, backend)
        per_system["khutwa"].append(metrics)
        log_entry["khutwa"] = metrics

        per_role_log.append(log_entry)

    # Aggregate.
    aggregate: Dict[str, Dict[str, Dict]] = {}
    for name, rows in per_system.items():
        if not rows:
            continue
        aggregate[name] = {
            metric: summarize([r[metric] for r in rows])
            for metric in ("coverage", "overlap", "semantic_similarity", "diversity", "novelty")
        }

    # Significance vs Khutwa on coverage.
    khutwa_cov = [r["coverage"] for r in per_system["khutwa"]]
    sig = {}
    for name in ("random", "popularity", "lexical"):
        base_cov = [r["coverage"] for r in per_system[name]]
        sig[f"khutwa_vs_{name}_coverage"] = wilcoxon_signed_rank(khutwa_cov, base_cov).to_dict()

    payload = {
        "systems": aggregate,
        "tests": sig,
        "per_role_log": per_role_log,
    }
    print("\n  -- per-system aggregate --")
    for name, m in aggregate.items():
        print(
            f"    {name:11s}: coverage={m['coverage']['mean']:.3f}  "
            f"semantic={m['semantic_similarity']['mean']:.3f}  "
            f"diversity={m['diversity']['mean']:.3f}  "
            f"novelty={m['novelty']['mean']:.3f}"
        )
    return payload


if __name__ == "__main__":
    print_header("Experiment 4 - Roadmap quality (with baselines)")
    payload = run()
    path = save_report("exp4_roadmap_quality", payload)
    print(f"\nSaved {path}")
