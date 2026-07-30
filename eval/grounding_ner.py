"""Deterministic entity spotter for feedback grounding.

A bullet is grounded when every entity it names also appears in the
source CV. Entities come from a fixed vocabulary over three classes
(TECH, ORG_PROD, PROPER); no model weights are involved."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set


# Vocabulary -- TECH entities, lower case. Only domain terms appear, so
# generic English words are never flagged regardless of capitalisation.

_TECH_TERMS: Set[str] = {
    # languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
    "ruby", "php", "scala", "kotlin", "swift", "dart", "r",
    # web / frontend
    "react", "vue", "vue.js", "angular", "next.js", "nuxt", "svelte",
    "tailwind", "html", "css", "sass", "redux", "rxjs", "ngrx",
    # backend
    "node", "node.js", "django", "flask", "fastapi", "spring", "spring boot",
    "express", "rails", "gin",
    # data / ml
    "pandas", "numpy", "scikit-learn", "scikit", "sklearn", "pytorch",
    "tensorflow", "keras", "transformers", "xgboost", "lightgbm",
    "spark", "apache spark", "airflow", "apache airflow", "mlflow", "huggingface",
    # databases
    "postgres", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "snowflake", "bigquery", "cosmos db",
    # devops / cloud
    "docker", "kubernetes", "k8s", "terraform", "ansible", "helm", "jenkins",
    "github actions", "gitlab ci", "aws", "azure", "gcp", "ec2", "s3",
    "lambda", "rds", "guardduty", "cloudtrail",
    # bi / viz
    "tableau", "power bi", "looker", "metabase", "plotly", "matplotlib",
    "seaborn", "figma", "sketch",
    # query / data eng
    "sql", "graphql", "kafka", "rabbitmq", "grpc", "rest", "rest api",
    "websocket", "webrtc", "etl",
    # security
    "siem", "splunk", "wireshark", "nmap", "burp suite", "metasploit",
    "owasp", "kali", "mitre att&ck",
    # certifications
    "aws certified solutions architect", "saa-c03", "saa-c02", "clf-c02",
    "az-900", "az-104", "az-204",
    "google cloud associate cloud engineer", "ace",
    "certified kubernetes administrator", "cka",
    "certified scrummaster", "csm",
    "pmp", "cissp", "ceh", "security+", "sy0-701",
    "tensorflow developer certificate",
    "google data analytics professional certificate",
    "ibm data science professional certificate",
    "google ux design professional certificate",
    "meta front-end developer certificate",
    "toefl ibt", "ielts",
    # protocols / standards
    "tcp/ip", "tcp", "ip", "oauth", "oauth 2.0", "jwt", "owasp top 10",
    "iso 27001",
    # methodology
    "agile", "scrum", "kanban", "tdd", "ci/cd",
}

# Common acronyms that should be flagged as TECH/PROPER even though they look
# like ordinary capitalised tokens. Listed lower-case for consistency.
_TECH_ACRONYMS: Set[str] = {
    "api", "sdk", "cli", "gui", "saas", "iaas", "paas", "etl", "elt",
    "kpi", "okr", "roi", "qa", "ux", "ui", "ner", "llm", "nlp", "cv",
    "ats", "irb", "sus", "umux", "ksa", "ksu", "mvp",
}

# Generic English / structural words: never count these as entities even
# when capitalised at the start of a sentence.
_STRUCTURAL: Set[str] = {
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "we", "they", "he", "she", "it",
    "and", "or", "but", "so", "if", "when", "while", "because",
    "add", "include", "use", "avoid", "move", "remove", "rewrite", "expand",
    "reduce", "highlight", "clarify", "emphasize", "reorder", "practice",
    "study", "learn", "complete", "enrol", "prepare", "tell", "walk",
    "describe", "explain", "how", "what", "when", "why",
    "your", "my", "our", "their",
    "is", "are", "was", "were", "be", "been", "being",
    "section", "summary", "experience", "education", "skills", "projects",
    "certifications", "publications", "awards", "objective",
    "star", "pdf", "docx", "url", "ats", "cv", "khutwa", "ai",
    "english", "arabic",
}

# Pattern for proper-noun-like tokens (English-capitalised).
_PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z+#./0-9-]{1,})\b")

# Multi-word TECH terms sorted longest-first so "spring boot" is matched
# before "spring" alone.
_TECH_SORTED: List[str] = sorted(_TECH_TERMS, key=len, reverse=True)


@dataclass
class GroundingReport:
    """Per-bullet grounding outcome with entity-type breakdown."""
    bullets: int = 0
    grounded: int = 0
    ungrounded_by_type: Dict[str, int] = field(default_factory=dict)
    ungrounded_entities: List[str] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        if not self.bullets:
            return 1.0
        return self.grounded / self.bullets

    @property
    def hallucination_rate(self) -> float:
        return 1.0 - self.grounding_rate


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _detect_entities(text: str) -> Dict[str, List[str]]:
    """Return entities grouped by type.

    The detection is conservative: a token is only added once, the first
    type that matches wins (TECH > ORG_PROD > PROPER), and structural /
    pronoun-like words are excluded.
    """
    low = _normalise(text)
    seen: Set[str] = set()
    found: Dict[str, List[str]] = {"TECH": [], "ORG_PROD": [], "PROPER": []}

    # TECH (multi-word, longest-first).
    for term in _TECH_SORTED:
        if term in low and term not in seen:
            found["TECH"].append(term)
            seen.add(term)

    # TECH acronyms (whole-word match).
    for ac in _TECH_ACRONYMS:
        if re.search(rf"\b{re.escape(ac)}\b", low) and ac not in seen:
            found["TECH"].append(ac)
            seen.add(ac)

    # PROPER nouns (English-capitalised, fall-through).
    for m in _PROPER_RE.findall(text):
        m_low = m.lower()
        if m_low in seen or m_low in _STRUCTURAL or m_low in _TECH_TERMS:
            continue
        # Treat all-uppercase 2-4 letter tokens as ORG_PROD (e.g., IBM, NASA).
        if m.isupper() and 2 <= len(m) <= 4:
            found["ORG_PROD"].append(m_low)
        else:
            found["PROPER"].append(m_low)
        seen.add(m_low)

    return found


def grounding_evaluation(
    predicted_bullets: Sequence[str], source_text: str
) -> GroundingReport:
    """Compute the grounding rate of ``predicted_bullets`` against ``source_text``."""
    src_low = _normalise(source_text)
    report = GroundingReport(bullets=len(predicted_bullets))
    for bullet in predicted_bullets:
        entities = _detect_entities(bullet)
        bullet_grounded = True
        for entity_type, items in entities.items():
            for item in items:
                if item not in src_low:
                    bullet_grounded = False
                    report.ungrounded_by_type[entity_type] = (
                        report.ungrounded_by_type.get(entity_type, 0) + 1
                    )
                    report.ungrounded_entities.append(item)
        if bullet_grounded:
            report.grounded += 1
    return report


def grounding_summary(
    cases: Sequence[Dict],
) -> Dict[str, float]:
    """Aggregate grounding across a list of {predicted: [...], source: str} cases."""
    if not cases:
        return {"grounding_rate": 1.0, "hallucination_rate": 0.0, "n_cases": 0}

    per_case_rate: List[float] = []
    total_bullets = 0
    grounded_bullets = 0
    type_counts: Dict[str, int] = {}

    for case in cases:
        report = grounding_evaluation(case["predicted"], case["source"])
        per_case_rate.append(report.grounding_rate)
        total_bullets += report.bullets
        grounded_bullets += report.grounded
        for k, v in report.ungrounded_by_type.items():
            type_counts[k] = type_counts.get(k, 0) + v

    return {
        "grounding_rate": grounded_bullets / max(total_bullets, 1),
        "hallucination_rate": 1.0 - grounded_bullets / max(total_bullets, 1),
        "n_cases": len(cases),
        "n_bullets": total_bullets,
        "per_case_rate_mean": sum(per_case_rate) / len(per_case_rate),
        "ungrounded_by_type": type_counts,
    }


# Context-aware grounding. Each bullet is classified, then checked under a
# class-specific rule:
#
#   CLAIM           asserts something about the candidate -> entities must be in CV
#   RECOMMENDATION  prescribes an action ("Learn X")      -> entities need not be in CV
#   REFERENCE       states a role fact ("role requires Z") -> entities must be in JD text
#
# Bullets default to CLAIM when no cue matches.

_RECOMMEND_CUES = (
    r"\b(learn|study|practice|practi[cs]e|complete|enrol|take a course|"
    r"add|include|incorporate|consider|build|develop|earn|obtain|"
    r"pursue|familiari[sz]e|acquire)\b"
)
_REFERENCE_CUES = (
    r"\b(the role|the job|the position|this (role|job|position)|"
    r"job description|jd|required skills?|target role|role requires?|"
    r"is required for|expects?|asks? for)\b"
)
_RECOMMEND_RE = re.compile(_RECOMMEND_CUES, re.IGNORECASE)
_REFERENCE_RE = re.compile(_REFERENCE_CUES, re.IGNORECASE)


def classify_bullet_context(bullet: str) -> str:
    """Return one of ``CLAIM``, ``RECOMMENDATION``, ``REFERENCE``."""
    if _REFERENCE_RE.search(bullet):
        return "REFERENCE"
    if _RECOMMEND_RE.search(bullet):
        return "RECOMMENDATION"
    return "CLAIM"


@dataclass
class ContextAwareGroundingReport:
    bullets: int = 0
    grounded: int = 0
    per_context: Dict[str, Dict[str, int]] = field(default_factory=dict)
    ungrounded_entities: List[Dict[str, str]] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        if not self.bullets:
            return 1.0
        return self.grounded / self.bullets


def context_aware_grounding_evaluation(
    predicted_bullets: Sequence[str],
    cv_text: str,
    reference_text: str = "",
) -> ContextAwareGroundingReport:
    """Context-aware grounding rule.

    CLAIM entities must be in ``cv_text``; RECOMMENDATION bullets are
    always grounded (their entities are by design absent from the CV);
    REFERENCE entities must be in ``reference_text`` (the JD or
    required-skills text).
    """
    cv_low = _normalise(cv_text)
    ref_low = _normalise(reference_text)
    report = ContextAwareGroundingReport(bullets=len(predicted_bullets))

    for bullet in predicted_bullets:
        ctx = classify_bullet_context(bullet)
        ctx_block = report.per_context.setdefault(
            ctx, {"bullets": 0, "grounded": 0, "ungrounded_entities": 0}
        )
        ctx_block["bullets"] += 1

        if ctx == "RECOMMENDATION":
            ctx_block["grounded"] += 1
            report.grounded += 1
            continue

        target_low = cv_low if ctx == "CLAIM" else ref_low
        entities = _detect_entities(bullet)
        bullet_grounded = True
        for entity_type, items in entities.items():
            for item in items:
                if item not in target_low:
                    bullet_grounded = False
                    ctx_block["ungrounded_entities"] += 1
                    report.ungrounded_entities.append({
                        "context": ctx, "type": entity_type, "entity": item,
                    })
        if bullet_grounded:
            ctx_block["grounded"] += 1
            report.grounded += 1

    return report


def context_aware_grounding_summary(cases: Sequence[Dict]) -> Dict[str, object]:
    """Aggregate context-aware grounding across cases.

    Each case is ``{predicted: [bullets], source: cv_text, reference: jd_text}``.
    The ``reference`` field defaults to empty.
    """
    if not cases:
        return {"grounding_rate": 1.0, "n_cases": 0}

    total_bullets = 0
    grounded_bullets = 0
    per_context: Dict[str, Dict[str, int]] = {}
    ungrounded_by_type_and_context: Dict[str, Dict[str, int]] = {}

    for case in cases:
        report = context_aware_grounding_evaluation(
            case["predicted"], case["source"], case.get("reference", "")
        )
        total_bullets += report.bullets
        grounded_bullets += report.grounded
        for ctx, block in report.per_context.items():
            agg = per_context.setdefault(
                ctx, {"bullets": 0, "grounded": 0, "ungrounded_entities": 0}
            )
            agg["bullets"] += block["bullets"]
            agg["grounded"] += block["grounded"]
            agg["ungrounded_entities"] += block["ungrounded_entities"]
        for entry in report.ungrounded_entities:
            ctx_block = ungrounded_by_type_and_context.setdefault(entry["context"], {})
            ctx_block[entry["type"]] = ctx_block.get(entry["type"], 0) + 1

    rate = grounded_bullets / max(total_bullets, 1)
    return {
        "grounding_rate": rate,
        "hallucination_rate": 1.0 - rate,
        "n_cases": len(cases),
        "n_bullets": total_bullets,
        "per_context": per_context,
        "ungrounded_by_type_and_context": ungrounded_by_type_and_context,
    }
