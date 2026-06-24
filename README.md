# CareerStep — career-positioning artefact and evaluation data

Companion repository for the paper *"A Bilingual Socio-Technical System for
University Career Readiness with Values-Based Role Positioning: A
Design-Science Study"* (under review at *Systems*, MDPI).

## Contents

- **`careerstep/career_positioning.py`** — reference implementation of the
  Career Positioning Module: 25-item CSMQ scoring, the fixed 5×6 CSMQ→O*NET
  projection matrix *W*, and the role recommender (cosine over role-orientation
  centroids, with separable value-fit / skill-readiness / feasibility terms).
- **`data/`** — `saudi_cyber_roles.csv` (the 25-role catalogue),
  `onet_work_values.csv` (O*NET Work-Value anchors), and `csmq_items.json` (the
  25-item questionnaire).
- **`results/`** — frozen benchmark outputs that feed the paper's tables:
  `career_positioning_benchmark.json` (self-consistency and ranking),
  `projection_weight_sensitivity.json`, and `suitability_weight_sweep.json`.
- **`survey/`** — the bilingual (Arabic/English) usability instrument.

## Quick start

```python
from careerstep.career_positioning import project_work_values_to_csmq

# Project an O*NET Work-Value vector (0–7 Extent scale) into the CSMQ space:
profile = project_work_values_to_csmq({
    "achievement": 6.0, "independence": 5.0, "recognition": 4.0,
    "relationships": 3.0, "support": 4.0, "working_conditions": 5.0,
})
print(profile.scores)
```

See `careerstep/career_positioning.py` for the full recommender API
(`RoleRecommender`, `score_csmq`).

## Notes

- O*NET is a public dataset; the files here are the curated cybersecurity-domain
  subset used by the prototype.
- The user study was anonymous; only the survey instrument is included here.
  Aggregate, de-identified per-item response counts are available from the
  corresponding author.

## License

MIT (see `LICENSE`).
