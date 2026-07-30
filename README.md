# CareerStep

Reference implementation and evaluation code for *"A Bilingual
Socio-Technical System for University Career Readiness with Values-Based Role
Positioning: A Design-Science Study"*.

This is the research implementation, not the client application. It involves
no human participants: every benchmark runs on the public O*NET Work Values
data, on synthetic profiles, or on corpora authored for the evaluation.

## Reproduce

```bash
pip install -r requirements.txt
python -m data.download --all     # build the corpora into data/cache/
python run_experiments.py          # write results/*.json
```

Seeds are fixed in `reproducibility/seeds.txt` and applied by
`careerstep.seeding.set_global_seeds()`. Each corpus-construction routine
carries its own fixed seed, so `data.download` is deterministic. The
committed `results/*.json` are the values reported in the paper.

## Layout

| Path | Contents |
|---|---|
| `careerstep/` | the modules: career positioning, CV parsing and scoring, CV/JD alignment, skill-gap matching, roadmap generation, interview generation, feedback aggregation |
| `eval/` | metrics and statistics |
| `data/` | corpus construction, plus the curated inputs (`csmq_items.json`, `onet_work_values.csv`, `saudi_cyber_roles.csv`) |
| `experiments/` | one script per reported measurement |
| `results/` | frozen outputs |

## Which script produces which table

Manuscript table numbers refer to the submitted version.

| Table | Script | Raw output |
|---|---|---|
| 5 — item-level questionnaire scoring | `experiments/exp7_questionnaire_scoring.py` | `results/exp7_questionnaire_scoring.json` |
| 6 — self-consistency and noise sweep | `experiments/exp8_career_positioning.py` | `results/exp8_career_positioning.json` |
| 7 — ranking benchmark | `experiments/exp8_career_positioning.py` | `results/exp8_career_positioning.json` |
| 8 — shortlist stability | `experiments/exp8d_shortlist_stability.py` | `results/exp8_shortlist_stability.json` |
| 9 — weight sensitivity (a) | `experiments/exp8b_projection_sensitivity.py` | `results/exp8_projection_sensitivity.json` |
| 9 — weight sensitivity (b) | `experiments/exp8c_suitability_weight_sweep.py` | `results/exp8_suitability_weight_sweep.json` |
| 10 — CV/JD alignment | `experiments/exp2_resume_job_alignment.py` | `results/exp2_resume_job_alignment.json` |
| 11 — roadmap generation | `experiments/exp4_roadmap_quality.py` | `results/exp4_roadmap_quality.json` |
| 12 — interview-question generation | `experiments/exp5_interview_generation.py` | `results/exp5_interview_generation.json` |
| 13 — feedback grounding | `experiments/exp6_feedback_evaluation.py` | `results/exp6_feedback_evaluation.json` |

Tables 1–4 and 14 are design tables with no computed values.

## Environment

Python 3.11 with the pinned versions in `requirements.txt`. Encoders:
`sentence-transformers/all-MiniLM-L6-v2` for embeddings and
`cross-encoder/ms-marco-MiniLM-L-6-v2` for the CV/JD reranker. No LLM or
network service is used by any experiment. Seeds are in
`reproducibility/seeds.txt`; every experiment calls
`set_global_seeds()` first, and `data.download` seeds each corpus routine
independently. Runtime for the full suite is a few minutes on CPU.

## Data

`onet_work_values.csv` is a curated cybersecurity subset of O*NET 28.0 Work
Values, which is public. `saudi_cyber_roles.csv` (25 roles) and
`csmq_items.json` (25 Likert items over Derr's five career orientations) were
authored for this study; the CSMQ items are original wording and have not been
psychometrically validated. Every other corpus built by `data.download` is
generated or hand-authored for evaluation and is not a substitute for real
labour-market data.

Set `KAGGLE_USERNAME`/`KAGGLE_KEY` to pull a real English resume corpus
instead of the synthetic fallback. The reported numbers use the fallback.

## Quick start

```python
from careerstep.career_positioning import project_work_values_to_csmq

profile = project_work_values_to_csmq({
    "achievement": 6.0, "independence": 5.0, "recognition": 4.0,
    "relationships": 3.0, "support": 4.0, "working_conditions": 5.0,
})
print(profile.scores)
```

## License

MIT.
