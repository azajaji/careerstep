# CareerStep

Reference implementation and evaluation code for *"A Socio-Technical Design
Pattern for Integrated University Career Readiness with Values-Based Role
Positioning"*.

This is the research implementation, not the client application. It involves
no human participants: every benchmark runs on public data (O*NET Work
Values, the MELO benchmark), on synthetic profiles, or on corpora authored
for the evaluation.

## Reproduce

```bash
pip install -r requirements.txt
python -m data.download --all     # build the corpora into data/cache/
python run_experiments.py          # write results/*.json
python -m tests.test_melo          # optional: checks on the MELO scoring
```

`exp12_melo_external` downloads the MELO benchmark on first run and caches it
under `data/cache/`; the other experiments need no network access.

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
| 6 — item-level inventory scoring | `experiments/exp7_questionnaire_scoring.py` | `results/exp7_questionnaire_scoring.json` |
| 7 — self-consistency and noise sweep | `experiments/exp8_career_positioning.py` | `results/exp8_career_positioning.json` |
| 8 — ranking benchmark | `experiments/exp8_career_positioning.py` | `results/exp8_career_positioning.json` |
| 9 — shortlist stability | `experiments/exp8d_shortlist_stability.py` | `results/exp8_shortlist_stability.json` |
| 10 — weight sensitivity (a) | `experiments/exp8b_projection_sensitivity.py` | `results/exp8_projection_sensitivity.json` |
| 10 — weight sensitivity (b) | `experiments/exp8c_suitability_weight_sweep.py` | `results/exp8_suitability_weight_sweep.json` |
| 11 — CV/JD alignment | `experiments/exp2_resume_job_alignment.py` | `results/exp2_resume_job_alignment.json` |
| 12 — roadmap generation | `experiments/exp4_roadmap_quality.py` | `results/exp4_roadmap_quality.json` |
| 13 — interview-question generation | `experiments/exp5_interview_generation.py` | `results/exp5_interview_generation.json` |
| 14 — external occupation linking (MELO) | `experiments/exp12_melo_external.py` | `results/exp12_melo_external.json` |

The feedback-grounding comparison is reported in the text of Section 6.3
rather than as a table; it comes from
`experiments/exp6_feedback_evaluation.py` →
`results/exp6_feedback_evaluation.json`.

Tables 1–5 and 15 are design or descriptive tables with no computed values.
`results/exp12_melo_external.json` keeps the per-query ranked occupations and
scores for all 633 queries, not only the aggregates in the table.

## Environment

Encoders: `sentence-transformers/all-MiniLM-L6-v2` for embeddings and
`cross-encoder/ms-marco-MiniLM-L-6-v2` for the reranker. No LLM or paid
service is used by any experiment; the only network access is the one-off
MELO download in `exp12`. Seeds are in `reproducibility/seeds.txt`; every
experiment calls `set_global_seeds()` first, and `data.download` seeds each
corpus routine independently. Set `PYTHONHASHSEED=0` before the interpreter
starts, since Python randomises string hashing per process.

The reported numbers were produced on this configuration:

| | |
|---|---|
| CPU | Intel Core Ultra 7 265F, 20 cores / 20 threads |
| RAM | 64 GB |
| OS | Windows 11 (build 26200) |
| Python | 3.13.9 |
| PyTorch | 2.11.0, running on CPU |
| sentence-transformers | 5.4.1 |
| NumPy / SciPy / scikit-learn | 2.3.5 / 1.16.3 / 1.7.2 |

`exp12` (MELO) takes 53 s wall-clock on that machine with a peak resident
set of 1.8 GB: 13 s to encode the 33,809-element corpus, 6 s for BM25
search, 2 s for bi-encoder search, and 30 s for cross-encoder reranking.
The full suite takes a few minutes. Run it with `CUDA_VISIBLE_DEVICES=""`
to reproduce the CPU timings; the metrics are identical on GPU, only the
timings differ.

## Data

Two categories, and the distinction matters for how the results should be
read.

**Generated or author-curated (Tables 6–13).** `saudi_cyber_roles.csv`
(25 roles) and `csmq_items.json` (25 Likert items over Derr's five career
orientations) were authored for this study; the CSMQ items are original
wording and have not been psychometrically validated. The CV,
job-description, learning-resource, interview-question, and feedback corpora
built by `data.download` are procedurally generated or hand-authored for
evaluation. None is a substitute for real labour-market data, and the study
team produced both the inputs and the reference labels.
`onet_work_values.csv` is a curated subset of the public O*NET 28.0 Work
Values dataset.

**External and externally annotated (Table 14).** MELO (Retyk et al., 2024,
arXiv:2410.08319, MIT licence) is fetched from
<https://github.com/Avature/melo-benchmark> rather than vendored here. It is
the one evaluation set this study neither built nor annotated. `exp12` uses
its `usa_q_en_c_en` configuration: 633 occupation-title queries against
33,809 ESCO surface forms, scored zero-shot. Each query has exactly one
correct ESCO occupation, so rankings are collapsed to one entry per
occupation before scoring; `tests/test_melo.py` asserts that property
against the downloaded annotations.

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
