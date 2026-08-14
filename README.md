# CareerStep

Reference implementation and evaluation code for *"A Scale-Contract Audit for
Composite Scoring Systems: Detecting Weight-Influence Misalignment and
Evaluating Layer-Specific Repairs"*.

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

Manuscript numbers refer to the IEEE Access submission. Tables 1–4 describe the
artifact and the evaluation inputs. The primary evidence (RQ1–RQ5) is
Tables 5–12, and the external transfer case is Table 13. There is no
supplementary document; every reported number comes from the table below.

The O*NET Extent scale (`EX`) runs 1–7, so ratings are min-max mapped to
`[0,1]` as `(x-1)/6` in `careerstep/career_positioning.py`,
`experiments/exp15_external_structure.py`, and
`experiments/exp17_fitted_projection.py`.

This repository exists to let you re-derive the reported numbers, not to
rebuild the article. The manuscript's figures are not redistributed here; the
values they plot are in `results/`, namely the criterion ranges in
`exp8e_component_scale.json` (Figure 1) and the weight-versus-influence
comparison in `exp8f_criterion_influence.json` (Figure 2).

Run everything with `python run_experiments.py`.

### Main article

| Manuscript | Script | Raw output |
|---|---|---|
| Table 5 — criterion influence (RQ1) | `experiments/exp8f_criterion_influence.py` | `results/exp8f_criterion_influence.json` |
| Table 6 — the same measures over 20 seeded cohorts (RQ1) | `experiments/exp19_seed_sensitivity.py` | `results/exp19_seed_sensitivity.json` |
| Table 7 — operating region of the inversion (RQ1) | `experiments/exp20_operating_region.py` | `results/exp20_operating_region.json` |
| Table 8 — criterion scale and centroid geometry (RQ2) | `experiments/exp8e_component_scale.py` | `results/exp8e_component_scale.json` |
| Table 9 — projection vs. baselines and nulls (RQ3) | `experiments/exp15_external_structure.py` | `results/exp15_external_structure.json` |
| Table 10 — influence after scale calibration (RQ4) | `experiments/exp16_scale_calibration.py` | `results/exp16_scale_calibration.json` |
| Table 11 — what calibration changes and costs (RQ4) | `experiments/exp16_scale_calibration.py` | `results/exp16_scale_calibration.json` |
| Table 12 — fitted, refitted and unsupervised projections, held out (RQ5) | `experiments/exp17_fitted_projection.py` | `results/exp17_fitted_projection.json` |
| Table 13 — the audit on the Human Development Index | `experiments/exp18_external_index_audit.py` | `results/exp18_external_index_audit.json` |

Tables 1–4 (evaluation coverage, projection matrix, feasibility lookup,
evaluation data) are descriptive and are not produced by a script.

The remaining scripts under `experiments/` measure the workflow around the
audited scorer. They are kept because the code they exercise ships with the
release, and no number they produce appears in the article.


## Environment

Encoders: `sentence-transformers/all-MiniLM-L6-v2` for embeddings and
`cross-encoder/ms-marco-MiniLM-L-6-v2` for the reranker. The only network
access is the one-off MELO download in `exp12`.

No experiment calls a language model. This matters because
`careerstep.backends` selects a language-model path whenever
`OPENAI_API_KEY` is set, and on that path the interview generator returns
different questions and no longer satisfies its coverage constraint. The
reported numbers are the offline path, so `set_global_seeds()` — which every
experiment calls first — removes `OPENAI_API_KEY` from the process and says
so on stdout. A key left in your environment therefore cannot change the
results.

Seeds are in `reproducibility/seeds.txt`; `data.download` seeds each corpus
routine independently. Set `PYTHONHASHSEED=0` before the interpreter starts,
since Python randomises string hashing per process.

Every reported number, Tables 6–24, was produced on one
configuration; there is no separate environment for the MELO benchmark:

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

**Generated or author-curated (Tables 6–16, 19–23).** `saudi_cyber_roles.csv`
(25 roles) and `csmq_items.json` (25 Likert items over Derr's five career
orientations) were authored for this study; the CSMQ items are original
wording and have not been psychometrically validated. The CV,
job-description, learning-resource, interview-question, and feedback corpora
built by `data.download` are procedurally generated or hand-authored for
evaluation. None is a substitute for real labour-market data, and the study
team produced both the inputs and the reference labels.
`onet_work_values.csv` is a curated subset of the public O*NET 28.0 Work
Values dataset.

**External and externally annotated (Tables 17 and 24).** MELO (Retyk et al., 2024,
arXiv:2410.08319, MIT licence) is fetched from
<https://github.com/Avature/melo-benchmark> rather than vendored here. It is
one of two inputs this study neither built nor annotated. `exp18` uses the
other: the UNDP Human Development Report 2023-24 composite-indices time
series, fetched from <https://hdr.undp.org>, from which the three HDI
dimension indices are recomputed using UNDP's published goalposts and checked
against the published HDI before the audit runs. `exp12` uses
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
