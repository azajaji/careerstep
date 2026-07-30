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
