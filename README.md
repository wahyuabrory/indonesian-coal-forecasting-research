# Indonesian coal forecasting research

Short-horizon return forecasting for Indonesian coal equities: do
literature-supported external market variables improve next-day return
forecasts over stock history alone, and does more model complexity pay?

The short answer is no. Across ADRO, PTBA, and ITMG, external variables
and model complexity did not consistently beat the zero-return naive
benchmark. Small GRU gains show up in isolated folds and groups but vanish
across equities. Details in [`docs/results.md`](docs/results.md).
Machine-readable evidence in [`results/`](results/).

## Research question

> Do literature-supported external market variables improve next-day return
> forecasting for Indonesian coal equities compared with stock-history-only
> models?

Secondary questions: does extra complexity, from Naive to XGBoost to GRU
to Transformer, buy consistent out-of-sample improvement, and does anything
generalize across ADRO, PTBA, and ITMG?

## Why this problem

Indonesian coal equities are exposed to shared external shocks (coal
demand, exchange rates) and to their own trading dynamics. Raw prices are
near-persistent. Tomorrow's price is roughly today's price, so the project
forecasts next-day adjusted-close log returns
`r(t+1) = log(P(t+1)/P(t))`. Returns strip out persistence and force models
to earn every improvement.

## Research flow

```text
Literature review
      ↓
Research hypotheses
      ↓
Data acquisition
      ↓
Data audit & EDA
      ↓
Preprocessing
      ↓
Feature engineering
      ↓
Expanding-window validation
      ↓
Naive / XGBoost / GRU / Transformer
      ↓
Feature ablation
      ↓
Frozen final test
      ↓
Historical VaR
      ↓
Research conclusions
```

## Data

Yahoo Finance daily snapshots for `ADRO.JK`, `PTBA.JK`, `ITMG.JK`, and
`IDR=X` over `[2016-01-01, 2026-05-01)`. See [`docs/data.md`](docs/data.md)
for provenance and [`docs/eda.md`](docs/eda.md) for findings. Raw snapshots
stay local; `results/data_manifest.json` records SHA-256 hashes so every
result traces to its exact dataset.

## Exploratory analysis

`notebooks/01_data_audit.ipynb` (coverage, validity) and
`notebooks/02_eda.ipynb` (returns, volatility, lagged relationships,
ADRO/AADI event window). Key facts: returns are near-zero-mean and
heavy-tailed; USD/IDR moves are an order of magnitude calmer than equity
returns; ADRO shows a late-2024 regime shift carried as a limitation.

## Features

| Group | Content | Hypothesis |
| --- | --- | --- |
| E0 | own return lags + rolling volatility | baseline |
| E1 | E0 + lagged USD/IDR | H1 |
| E2 | E0 + lagged peer returns (exploratory) | H2 |
| E3 | E0 + USD/IDR + peers | H3 |

Every external input uses a one-observation availability lag with backward
as-of alignment. No future information enters. No coal-price feature exists
yet. One enters only with a release-dated history and a publication-timing
rule.

## Models

Zero-return Naive, then XGBoost with bounded candidates, then a small GRU
with early stopping, then a compact one-layer Transformer. The Transformer
is a complexity experiment, not an expected winner. Every family sees at
most the previous 20 trading days. Full contract in
[`docs/methodology.md`](docs/methodology.md).

## Validation design

Three expanding development folds (validate 2020, 2021, 2022) with per-fold
refit preprocessing, aggregated by mean RMSE for all decisions. The final
test `[2023-01-01, 2026-05-01)` is evaluated once, after freezing. Test
rows never influence selection. See
[`docs/research-design.md`](docs/research-design.md).

## Results

Per-question results in [`docs/results.md`](docs/results.md):

1. Dataset and EDA findings
2. Validation stability across folds
3. Naive baseline
4. Model-complexity comparison
5. External-feature ablation
6. Cross-equity consistency
7. Final-test results
8. Secondary VaR
9. Main research findings

## Main findings

- XGBoost lost to naive on test RMSE in every comparison.
- Transformer validation gains died on test.
- The best GRU delta anywhere was ADRO E1 at −0.000008. Noise.
- USD/IDR and peer features never helped twice in a row.
- My reading: more information and more complexity do not automatically
  produce better out-of-sample forecasts. I report the misses instead of
  burying them.

## Risk analysis

One-day 95% unconditional historical-simulation VaR over adjusted-close log
returns. It is a secondary descriptive analysis. It uses no forecasts and
drives no selection. See `results/risk_summary.csv`.

## Reproduce the experiment

Use the default/global Python environment:

```bash
pip install -r requirements.lock
pip install -e .
pytest tests/ -q
```

Runs (first run downloads snapshots; later runs reuse the validated cache):

```bash
# Single target/group, naive + XGBoost
coal-forecast --target ADRO.JK --feature-group E0
# Full comparison for all targets (E3 included)
coal-forecast --target all --feature-group E3 --model all
# Full expanding-window research experiment (writes results/*.csv + manifests)
python3 scripts/run_expanding_validation.py
```

`results/` holds the committed machine-readable evidence: `summary.csv`,
`fold_metrics.csv`, `final_test_metrics.csv`, `feature_ablation.csv`,
`risk_summary.csv`, `run_manifest.json`, `data_manifest.json`, and
`figures/`.

## Limitations

Yahoo revisions, small daily samples, one market, three equities, the
ADRO/AADI break, timestamp uncertainty, no causal claims, no trading
evaluation. Full list in [`docs/limitations.md`](docs/limitations.md).

## Origin

I started from an ADRO GRU closing-price experiment
(`adro gru haqi.ipynb`, kept local and unchanged). Its honest result was
that USD/IDR did not automatically help. I rebuilt it as a reproducible
multi-equity research pipeline with return targets, three equities,
ablation, and temporal validation.

## References

[`docs/literature-review.md`](docs/literature-review.md) and
[`references/references.bib`](references/references.bib).
