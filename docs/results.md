# Results

Machine-readable source of truth: [`results/`](../results/) (`summary.csv`,
`fold_metrics.csv`, `final_test_metrics.csv`, `feature_ablation.csv`,
`risk_summary.csv`, `run_manifest.json`, `data_manifest.json`). Tables below
are verified against those files. Figures:
[`results/figures/`](../results/figures/).

Scope: `ADRO.JK`, `PTBA.JK`, `ITMG.JK`; groups E0 (own history), E1 (+USD/IDR),
E2 (+peers, exploratory), E3 (combined). Target: next-day adjusted-close log
return. Selection: mean fold RMSE over three expanding development folds;
final test `[2023-01-01, 2026-05-01)` evaluated once, after freezing.

## 1. Dataset and EDA findings

Requested interval `[2016-01-01, 2026-05-01)`; observed 2016-01-04 to
2026-04-29 (ADRO/PTBA 2,541 rows, ITMG 2,542). No duplicates, no missing or
non-finite prices. Returns are near-zero-mean, heavy-tailed (ADRO σ 0.0286,
PTBA 0.0253, ITMG 0.0250); USD/IDR changes are an order of magnitude calmer
(σ 0.0065). Lagged external correlations are weak. ADRO shows a late-2024
restructuring regime shift, carried as a limitation. Full detail:
[`eda.md`](eda.md).

## 2. Validation stability

Mean validation RMSE across the three folds (lower is better):

| Target | Group | Naive | XGBoost | GRU | Transformer |
| --- | --- | ---: | ---: | ---: | ---: |
| ADRO | E0 | 0.030829 | 0.030986 | 0.030774 | 0.030763 |
| ADRO | E1 | 0.030829 | 0.031275 | 0.030776 | 0.030899 |
| ADRO | E2 | 0.030829 | 0.031087 | 0.030463 | 0.030630 |
| ADRO | E3 | 0.030829 | 0.031447 | 0.030576 | 0.030802 |
| PTBA | E0 | 0.027120 | 0.027252 | 0.027040 | 0.027007 |
| PTBA | E1 | 0.027120 | 0.027302 | 0.026990 | 0.027010 |
| PTBA | E2 | 0.027120 | 0.027177 | 0.026881 | 0.027061 |
| PTBA | E3 | 0.027120 | 0.027439 | 0.026831 | 0.027113 |
| ITMG | E0 | 0.029358 | 0.029769 | 0.029207 | 0.029337 |
| ITMG | E1 | 0.029358 | 0.029544 | 0.029080 | 0.029220 |
| ITMG | E2 | 0.029358 | 0.029363 | 0.029150 | 0.029181 |
| ITMG | E3 | 0.029358 | 0.029552 | 0.029162 | 0.029240 |

GRU wins most folds; XGBoost never beats Naive on mean fold RMSE.
Fold-selected winners per group are recorded in `results/summary.csv`.

## 3. Naive baseline

The zero-return forecast is competitive everywhere because daily mean
returns sit near zero. Any model has to beat it to claim value. Almost none
does (see section 7).

## 4. Model-complexity comparison

Fold-selected winner vs Naive on the frozen final test (E0):

| Target | Winner | Winner RMSE | Naive RMSE |
| --- | --- | ---: | ---: |
| ADRO | GRU lookback10_hidden32 | 0.027035 | 0.027035 |
| PTBA | Transformer lookback10_d32_heads4_ff64 | 0.020056 | 0.019960 |
| ITMG | GRU lookback10_hidden32 | 0.017493 | 0.017460 |

My read of the ladder: plain ML with XGBoost added nothing over Naive.
GRU matched Naive without clearly beating it. Transformer complexity paid
nothing (PTBA came out worse than Naive, and ITMG E2 printed 0.018816
against 0.017460).

## 5. External-feature ablation

Fold-selected model test RMSE and delta vs E0 (negative = improvement):

| Target | E0 | E1 Δ | E2 Δ | E3 Δ |
| --- | ---: | ---: | ---: | ---: |
| ADRO | 0.027035 | −0.000008 | +0.000054 | +0.000124 |
| PTBA | 0.020056 | −0.000016 | −0.000045 | −0.000044 |
| ITMG | 0.017493 | +0.000072 | +0.001323 | +0.000107 |

H1 on USD/IDR: deltas at 1e-05 scale with mixed signs. No evidence of
value. H2 on peers: the ITMG E2 transformer falls apart (+0.0013), and
nothing else repeats. H3 on the combo: nothing on top of E0.

## 6. Cross-equity consistency

No feature group improves all three equities; no model beats Naive on all
three. The only negative deltas (ADRO E1, PTBA E1/E2/E3) sit at or under
5e-05. That is noise scale, and another equity contradicts each one.

## 7. Final-test results

Full table in `results/final_test_metrics.csv`. Every fold-selected winner
was scored once on the untouched test alongside the naive reference. In all
12 target/group comparisons the winner fails to beat Naive by any
meaningful margin (best case: ADRO E1 −0.000008).

## 8. Secondary VaR

One-day 95% unconditional historical VaR (calibration before 2023-01-01):

| Target | VaR magnitude | Loss on IDR 1,000,000 | Test breaches |
| --- | ---: | ---: | --- |
| ADRO | 0.042849 | IDR 41,944.32 | 26/788, 0.0330 |
| PTBA | 0.038726 | IDR 37,986.16 | 18/788, 0.0228 |
| ITMG | 0.039933 | IDR 39,146.02 | 20/788, 0.0254 |

Descriptive only. The VaR uses no forecasts, drives no selection, and
guarantees no coverage. Values repeat across groups by construction.

## 9. Main research findings

- H1 to H3: no consistent evidence that USD/IDR, peer, or combined
  external information improves next-day return forecasts.
- H4 to H5: no consistent evidence that sequence deep learning or
  Transformer complexity beats the naive benchmark out of sample.
- What holds up: more information and more complexity do not automatically
  produce better out-of-sample financial forecasts.
- These are descriptive results from three folds and one test period. They
  are not causal claims and not general performance claims.

## Output files

- `results/fold_metrics.csv` holds every candidate by fold validation metric.
- `results/final_test_metrics.csv` holds frozen test metrics for winners plus naive.
- `results/feature_ablation.csv` holds test RMSE deltas against E0.
- `results/risk_summary.csv` holds VaR per target.
- `results/run_manifest.json` holds commit, config, seed, and library versions.
- `results/data_manifest.json` holds snapshot hashes and retrieval times.

Method contract: [`methodology.md`](methodology.md). Caveats:
[`limitations.md`](limitations.md).
