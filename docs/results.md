# Results

Machine-readable source of truth: [`results/`](../results/) (`summary.csv`,
`fold_metrics.csv`, `validation_winners.csv`, `final_test_metrics.csv`,
`feature_ablation.csv`, `risk_summary.csv`, `run_manifest.json`,
`data_manifest.json`). Tables below are verified against those files. Figures:
[`results/figures/`](../results/figures/).

Scope: `ADRO.JK`, `PTBA.JK`, `ITMG.JK`; groups E0 (own history), E1 (+USD/IDR),
E2 (+peers, exploratory), E3 (combined). Target: next-day adjusted-close log
return. Selection: mean fold RMSE over three expanding development folds;
final test `[2023-01-01, 2026-05-01)` evaluated once, after freezing.

## 1. Dataset and EDA findings

Requested interval `[2016-01-01, 2026-05-01)`; observed 2016-01-04 to
2026-04-29 (ADRO/PTBA 2,541 rows, ITMG 2,542). No duplicates, no missing or
non-finite prices. Returns are near-zero-mean, heavy-tailed (ADRO σ 0.0286,
PTBA 0.0253, ITMG 0.0250); USD/IDR changes are about 4x calmer
(σ 0.0065, ratios 3.9 to 4.4). Lagged external correlations are weak. ADRO shows a late-2024
restructuring regime shift, carried as a limitation. Full detail:
[`eda.md`](eda.md).

## 2. Validation stability

Mean validation RMSE of the frozen per-family candidate (raw candidate
rows stay in `results/fold_metrics.csv`). Bold marks the best family mean
per row.

| Target | Group | Naive | XGBoost (candidate) | GRU (candidate) | Transformer (candidate) |
| --- | --- | ---: | --- | ---: | --- |
| ADRO | E0 | 0.030829 | 0.030958 (depth3_lr003) | **0.030751** (lookback10_hidden32) | 0.030754 (lookback10_d32_heads4_ff64) |
| ADRO | E1 | 0.030829 | 0.031189 (depth2_lr005) | **0.030760** (lookback5_hidden16) | 0.030855 (lookback10_d32_heads4_ff64) |
| ADRO | E2 | 0.030829 | 0.031085 (depth2_lr005) | **0.030461** (lookback10_hidden32) | 0.030629 (lookback10_d32_heads4_ff64) |
| ADRO | E3 | 0.030829 | 0.031384 (depth2_lr005) | **0.030511** (lookback10_hidden32) | 0.030796 (lookback5_d16_heads2_ff32) |
| PTBA | E0 | 0.027120 | 0.027080 (depth2_lr005) | 0.027039 (lookback5_hidden16) | **0.026980** (lookback10_d32_heads4_ff64) |
| PTBA | E1 | 0.027120 | 0.027158 (depth2_lr005) | **0.026985** (lookback5_hidden16) | 0.027009 (lookback5_d16_heads2_ff32) |
| PTBA | E2 | 0.027120 | 0.027136 (depth2_lr005) | **0.026857** (lookback10_hidden32) | 0.026967 (lookback10_d32_heads4_ff64) |
| PTBA | E3 | 0.027120 | 0.027232 (depth2_lr005) | **0.026770** (lookback10_hidden32) | 0.026943 (lookback5_d16_heads2_ff32) |
| ITMG | E0 | 0.029358 | 0.029682 (depth2_lr005) | **0.029179** (lookback10_hidden32) | 0.029245 (lookback5_d16_heads2_ff32) |
| ITMG | E1 | 0.029358 | 0.029532 (depth3_lr003) | **0.029063** (lookback5_hidden16) | 0.029198 (lookback5_d16_heads2_ff32) |
| ITMG | E2 | 0.029358 | 0.029314 (depth2_lr005) | 0.029146 (lookback10_hidden32) | **0.029128** (lookback5_d16_heads2_ff32) |
| ITMG | E3 | 0.029358 | 0.029400 (depth2_lr005) | **0.029102** (lookback10_hidden32) | 0.029157 (lookback5_d16_heads2_ff32) |

GRU wins 10 of 12 groups on folds; XGBoost did not outperform Naive
consistently. Full detail
in `results/validation_winners.csv`.

## 3. Naive baseline

The zero-return forecast is competitive everywhere because daily mean
returns sit near zero. Any model has to beat it to claim value. Almost none
does (see section 7).

## 4. Model-complexity comparison

All four frozen families, refit on pre-2023 development data, scored once
on the final test (E0):

| Target | Naive | XGBoost | GRU | Transformer |
| --- | ---: | ---: | ---: | ---: |
| ADRO | 0.027035 | 0.027625 | 0.027030 | 0.027205 |
| PTBA | **0.019960** | 0.020393 | 0.019927 | 0.020077 |
| ITMG | **0.017460** | 0.017622 | 0.017515 | 0.017626 |

My read of the ladder: plain ML with XGBoost showed no consistent
out-of-sample evidence over Naive. GRU matched Naive without clearly
beating it (ADRO differs by 5e-06). Transformer complexity showed no
consistent out-of-sample evidence.

## 5. External-feature ablation

Same family, E1/E2/E3 against E0. Validation delta uses the frozen
candidate's mean fold RMSE; test delta uses the refit frozen model.
Negative deltas favor the external group.

| Target | Family | E1 val / test | E2 val / test | E3 val / test |
| --- | --- | ---: | ---: | ---: |
| ADRO | XGBoost | +0.000231 / -0.000191 | +0.000128 / -0.000427 | +0.000426 / -0.000266 |
| ADRO | GRU | +0.000009 / -0.000019 | -0.000290 / +0.000064 | -0.000240 / +0.000058 |
| ADRO | Transformer | +0.000101 / +0.000083 | -0.000125 / -0.000162 | +0.000042 / -0.000111 |
| PTBA | XGBoost | +0.000078 / -0.000174 | +0.000057 / +0.000002 | +0.000152 / -0.000063 |
| PTBA | GRU | -0.000054 / +0.000086 | -0.000181 / +0.000083 | -0.000269 / +0.000090 |
| PTBA | Transformer | +0.000029 / +0.000006 | -0.000013 / -0.000053 | -0.000037 / -0.000032 |
| ITMG | XGBoost | -0.000150 / +0.000104 | -0.000368 / +0.000185 | -0.000283 / +0.000143 |
| ITMG | GRU | -0.000116 / +0.000085 | -0.000033 / +0.000135 | -0.000077 / +0.000122 |
| ITMG | Transformer | -0.000047 / +0.000083 | -0.000116 / +0.000079 | -0.000088 / +0.000413 |

H1 on USD/IDR: small validation improvements in some families, but the
test deltas flip sign across families and equities. No consistent
out-of-sample evidence. H2 on peers: same story, with the clearest
example in ADRO GRU E2, where a -0.000290 validation delta becomes
+0.000064 on test. H3 on the combo: nothing on top of E0 that survives
the test.

## 6. Cross-equity consistency

No feature group improves all three equities within any family; no model
beats Naive on all three. The improvements were very small and did not
generalize, with sign flips between validation and test inside the same
family.

## 7. Final-test results

Full table in `results/final_test_metrics.csv`: 48 rows, all four families
for every target and feature group. Each frozen model was refit on
pre-2023 development data (deep models at their median fold best epoch)
and scored once on the frozen evaluation period. No family outperformed
Naive; differences were small and not consistent across equities or
feature groups.

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

- H1 to H3: no consistent out-of-sample evidence that USD/IDR, peer, or
  combined external information improves next-day return forecasts. Small
  validation improvements did not persist consistently on the final test.
- H4 to H5: no consistent out-of-sample evidence that sequence deep
  learning or Transformer complexity beats the naive benchmark.
- What holds up: more information and more complexity do not automatically
  produce better out-of-sample financial forecasts.
- These are descriptive results from three folds and one test period. They
  are not causal claims and not general performance claims.

## Output files

- `results/fold_metrics.csv` holds every candidate by fold validation metric.
- `results/validation_winners.csv` holds the frozen candidate per family.
- `results/final_test_metrics.csv` holds all four refit families on the test.
- `results/feature_ablation.csv` holds same-family deltas against E0.
- `results/risk_summary.csv` holds VaR per target.
- `results/run_manifest.json` holds code commit, dirty flag, config and
  lockfile hashes, seed, and library versions.
- `results/data_manifest.json` holds snapshot hashes and retrieval times.

Method contract: [`methodology.md`](methodology.md). Caveats:
[`limitations.md`](limitations.md).
