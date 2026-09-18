# Results

## Scope

These results cover `ADRO.JK`, `PTBA.JK`, and `ITMG.JK` with three feature groups:

- `E0`: target-equity return lags and rolling volatility.
- `E1`: `E0` plus conservatively lagged USD/IDR return features.
- `E2`: `E0` plus conservatively lagged peer-equity return features.

The target is the next observed target-equity adjusted-close log return. RMSE is in log-return units, and lower is better. The tables use the verified files under `outputs/<target>/<feature-group>/`. Each run uses `model_mode=all`, selects on validation RMSE, and keeps the test partition out of selection.

The results are descriptive. They do not show that any model has stable predictive power. This is one fixed validation/test split, not multi-fold stability evidence.

## Data snapshot and split

The metadata records a requested interval of `[2016-01-01, 2026-05-01)`. The configured chronological splits are:

| Partition | Configured interval | Observed dates | Rows |
| --- | --- | --- | ---: |
| Train | `[2016-01-01, 2022-01-01)` | 2016-02-01 to 2021-12-30 | 1,487 for ADRO/PTBA; 1,488 for ITMG |
| Validation | `[2022-01-01, 2023-01-01)` | 2022-01-03 to 2022-12-30 | 246 |
| Test | `[2023-01-01, 2026-05-01)` | 2023-01-02 to 2026-04-29 | 787 |

The snapshot retrieval timestamps in the metadata are:

| Symbol | Retrieved at UTC |
| --- | --- |
| `ADRO.JK` | `2026-09-18T07:58:38.532557+00:00` |
| `PTBA.JK` | `2026-09-18T07:58:39.368953+00:00` |
| `ITMG.JK` | `2026-09-18T07:58:40.422291+00:00` |
| `IDR=X` | `2026-09-18T07:58:41.239144+00:00` |

The recorded model framework versions are XGBoost `3.4.1` and PyTorch `2.14.0+cpu`. The configured seed is `42`.

## Validation selection and test hindsight

Validation selection is the model recorded as selected in `metadata.json`. The final column is different. It is the model with the lowest test RMSE after looking at the test rows. It is a hindsight comparison, not a selected model.

| Target | Group | Validation-selected model | Hindsight test-best model |
| --- | --- | --- | --- |
| ADRO | E0 | Transformer, `lookback10_d32_heads4_ff64` | Naive |
| ADRO | E1 | GRU, `lookback10_hidden32` | GRU |
| ADRO | E2 | GRU, `lookback10_hidden32` | Naive |
| PTBA | E0 | GRU, `lookback5_hidden16` | GRU |
| PTBA | E1 | GRU, `lookback10_hidden32` | Naive |
| PTBA | E2 | Transformer, `lookback10_d32_heads4_ff64` | Naive |
| ITMG | E0 | Transformer, `lookback5_d16_heads2_ff32` | Naive |
| ITMG | E1 | GRU, `lookback5_hidden16` | Naive |
| ITMG | E2 | Transformer, `lookback5_d16_heads2_ff32` | Naive |

The validation-selected models are not test winners by default. The test-best column must not be used to choose a model.

## Test RMSE

Values are rounded to eight decimal places. The source JSON files retain full precision.

| Target | Group | Naive | XGBoost | GRU | Transformer |
| --- | --- | ---: | ---: | ---: | ---: |
| ADRO | E0 | 0.02703467 | 0.02765760 | 0.02703510 | 0.02721939 |
| ADRO | E1 | 0.02703467 | 0.02763784 | 0.02701679 | 0.02718165 |
| ADRO | E2 | 0.02703467 | 0.02730095 | 0.02708917 | 0.02722194 |
| PTBA | E0 | 0.01996016 | 0.02042201 | 0.01992306 | 0.02002513 |
| PTBA | E1 | 0.01996016 | 0.02032870 | 0.02001561 | 0.02010131 |
| PTBA | E2 | 0.01996016 | 0.02046868 | 0.01997575 | 0.02001677 |
| ITMG | E0 | 0.01746045 | 0.01768252 | 0.01757579 | 0.01769942 |
| ITMG | E1 | 0.01746045 | 0.01771034 | 0.01756577 | 0.01810996 |
| ITMG | E2 | 0.01746045 | 0.01783510 | 0.01759846 | 0.01881598 |

External features do not improve test RMSE consistently. XGBoost does not beat the naive model on test RMSE in any of the nine comparisons. The only lower test RMSE values are small GRU gains for ADRO E1 and PTBA E0. They do not repeat across companies or feature groups. Transformer validation gains do not persist on test.

## Unconditional historical VaR

`risk.json` reports a one-day, 95% unconditional historical-simulation estimate. Calibration uses returns dated in `[2016-01-01, 2023-01-01)`. Breaches use returns dated in `[2023-01-01, 2026-05-01)`. The risk calculation uses adjusted-close log returns, not forecasts, and it does not affect model selection.

| Target | VaR log-return magnitude | Simple loss fraction | Estimated loss on IDR 1,000,000 | Final-test breaches |
| --- | ---: | ---: | ---: | ---: |
| ADRO | 0.04284938 | 0.04194432 | IDR 41,944.32 | 26/788, 0.03299492 |
| PTBA | 0.03872644 | 0.03798616 | IDR 37,986.16 | 18/788, 0.02284264 |
| ITMG | 0.03993283 | 0.03914602 | IDR 39,146.02 | 20/788, 0.02538071 |

These are unconditional historical estimates. They are not forecast-conditional risk and do not promise a coverage rate. Values repeat across feature groups for a target because the calculation uses the same target snapshot and split.

## Output files

For each target and group, see:

- `metrics.json` for validation and frozen-test metrics.
- `metadata.json` for snapshot identity, dates, split rows, selection, and framework versions.
- `risk.json` for the secondary risk calculation.

The method contract is in [`methodology.md`](methodology.md). The data and model caveats are in [`limitations.md`](limitations.md).
