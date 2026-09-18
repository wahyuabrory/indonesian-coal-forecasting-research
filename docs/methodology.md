# Forecasting methodology

This document is the stable method contract for the baseline and optional GRU and Transformer experiments. The run-specific dates, lags, seed, and candidates live in [`configs/baseline.toml`](../configs/baseline.toml). Research questions and decision rules are in [`research-design.md`](research-design.md); data provenance in [`data.md`](data.md); EDA in [`eda.md`](eda.md); preprocessing in [`preprocessing.md`](preprocessing.md). Verified metrics and risk summaries are in [`results.md`](results.md). See [`literature-review.md`](literature-review.md) for design evidence and [`limitations.md`](limitations.md) for interpretation limits.

## Data contract

The loader requests Yahoo Finance daily data with `start` inclusive and `end` exclusive. It keeps `Date`, `Close`, and `Adj Close` for every configured symbol. A snapshot is accepted only when:

- all required columns exist;
- dates parse, are unique, sorted, and fall inside the requested interval after trimming;
- `Close` and `Adj Close` are numeric, finite, and strictly positive.

Snapshots are CSV files with JSON sidecars containing the symbol, request interval, daily interval, and retrieval time. A cached file is reused only when all four sidecar identity fields match the active config. A missing sidecar or any mismatch names the cache and stops the run; `--refresh` is required to replace it. The CSV is validated again on every run. A missing required column, an empty interval, or an invalid price stops the run with an error.

The forecast row is timestamped by the target equity's observed trading date. The code treats its values as available after that day's close. The target is the next observed target-equity trading day's adjusted-close log return:

```text
y_t = log(AdjClose[t+1]) - log(AdjClose[t])
```

`Close` is retained for the data contract and audit trail. The baseline target and features use `Adj Close`.

## Feature groups

For an equity, define the observed one-day return ending at row `t` as:

```text
r_t = log(AdjClose[t]) - log(AdjClose[t-1])
```

`E0` contains the target equity's `r_t` history. `return_lag_1` is `r_t`, `return_lag_2` is `r_(t-1)`, and so on for lags 1, 2, 3, 5, and 10. Rolling volatility uses only returns available through row `t`, with windows 5 and 20.

`E1` adds the same lagged returns and rolling volatilities for `IDR=X`. Its configured availability lag shifts the exchange-rate observations back before they are aligned to the equity calendar. The default is one source observation. Alignment uses the most recent source date at or before the equity date.

`E2` adds peer-equity lagged returns. The peer availability lag applies the same conservative one-observation rule. The target equity is removed from its own peer list. E2 is exploratory: industry spillover literature motivates testing it, but does not establish it for daily Indonesian coal data.

`E3` combines E1 and E2: own history plus lagged USD/IDR plus lagged peer returns. It answers H3 on combined value. It is also the last group added. A future coal-price feature gets a clean group redesign instead of endless E4 and E5 additions.

Every feature in row `t` is built from information no later than the feature timestamp. The target uses row `t+1` only as the label. No coal feature is present. HBA can be added only after a release-dated input and publication-time rule exist, so this pipeline does not backfill HBA.

## GRU and Transformer sequences

The GRU and Transformer use the same target dates, half-open split boundaries, and external availability alignment as the tabular builder. The shared return-channel builder creates one channel per observed return source, then applies the configured one-observation availability lag and backward as-of alignment for external sources. It does not create a second time rule.

For lookback `L`, the sequence tensor has shape `(rows, L, channels)`. The final row in each window is target-equity date `t`. Its channels contain only returns available at or before `t`. The label is the next observed target-equity adjusted-close log return. E0 contains the target equity return channel. E1 adds the conservatively lagged USD/IDR return channel. E2 adds conservatively lagged peer-equity return channels. Windows without enough complete history fail with an explicit insufficient-history error.

The two configured GRU candidates use one GRU layer, lookbacks no longer than 20, hidden sizes no larger than 32, Adam, modest weight decay, at most 100 epochs, validation early stopping, and the restored best validation checkpoint. Training uses CPU and the configured deterministic seed.

The two configured Transformer candidates use one encoder layer, lookbacks no longer than 20, `d_model` no larger than 32, feed-forward widths no larger than 64, valid head divisibility, Adam, at most 100 epochs, validation early stopping, and the restored best validation checkpoint. Each model adds a learned positional embedding before the encoder. The run records candidate parameters, the winning framework version, best epoch, and trainable parameter count. It does not save checkpoints or candidate model dumps.

## Model-input fairness

This is a forecasting-pipeline comparison under a shared information
horizon, not a pure architecture comparison. Every model family may use
information from at most the previous 20 trading days: XGBoost via
engineered lags (1, 2, 3, 5, 10) and rolling windows (5, 20), GRU and
Transformer via raw return sequences with lookback ≤ 20. Representations
differ; the horizon does not.

## Splits and preprocessing

Single-split rows are assigned by date using the configured half-open
intervals:

```text
train:      [start_inclusive, train_end_exclusive)
validation: [train_end_exclusive, validation_end_exclusive)
test:       [validation_end_exclusive, test_end_exclusive)
```

The test partition remains untouched until the validation winner is fixed. Each partition must meet `splits.min_rows`. The tabular `StandardScaler` is fit on train features only. Validation and test features are transformed with that frozen scaler. For GRU and Transformer candidates, the feature scaler is fit on training sequence values only and the target scaler is fit on training labels only. Validation and test are transform-only. No imputation is performed; incomplete rows or sequences are removed before splitting, and non-finite values fail the data boundary.

## Expanding-window validation

`src/coal_forecasting/validation.py` defines three development folds with an
expanding training origin at `data.start_inclusive`:

```text
Fold 1  Train: [2016-01-01, 2020-01-01)  Validate: [2020-01-01, 2021-01-01)
Fold 2  Train: [2016-01-01, 2021-01-01)  Validate: [2021-01-01, 2022-01-01)
Fold 3  Train: [2016-01-01, 2022-01-01)  Validate: [2022-01-01, 2023-01-01)
Final test (untouched): [2023-01-01, 2026-05-01)
```

Fold dates can be overridden with `[[expanding_folds]]` in the config, but
every fold must satisfy `train_end <= validation_end <= final-test origin`.
Each fold refits preprocessing on its own training data, trains, and scores
validation. Raw candidate results live in `results/fold_metrics.csv`.

Selection freezes one candidate per family using mean fold RMSE inside that
family: one XGBoost, one GRU, one Transformer, plus the Naive benchmark.
Families never compete on folds. The frozen models are refit on all
pre-2023 development data, then scored once on the untouched final test, so
`results/final_test_metrics.csv` holds all four families for every target
and feature group.

XGBoost refits directly on the full development rows. GRU and Transformer
retrain for the median of their fold best epochs, since no held-out set
remains inside dev for early stopping and the test must stay untouched.
The epoch counts come from folds only. Per-family winners are recorded in
`results/validation_winners.csv`.

## Secondary historical VaR

Each target run also writes `risk.json`. This is a secondary descriptive analysis, not a forecasting result. The checked-in risk configuration uses confidence `0.95` and a portfolio value of `1,000,000` IDR. The loader rejects confidence values outside `(0, 1)`, non-finite values, and non-positive notionals.

For target adjusted closes, the one-day return dated `d` is the return ending on the observed date `d`:

```text
r_d = log(Adj Close[d]) - log(Adj Close[previous observed target date])
```

Calibration uses only finite returns with dates in `[data.start_inclusive, splits.validation_end_exclusive)`. The final-test origin is `splits.validation_end_exclusive`, so no return dated on or after that boundary enters the quantile. Breaches use only finite returns in `[splits.validation_end_exclusive, splits.test_end_exclusive)`. Both sets must meet `splits.min_rows`. A breach is a realized final-test return strictly below the calibration threshold.

The lower tail is explicit and deterministic:

```text
alpha = 1 - confidence
q = numpy.quantile(calibration_returns, alpha, method="linear")
VaR log-return magnitude = max(0, -q)
simple loss fraction = max(0, 1 - exp(q))
estimated loss IDR = portfolio_value_idr * simple loss fraction
```

`risk.json` records the calibration and final-test date ranges and row counts, confidence, alpha, `q`, both loss measures, notional, estimated loss, breach count, and breach rate. It also records the target, feature group, model mode, selected model and candidate, snapshot path, return definition, and the quantile method. `risk_used_for_model_selection` is always `false`. The VaR calculation is unconditional. Its quantile and loss conversion do not use forecasts, model residuals, scaling, early stopping, or selection state. Therefore the VaR values repeat across feature groups and model modes for the same target snapshot and split, although each output keeps its run context. Verified values are reported in [`results.md`](results.md).

## Models and selection

The zero-return naive model predicts `0.0` for every next-day log return. XGBoost uses only the small candidate list in the config, a fixed random seed, one thread, and bounded tree settings. Each candidate is fit on train and scored on validation using the configured selection metric. Error metrics minimize; directional accuracy maximizes.

With `model_mode="both"`, validation compares the zero-return baseline with the best XGBoost candidate. With `model_mode="gru"` or `model_mode="transformer"`, selection is limited to that model's two configured candidates. With `model_mode="all"`, validation compares the zero-return naive, best XGBoost, best GRU, and best Transformer. With `model_mode="xgboost"`, selection is limited to XGBoost candidates. With `model_mode="baseline"`, the baseline is selected directly. The selected model, candidate, parameters, framework version, best epoch, and parameter count in metadata describe the frozen validation winner.

After selection is frozen, the test partition is evaluated. In `both` mode, both frozen baseline and best-XGBoost test metrics are reported. In `all` mode, frozen baseline, best-XGBoost, best-GRU, and best-Transformer test metrics are all reported, even when another model wins selection. No test metric, test prediction, or test row affects selection, and the code never retrains on test data. Outputs report MAE, RMSE, and directional accuracy. Directional accuracy compares the signs of actual and predicted returns; a zero prediction matches only an actual zero.

## Timestamp and structural-break caveats

The timestamp assumption is explicit: a row dated `t` represents information available after the target equity's daily close, and the next target-equity trading row is the forecast horizon. Same-day external observations are not treated as available for E1, E2, or E3 runs because their availability lags are one source observation.

The ADRO/AADI restructuring and ticker history create a structural break around late 2024. This baseline does not model that event, corporate-action interpretation, or a stable pre/post regime. Results need a separate event-aware treatment before they are used for an economic claim.
