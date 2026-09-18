# Preprocessing

Pipeline order:

```text
raw snapshot
↓ validation (columns, dates, finite positive prices)
↓ date normalization and requested-interval trim
↓ Adj Close selection
↓ log-return transformation
↓ external-data alignment (backward as-of + availability lag)
↓ missing-row handling (drop, no imputation)
↓ feature generation
↓ temporal splitting
↓ train-only scaling
```

## Target

```text
target(t) = log(AdjClose(t+1)) - log(AdjClose(t))
```

The label uses row `t+1`; every feature in row `t` uses information no
later than `t`. The last trading day has no label and is dropped.

## Features

E0 (own history): return lags 1, 2, 3, 5, 10 and rolling volatility (pop
std) over windows 5 and 20, all from returns available through row `t`.

E1: E0 plus the same FX return/volatility set for IDR=X, shifted by the
configured availability lag (default 1 observation), aligned backward.

E2: E0 plus lagged peer-equity returns (same lags, same peer lag rule).
The target equity is excluded from its own peer list.

E3: E1 + E2 combined (own history + USD/IDR + peer returns).

No RSI, MACD, Bollinger, or other indicators. Each group maps to a stated
hypothesis, H1 to H3, and nothing else earned a place.

## Alignment and leakage controls

- Backward as-of merge only (`direction="backward"`, exact matches
  allowed). An equity row never sees a future external observation.
- One-observation availability lag on every external source.
- Sequence windows for GRU/Transformer end at forecast date `t`; the label
  is the next-day return. Max lookback 20, so every model family sees at
  most the previous 20 trading days (XGBoost via lags/volatility windows,
  deep models via raw sequences).
- Splits are half-open date intervals; rows belong to exactly one
  partition.

## Scaling

- Tabular scaler fit on training rows only; validation/test transform-only.
- GRU/Transformer: feature scaler fit on training sequence values only,
  target scaler fit on training labels only.
- Per-fold rule: in expanding-window validation each fold refits scalers on
  that fold's training data.

## Missing data

Dropped, never imputed. Non-finite values after construction fail the run.
