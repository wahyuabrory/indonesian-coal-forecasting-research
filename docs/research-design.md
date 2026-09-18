# Research design

## Problem

Indonesian coal equities may respond to their own historical behavior and to
external market information. This project tests whether those external
variables contain useful information for forecasting the next trading-day
return.

## Target and horizon

The target is the next observed trading-day adjusted-close log return:

```text
target(t) = log(AdjClose(t+1)) - log(AdjClose(t))
```

Raw prices are near-persistent. Tomorrow's price is roughly today's price,
so a model can post a small price error while learning nothing. Returns
strip that away. A price can be reconstructed later as
`P(t+1) = P(t) * exp(r(t+1))` when needed.

## Research questions

1. Do literature-supported external variables improve next-day return
   forecasting compared with stock-history-only models?
2. Does extra complexity, from Naive to XGBoost to GRU to Transformer,
   buy consistent out-of-sample improvement?
3. Are results consistent across ADRO, PTBA, and ITMG?

## Hypotheses

| ID | Statement | Test |
| --- | --- | --- |
| H1 | USD/IDR information improves forecasting over stock history alone. | E1 vs E0 |
| H2 | Peer-equity information improves forecasting over stock history alone (exploratory). | E2 vs E0 |
| H3 | Combining supported external variables adds further value. | E3 vs E0/E1/E2 |
| H4 | Sequential deep learning beats simpler baselines. | GRU vs Naive/XGBoost |
| H5 | A compact Transformer improves on GRU enough to justify its complexity. | Transformer vs GRU |

H2 is exploratory: industry spillover literature (Moskowitz and Grinblatt,
1999; Hou, 2007) supports testing it, but the evidence is monthly US data,
not daily Indonesian coal data.

## Sampling rule

Targets are selected before final-test results are interpreted:

```text
Indonesian listed coal equities
+ sufficient daily history from the study start (2016-01-01)
+ continuous usable price data
+ same market (IDX)
+ required external-data availability (IDR=X coverage)
= ADRO.JK, PTBA.JK, ITMG.JK
```

The rule is stated here so the three stocks cannot be read as chosen for
favorable results.

## Scope

In scope: next-day log returns; Naive, XGBoost, GRU, compact Transformer;
E0 to E3 ablation; expanding-window validation with a frozen final test;
historical VaR as a secondary descriptive analysis.

Out of scope: news sentiment, LLMs, portfolio optimization, trading
strategies, GARCH benchmarks, large architectures, dashboards.

## Decision rules

- Feature and model decisions use aggregated development-fold metrics only,
  one frozen candidate per family.
- Frozen models refit on all pre-2023 development data (deep models use the
  median fold best epoch) before the single final-test scoring.
- The frozen evaluation period (`[2023-01-01, 2026-05-01)`) is evaluated
  once, after all decisions are frozen. It never influences selection.
- A feature is useful only with repeated evidence: better mean validation
  RMSE across folds and consistent behavior on the final test across
  equities. One favorable run is not enough.
- Model comparison is a pipeline comparison under a shared information
  horizon, at most the previous 20 trading days. It is not a pure
  architecture comparison. XGBoost reads engineered summaries while
  GRU and Transformer read raw sequences over the same horizon.
