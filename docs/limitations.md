# Limitations

These limits apply to the method and the verified results. The results are descriptive research outputs, not causal evidence.

## Structural breaks and corporate actions

ADRO and AADI have a structural break around late 2024. The runs do not model the restructuring, ticker history, corporate-action interpretation, or separate pre-event and post-event regimes. A relationship learned before the break may not hold after it. The same warning applies to unobserved changes in PTBA or ITMG.

Adjusted close reduces some price-series problems, but it does not resolve every corporate-action or ticker-history question. The results should not be read as stable pre/post ADRO evidence.

## Data source and revisions

Yahoo Finance is practical for daily snapshots and cache reuse. It is mutable. Historical rows, adjusted prices, and calendar coverage can change between retrievals. The metadata records the snapshot path and retrieval timestamp for each run, but it cannot recreate a provider revision that is no longer available.

The project uses daily observations. A row is treated as available after the target equity's close. The label is the next observed target-equity trading-day return. Exchange holidays, different market hours, delayed external publication, and timestamp ambiguity can still affect availability. E1 and E2 use a conservative one-observation availability lag, but that rule is not proof of perfect information timing.

## Features and omitted information

HBA is not included. HBA is relevant to Indonesian coal research, but this project does not have a release-dated HBA history with a publication-time rule. Backfilling a later-known HBA value would leak information into earlier rows. The current feature groups test own history, USD/IDR, and peer returns only.

The model does not include news, order flow, macroeconomic releases, production data, weather, transaction costs, liquidity, or other market information. Omitted variables can affect both the target and the included features.

## Evaluation design

Development uses three expanding folds (2020, 2021, 2022 validation years)
with a frozen final test `[2023-01-01, 2026-05-01)`. Each validation year
has roughly 240 scored rows and the test has 787. Small samples for
comparing several models. Daily sequence windows can overlap, so row counts
do not equal independent observations.

The runs compare several model families and candidates across twelve target/group combinations (E3 included). This creates multiple opportunities for an isolated low error. The hindsight test-best label in [results.md](results.md) is only a descriptive comparison. It is not a valid selection rule and does not correct for multiple comparisons.

Test metrics remain frozen after validation selection. That protects the reported selection procedure, but one fixed test period is still not enough to support a general performance claim.

## Model and result interpretation

No feature group shows consistent out-of-sample evidence against the naive
benchmark in any of the twelve comparisons. The largest improvement
anywhere was very small and did not generalize. XGBoost shows no
consistent out-of-sample evidence. Transformer validation improvements did
not persist consistently on the final test.

The naive model predicts zero next-day log return. Its low RMSE does not mean that returns are predictable. Directional accuracy and RMSE answer different questions, and neither measures economic value after costs.

No result identifies a causal effect. The observational design does not control for all confounding, establish an intervention, or show that a feature causes a return.

## Risk calculation

The reported VaR is a one-day unconditional historical-simulation estimate from adjusted-close log returns. It is not conditioned on forecasts, model residuals, volatility forecasts, or a selected model. A historical quantile does not guarantee future coverage. Structural breaks, revisions, non-stationarity, tail changes, and the short evaluation design can make it stale.

The breach rate is a descriptive count in the final-test period. It is not a risk-model validation certificate or a forecast of future losses.
