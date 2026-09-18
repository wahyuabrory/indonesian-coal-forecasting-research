# Literature review

This review links each major design choice to prior research. Sources make a
variable eligible for testing; they do not prove it will improve forecasting.
Full verified records are in [`references/references.bib`](../references/references.bib).

## Why Indonesian coal equities

Indonesian coal companies are exposed to common external shocks through
export sales, coal prices, the rupiah exchange rate, and foreign demand.
Komara, Sinaga, and Andati (2019) model Indonesian listed coal companies and
find that external factors, coal prices, the rupiah exchange rate, and
China's GDP, influence coal export sales, firm value, and stock returns,
with the rupiah exchange rate hitting stock returns hardest.
That mechanism supports studying a set of Indonesian listed coal equities
together rather than one stock in isolation.

## Why ADRO, PTBA, and ITMG

The three targets are selected by a transparent sampling rule, not by a
paper claiming they are the best stocks (see
[`research-design.md`](research-design.md)): Indonesian listed coal equities
with sufficient daily history from the study start, continuous usable price
data, the same market, and required external-data availability. ADRO, PTBA,
and ITMG satisfy that rule. Sihotang and Munir (2021) also discuss these
companies alongside HBA and are cited here as background context only; they
do not determine the sampling decision.

## Why USD/IDR (feature group E1)

Putra and Robiyanto (2019) test commodity-price and USD/IDR effects on
Indonesian mining stock returns from 2011 to 2017 and report a clear
negative exchange-rate effect for several mining firms including ITMG and
PTBA. Komara, Sinaga, and Andati (2019) independently identify the rupiah
exchange rate as a dominant external driver of coal-company stock returns.
Together these sources make lagged USD/IDR information eligible for a
controlled test (H1). They do not imply the effect helps every target or
persists out of sample.

## Why peer equities (feature group E2, exploratory)

E2 tests whether lagged returns of fellow coal equities carry information
about the target. The mechanism is industry information diffusion with
delay. Moskowitz and Grinblatt (1999) document that past industry returns
forecast
individual stock returns; Hou (2007) attributes industry lead-lag effects to
slow information diffusion; Ali and Hirshleifer (2020) unify such momentum
spillovers through shared analyst coverage and investor inattention. These
results are monthly US evidence, not daily Indonesian coal evidence, so E2
is labeled exploratory rather than literature-driven. It is tested, not
assumed.

## Why a coal-price variable is excluded for now

Komara, Sinaga, and Andati (2019) support coal prices as a relevant input,
but relevance is not enough: a forecasting feature also needs a historical
value, a reference period, a release date, and a publication-timing rule.
The project has no release-dated HBA history, so backfilling a monthly value
would leak later-known information into earlier rows. The variable stays out
until a safe dataset exists. This is a documented limitation, not an
oversight.

## Target, validation, and baseline

Hyndman and Athanasopoulos (*Forecasting: Principles and Practice*, 3rd ed.)
support return transformations, strict time order, and simple benchmarks.
Next-day log returns are used because raw prices are near-persistent.
Tomorrow's price is roughly today's price, so small price errors can hide
the absence of real predictive information. Tashman (2000) motivates genuine
out-of-sample testing. Bergmeir et al. (2018) and Cerqueira et al. (2020)
support chronological evaluation and warn that one split proves nothing about
stability. That warning is why this project uses expanding windows, three
development folds plus one frozen final test. Makridakis et al. (2018, M4) support comparing
methods broadly instead of assuming a complex model wins, which is why the
zero-return naive forecast stays visible.

## Models

Chen and Guestrin (2016) support a small bounded XGBoost as the conventional
nonlinear test on engineered features. Cho et al. (2014) support the GRU as
the central sequential test; the project originates from a GRU experiment,
so GRU is the natural deep-learning baseline. Vaswani et al. (2017) define
the Transformer, while Zeng et al. (2023) show Transformers lose to simpler
models on some time-series tasks. So the compact Transformer here is a
complexity experiment, not an expected winner. Fischer and
Krauss (2018) provide context for sequence models in financial prediction
without validating this project's data or risk calculation.

## Evaluation

RMSE and MAE measure error size; directional accuracy measures sign
agreement. Neither proves economic value after transaction costs, so trading
evaluation stays out of scope. Feature conclusions require repeated temporal
evidence (consistent gains across folds and equities), not one favorable
run.

## References

Putra, A. R., & Robiyanto, R. (2019). The effect of commodity price changes
and USD/IDR exchange rate on Indonesian mining companies' stock return.
*Jurnal Keuangan dan Perbankan*, *23*(1), 97-108.
https://doi.org/10.26905/jkdp.v23i1.2084

Komara, A. A., Sinaga, B. M., & Andati, T. (2019). The impact of changes in
external and internal factors on financial performance and stock returns of
coal companies. *Jurnal Aplikasi Bisnis dan Manajemen*, *5*(3), 513.
https://doi.org/10.17358/jabm.5.3.513
