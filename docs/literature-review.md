# Literature review

This review links the main design choices to the supplied literature. The sources motivate tests and safeguards. They do not establish that a feature causes a stock return or that a model will generalize to these data.

## Indonesian mining context

[Putra and Robiyanto (2019)](https://doi.org/10.26905/jkdp.v23i1.2084) study commodity prices and USD/IDR in Indonesian mining. That context supports testing the USD/IDR feature group, `E1`, rather than assuming that the exchange rate helps every target.

[Sihotang and Munir (2021)](https://doi.org/10.47970/jml.v4i1.204) cover ADRO, PTBA, ITMG, and HBA. This supports using the three companies as related targets and treating HBA as a relevant candidate input. HBA is excluded from the current runs because a valid feature needs release-dated history and a publication-time rule. A later publication date must not be aligned to an earlier forecast row.

[Antono et al. (2019)](https://doi.org/10.5267/j.msl.2019.5.018) examine factors in Indonesian mining. This supports keeping the feature set explicit and treating omitted economic factors as possible confounders. It does not justify a causal interpretation of the return associations in this project.

Yahoo Finance is practical for reproducible daily snapshots. It is also mutable. Historical rows and adjusted values can change between retrievals. The pipeline therefore caches the requested snapshot, validates it again, and records retrieval timestamps and cache identity. Those controls identify the data used for a run. They do not turn Yahoo into an immutable historical archive.

## Time-series evaluation

[Tashman (2000)](https://doi.org/10.1016/S0169-2070(00)00065-0) motivates out-of-sample forecast tests. The project freezes validation selection before it evaluates the test partition.

[Bergmeir et al. (2018)](https://doi.org/10.1016/j.csda.2017.11.003) discuss cross-validation for time-series prediction. [Cerqueira et al. (2020)](https://doi.org/10.1007/s10994-020-05910-7) examine time-series model evaluation methods. Together they support chronological evaluation and caution against treating one split as stability evidence. These runs use one fixed validation/test split, not multi-fold stability evidence.

[Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*](https://otexts.com/fpp3/) provides the forecasting reference for transformations, time order, and simple benchmarks. [Makridakis et al., M4](https://doi.org/10.1016/j.ijforecast.2018.06.001) provide the M4 competition reference for broad comparisons across forecasting methods. These sources support keeping the zero-return naive model visible instead of treating a complex model as the default winner.

## Model families

[Chen and Guestrin](https://doi.org/10.1145/2939672.2939785) provide the XGBoost reference. It supports a bounded tree model for lag and volatility features. It does not imply that XGBoost should beat the naive benchmark in this sample.

[Cho et al.](https://doi.org/10.3115/v1/D14-1179) provide the GRU reference. It supports testing a gated recurrent model for ordered return sequences. The configured lookbacks remain small because the project has one daily series per target and one fixed split.

[Vaswani et al., *Attention Is All You Need*](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need) provide the Transformer reference. [Zeng et al. (2023)](https://doi.org/10.1609/aaai.v37i9.26317) provide a caution against assuming that Transformers are effective for every time-series forecasting task. The project therefore treats the Transformer as a comparison model. Its validation result is not evidence of test or future superiority.

## Regime and risk

[Fischer and Krauss](https://doi.org/10.1016/j.ejor.2017.11.054) provide context for using sequence models in financial-market prediction. Their paper does not establish the ADRO/AADI structural-break caveat or validate this project's historical VaR calculation. Those remain independent limitations of this data and method.

The resulting method is deliberately narrow. It tests return history, USD/IDR, and peer returns under release-safe timing. It reports the naive benchmark, keeps test selection separate, and labels the risk output as unconditional. It does not claim causal effects or stable relationships.

## References

The DOI and official links above are also recorded in [`references/references.bib`](../references/references.bib). Entries omit fields that were not verified for this project.
