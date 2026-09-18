# Indonesian coal forecasting research

This repository contains the accepted daily-return baseline and compact GRU and Transformer experiments for `ADRO.JK`, `PTBA.JK`, and `ITMG.JK`. The verified outputs include naive, XGBoost, GRU, and Transformer metrics plus a separate unconditional historical-risk summary.

The pipeline uses Yahoo Finance daily snapshots, adjusted-close log returns, leak-safe lagged features, a zero-return benchmark, bounded XGBoost, and optional CPU PyTorch sequence models. The original `adro gru haqi.ipynb` file is kept unchanged. HBA backfilling, charts, and model artifact dumps remain out of scope.

Read the [methodology](docs/methodology.md), [results](docs/results.md), [literature review](docs/literature-review.md), and [limitations](docs/limitations.md) before interpreting a result.

## Install

Use the default/global Python environment, not `uv`:

```bash
python -m pip install -e .
```

The package requires Python 3.11 or newer. `yfinance` is used only when a requested snapshot is not cached. XGBoost is needed for the default model mode. PyTorch is optional and is required only for `gru`, `transformer`, and `all`:

```bash
python -m pip install -e '.[deep]'
```

## Run

The checked-in config is [`configs/baseline.toml`](configs/baseline.toml). It owns the symbols, inclusive/exclusive dates, chronological split boundaries, lags, seed, bounded XGBoost candidates, and two candidates for each deep model.

```bash
# One target, E0 stock-history features, naive plus XGBoost
coal-forecast --target ADRO.JK --feature-group E0

# All three targets
coal-forecast --target all --feature-group E0

# Reuse existing snapshots without network access
coal-forecast --target all --feature-group E1 --model baseline

# Run the two configured GRU candidates
coal-forecast --target ADRO.JK --feature-group E0 --model gru

# Run the two configured Transformer candidates
coal-forecast --target ADRO.JK --feature-group E0 --model transformer

# Compare the naive baseline and best candidate from each model family
coal-forecast --target all --feature-group E0 --model all
```

The first run downloads four daily snapshots (`ADRO.JK`, `PTBA.JK`, `ITMG.JK`, and `IDR=X`) into `data/yahoo_cache/`. Later runs reuse a cache only when its JSON sidecar matches the active symbol, dates, and interval. A mismatch stops the run; use `--refresh` to replace it.

Each target writes `metrics.csv`, `metrics.json`, `metadata.json`, and `risk.json` below `outputs/<target>/<feature-group>/`. Metadata records snapshot retrieval times, split dates, train-only scaling rules, framework versions, candidate parameters, best epochs, and trainable parameter counts. In `both` mode, validation compares the naive baseline with the best XGBoost candidate. In `all` mode, validation compares the naive baseline and the best candidate from XGBoost, GRU, and Transformer, while all four frozen test metrics remain visible. Test rows are not used for selection.

`risk.json` is a secondary descriptive analysis. It reports one-day unconditional historical-simulation VaR from adjusted-close log returns before the final-test origin, then counts breaches in the final-test returns. It is not conditioned on forecasts and is not used for model selection, so its VaR values repeat across feature groups and model modes for the same snapshot and split. Verified risk values and their caveats are documented in [`docs/results.md`](docs/results.md).

See [`docs/methodology.md`](docs/methodology.md) for the data contract, anti-leakage rules, and risk calculation definition. Do not interpret a run as evidence that ADRO relationships are stable across the ADRO/AADI structural break.
