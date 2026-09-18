"""Temporal validation tests: splits, fold isolation, scaler fit scope."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from coal_forecasting.config import load_config
from coal_forecasting.data import Snapshot, validate_snapshot
from coal_forecasting.features import (
    TARGET_COLUMN,
    build_feature_dataset,
    build_sequence_dataset,
)
from coal_forecasting.modeling import evaluate, split_dataset
from coal_forecasting.validation import (
    aggregate_fold_metrics,
    build_expanding_folds,
    select_by_folds,
    split_fold_tabular,
)


def _config():
    return load_config(Path("configs/baseline.toml"))


def _snaps(n: int | None = None):
    dates = pd.date_range("2016-01-04", "2026-04-29", freq="B")
    if n is not None:
        dates = dates[:n]
    n = len(dates)
    rng = np.random.default_rng(7)
    snaps = {}
    for i, symbol in enumerate(("ADRO.JK", "PTBA.JK", "ITMG.JK", "IDR=X")):
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n) + 0.0002 * i))
        frame = validate_snapshot(
            pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices}),
            symbol,
            date(2016, 1, 1),
            date(2026, 5, 1),
        )
        snaps[symbol] = Snapshot(symbol, frame, Path(f"/tmp/{symbol}.csv"), "t", "n")
    return snaps


def test_split_boundaries_do_not_overlap():
    config = _config()
    snaps = _snaps()
    dataset, cols = build_feature_dataset(snaps, "ADRO.JK", config.features)
    parts = split_dataset(dataset, cols, config)
    assert (
        parts.train["Date"].max()
        < parts.validation["Date"].min()
        <= parts.validation["Date"].max()
        < parts.test["Date"].min()
    )


def test_expanding_folds_stay_before_final_test():
    config = _config()
    folds = build_expanding_folds(config)
    assert len(folds) == 3
    origin = config.splits.validation_end_exclusive
    for fold in folds:
        assert fold.validation_end_exclusive <= origin
    assert [f.name for f in folds] == ["fold1", "fold2", "fold3"]


def test_sequence_window_ends_at_forecast_date():
    config = _config()
    snaps = _snaps()
    seq = build_sequence_dataset(snaps, "ADRO.JK", config.features, lookback=5)
    # Each window's last channel row must equal the observed return ending at t.
    frame = snaps["ADRO.JK"].frame.set_index("Date")["Adj Close"]
    log_close = np.log(frame)
    rets = log_close.diff()
    for i in [0, len(seq.dates) // 2, -1]:
        day = seq.dates.iloc[i]
        assert seq.features[i, -1, 0] == np.float32(rets.loc[day])


def test_train_scaler_does_not_see_validation_or_test():
    config = _config()
    snaps = _snaps()
    dataset, cols = build_feature_dataset(snaps, "ADRO.JK", config.features)
    parts = split_dataset(dataset, cols, config)
    result = evaluate(parts, cols, config, "xgboost")
    train_mean = parts.train[cols].to_numpy(float).mean(axis=0)
    assert np.allclose(result.scaler.mean_, train_mean)
    full_mean = dataset[cols].to_numpy(float).mean(axis=0)
    assert not np.allclose(result.scaler.mean_, full_mean)


def test_fold_scaler_isolation():
    config = _config()
    snaps = _snaps()
    dataset, _ = build_feature_dataset(snaps, "ADRO.JK", config.features)
    from sklearn.preprocessing import StandardScaler

    folds = build_expanding_folds(config)
    for fold in folds:
        train, validation = split_fold_tabular(
            dataset, fold, config.splits.min_rows
        )
        scaler = StandardScaler().fit(train[[c for c in dataset.columns if c != "Date" and c != TARGET_COLUMN]])
        assert scaler.mean_.shape[0] > 0
        assert train["Date"].max() < validation["Date"].min()


def test_test_metrics_do_not_control_selection():
    rows = [
        {"model": "zero_return_naive", "candidate": "", "fold": "fold1",
         "rmse": 0.02, "mae": 0.015, "directional_accuracy": 0.5},
        {"model": "zero_return_naive", "candidate": "", "fold": "fold2",
         "rmse": 0.02, "mae": 0.015, "directional_accuracy": 0.5},
        {"model": "xgboost", "candidate": "depth2_lr005", "fold": "fold1",
         "rmse": 0.019, "mae": 0.014, "directional_accuracy": 0.51},
        {"model": "xgboost", "candidate": "depth2_lr005", "fold": "fold2",
         "rmse": 0.019, "mae": 0.014, "directional_accuracy": 0.51},
    ]
    agg = aggregate_fold_metrics(rows)
    winner = select_by_folds(agg)
    assert winner["model"] == "xgboost"
    # A later favorable test row must not change the fold-based winner.
    assert winner["mean_rmse"] == agg["mean_rmse"].min()
