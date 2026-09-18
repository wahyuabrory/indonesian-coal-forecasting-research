"""Model-selection tests: validation-only selection, frozen test reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coal_forecasting.config import load_config
from coal_forecasting.data import Snapshot, validate_snapshot
from coal_forecasting.features import build_feature_dataset
from coal_forecasting.modeling import evaluate, split_dataset
from datetime import date


def _config():
    return load_config(Path("configs/baseline.toml"))


def test_selection_ignores_test_partition():
    from copy import deepcopy

    config = _config()
    dates = pd.date_range("2016-01-04", "2026-04-29", freq="B")
    rng = np.random.default_rng(3)
    snaps = {}
    for symbol in ("ADRO.JK", "PTBA.JK", "ITMG.JK", "IDR=X"):
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, len(dates))))
        frame = validate_snapshot(
            pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices}),
            symbol, date(2016, 1, 1), date(2026, 5, 1),
        )
        snaps[symbol] = Snapshot(symbol, frame, Path("/tmp/x.csv"), "t", "n")
    dataset, cols = build_feature_dataset(snaps, "ADRO.JK", config.features)
    parts = split_dataset(dataset, cols, config)
    first = evaluate(parts, cols, config, "both")
    # Corrupt the test targets: validation selection must not change.
    corrupted = deepcopy(parts.test)
    corrupted["target_next_day_log_return"] = 5.0
    from coal_forecasting.modeling import Partitions

    parts2 = Partitions(parts.train, parts.validation, corrupted)
    second = evaluate(parts2, cols, config, "both")
    assert first.selected_model == second.selected_model
    assert first.selected_candidate == second.selected_candidate


def test_smoke_baseline_runs_without_network():
    config = _config()
    dates = pd.date_range("2016-01-04", "2026-04-29", freq="B")
    rng = np.random.default_rng(11)
    snaps = {}
    for symbol in ("ADRO.JK", "PTBA.JK", "ITMG.JK", "IDR=X"):
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, len(dates))))
        frame = validate_snapshot(
            pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices}),
            symbol, date(2016, 1, 1), date(2026, 5, 1),
        )
        snaps[symbol] = Snapshot(symbol, frame, Path("/tmp/x.csv"), "t", "n")
    dataset, cols = build_feature_dataset(snaps, "PTBA.JK", config.features)
    parts = split_dataset(dataset, cols, config)
    result = evaluate(parts, cols, config, "baseline")
    assert result.selected_model == "zero_return_naive"
    assert any(m["split"] == "test" for m in result.metrics)
