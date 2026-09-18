"""Risk tests: unconditional VaR boundary and determinism."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from coal_forecasting.config import load_config
from coal_forecasting.data import Snapshot, validate_snapshot
from coal_forecasting.risk import calculate_historical_var


def test_risk_is_deterministic_and_unconditional():
    config = load_config(Path("configs/baseline.toml"))
    dates = pd.date_range("2016-01-04", "2026-04-29", freq="B")
    rng = np.random.default_rng(5)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    frame = validate_snapshot(
        pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices}),
        "ITMG.JK", date(2016, 1, 1), date(2026, 5, 1),
    )
    snap = Snapshot("ITMG.JK", frame, Path("/tmp/i.csv"), "t", "n")
    first = calculate_historical_var(
        snap, "ITMG.JK", "E0", "all", "gru", "lookback5_hidden16", config
    )
    second = calculate_historical_var(
        snap, "ITMG.JK", "E3", "baseline", "zero_return_naive", None, config
    )
    assert first["log_return_quantile"] == second["log_return_quantile"]
    assert first["risk_used_for_model_selection"] is False
    assert 0 <= second["final_test_breach_rate"] <= 1
