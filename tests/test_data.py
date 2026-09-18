"""Data trust-boundary tests: target timing, cache identity, VaR calibration."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coal_forecasting.config import load_config
from coal_forecasting.data import (
    DataValidationError,
    Snapshot,
    load_snapshot,
    validate_snapshot,
)
from coal_forecasting.features import TARGET_COLUMN, build_feature_dataset
from coal_forecasting.risk import calculate_historical_var


def _prices(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2016-01-04", periods=n, freq="B")
    drift = np.linspace(0, 0.05, n)
    prices = start * np.exp(drift + 0.001 * np.sin(np.arange(n)))
    return pd.DataFrame(
        {"Date": dates, "Close": prices, "Adj Close": prices}
    )


def _snapshot(symbol: str, n: int = 60) -> Snapshot:
    frame = validate_snapshot(
        _prices(n), symbol, date(2016, 1, 1), date(2026, 5, 1)
    )
    return Snapshot(symbol, frame, Path(f"/tmp/{symbol}.csv"), "test", "now")


def _config():
    return load_config(Path("configs/baseline.toml"))


def test_target_is_next_day_return():
    config = _config()
    snaps = {
        "ADRO.JK": _snapshot("ADRO.JK", 60),
        "PTBA.JK": _snapshot("PTBA.JK", 60),
        "ITMG.JK": _snapshot("ITMG.JK", 60),
        "IDR=X": _snapshot("IDR=X", 60),
    }
    dataset, _ = build_feature_dataset(snaps, "ADRO.JK", config.features)
    adj = snaps["ADRO.JK"].frame["Adj Close"].to_numpy(float)
    log_close = np.log(adj)
    expected = log_close[1:] - log_close[:-1]
    row = dataset.iloc[30]
    idx = int(
        np.flatnonzero(
            snaps["ADRO.JK"].frame["Date"].to_numpy() == np.datetime64(row["Date"])
        )[0]
    )
    assert row[TARGET_COLUMN] == pytest.approx(float(expected[idx]))


def test_no_future_feature_timestamp():
    config = _config()
    snaps = {
        "ADRO.JK": _snapshot("ADRO.JK", 60),
        "PTBA.JK": _snapshot("PTBA.JK", 60),
        "ITMG.JK": _snapshot("ITMG.JK", 60),
        "IDR=X": _snapshot("IDR=X", 60),
    }
    dataset, cols = build_feature_dataset(snaps, "ADRO.JK", config.features)
    adj = snaps["ADRO.JK"].frame.set_index("Date")["Adj Close"]
    for _, row in dataset.head(10).iterrows():
        t = row["Date"]
        r_t = float(np.log(adj.loc[t]) - np.log(adj.loc[:t].iloc[-2]))
        assert row["own_return_lag_1"] == pytest.approx(r_t)


def test_external_availability_lag():
    from dataclasses import replace

    config = _config()
    snaps = {
        "ADRO.JK": _snapshot("ADRO.JK", 60),
        "PTBA.JK": _snapshot("PTBA.JK", 60),
        "ITMG.JK": _snapshot("ITMG.JK", 60),
        "IDR=X": _snapshot("IDR=X", 60),
    }
    fc = replace(config.features, group="E1", usd_idr_availability_lag=1)
    dataset, cols = build_feature_dataset(snaps, "ADRO.JK", fc)
    assert any(c.startswith("usd_idr_") for c in cols)
    # With lag=1 the FX return used at equity date t must be older than the
    # same-day FX move: check the first usable row differs from lag=0.
    fc0 = replace(config.features, group="E1", usd_idr_availability_lag=0)
    dataset0, _ = build_feature_dataset(snaps, "ADRO.JK", fc0)
    merged = dataset.merge(
        dataset0[["Date", "usd_idr_return_lag_1"]].rename(
            columns={"usd_idr_return_lag_1": "lag0"}
        ),
        on="Date",
    )
    assert not np.allclose(
        merged["usd_idr_return_lag_1"].to_numpy(float),
        merged["lag0"].to_numpy(float),
    )


def test_cache_metadata_mismatch_fails(tmp_path):
    import json

    config = _config()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    object.__setattr__(
        config.data, "cache_dir", cache_dir
    ) if False else None
    # Write a cache whose sidecar requests different dates.
    frame = _prices(30)
    path = cache_dir / "adro_jk.csv"
    frame.to_csv(path, index=False)
    (cache_dir / "adro_jk.metadata.json").write_text(
        json.dumps(
            {
                "symbol": "ADRO.JK",
                "requested_start_inclusive": "2000-01-01",
                "requested_end_exclusive": "2000-02-01",
                "interval": "1d",
                "retrieved_at_utc": "2020-01-01T00:00:00+00:00",
            }
        )
    )
    from dataclasses import replace

    cfg = replace(config.data, cache_dir=cache_dir)
    import dataclasses

    cfg2 = dataclasses.replace(config, data=cfg)
    with pytest.raises(DataValidationError, match="does not match"):
        load_snapshot("ADRO.JK", cfg2)


def test_var_calibration_excludes_test_returns():
    config = _config()
    dates = pd.date_range("2016-01-04", "2026-04-29", freq="B")
    rng = np.random.default_rng(9)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    frame = validate_snapshot(
        pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices}),
        "ADRO.JK", date(2016, 1, 1), date(2026, 5, 1),
    )
    snap = Snapshot("ADRO.JK", frame, Path("/tmp/a.csv"), "t", "n")
    payload = calculate_historical_var(
        snap, "ADRO.JK", "E0", "baseline", "zero_return_naive", None, config
    )
    assert payload["risk_used_for_model_selection"] is False
    assert payload["conditioned_on_forecasts"] is False
    # Calibration must end before the final-test origin; breach window starts there.
    assert (
        payload["calibration_return_date_end_exclusive"]
        == config.splits.validation_end_exclusive.isoformat()
    )
    assert (
        payload["final_test_return_date_start_inclusive"]
        == config.splits.validation_end_exclusive.isoformat()
    )
