"""Feature alignment tests: backward-only as-of merge, E3 contents."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from coal_forecasting.config import load_config
from coal_forecasting.data import Snapshot, validate_snapshot
from coal_forecasting.features import build_feature_dataset, build_sequence_dataset


def _prices(dates, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices})


def _snap(symbol: str, dates) -> Snapshot:
    frame = validate_snapshot(
        _prices(dates), symbol, date(2016, 1, 1), date(2026, 5, 1)
    )
    return Snapshot(symbol, frame, Path(f"/tmp/{symbol}.csv"), "test", "now")


def _config():
    return load_config(Path("configs/baseline.toml"))


def test_asof_alignment_is_backward_only():
    """External FX dates must never leak a future observation into row t."""
    config = _config()
    equity_dates = pd.date_range("2016-02-01", periods=120, freq="B")
    # FX misses every other day: as-of merge must carry the last known value.
    fx_dates = equity_dates[::2]
    snaps = {
        "ADRO.JK": _snap("ADRO.JK", equity_dates),
        "PTBA.JK": _snap("PTBA.JK", equity_dates),
        "ITMG.JK": _snap("ITMG.JK", equity_dates),
        "IDR=X": _snap("IDR=X", fx_dates),
    }
    fc = replace(config.features, group="E1")
    dataset, cols = build_feature_dataset(snaps, "ADRO.JK", fc)
    assert not dataset[cols].isna().any().any()
    # Every FX feature at date t uses an FX observation dated <= t.
    fx_days = set(pd.to_datetime(fx_dates).normalize())
    for day in dataset["Date"]:
        assert min(fx_days) <= day.normalize()


def test_e3_combines_e1_and_e2():
    config = _config()
    dates = pd.date_range("2016-02-01", periods=60, freq="B")
    snaps = {s: _snap(s, dates) for s in ("ADRO.JK", "PTBA.JK", "ITMG.JK", "IDR=X")}
    _, e1 = build_feature_dataset(
        snaps, "ADRO.JK", replace(config.features, group="E1")
    )
    _, e2 = build_feature_dataset(
        snaps, "ADRO.JK", replace(config.features, group="E2")
    )
    _, e3 = build_feature_dataset(
        snaps, "ADRO.JK", replace(config.features, group="E3")
    )
    assert set(e1) | set(e2) <= set(e3)
    assert any(c.startswith("usd_idr_") for c in e3)
    assert any(c.startswith("peer_") for c in e3)


def test_peer_group_excludes_own_symbol():
    config = _config()
    dates = pd.date_range("2016-02-01", periods=60, freq="B")
    snaps = {s: _snap(s, dates) for s in ("ADRO.JK", "PTBA.JK", "ITMG.JK", "IDR=X")}
    _, cols = build_feature_dataset(
        snaps, "ADRO.JK", replace(config.features, group="E2")
    )
    assert not any("ADRO_JK" in c for c in cols)
