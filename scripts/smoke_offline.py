"""Deterministic offline smoke test: data -> features -> split -> metrics.

Builds seeded synthetic snapshots covering the configured study period, so
CI exercises the real pipeline without network access or cached data.
Fails loudly instead of skipping. XGBoost + naive only; deep models are
covered by the unit tests with tiny synthetic series.

Usage: python3 scripts/smoke_offline.py [--config PATH]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coal_forecasting.config import load_config  # noqa: E402
from coal_forecasting.data import Snapshot, validate_snapshot  # noqa: E402
from coal_forecasting.features import build_feature_dataset  # noqa: E402
from coal_forecasting.modeling import evaluate, split_dataset  # noqa: E402

SMOKE_SEED = 20260918


def _synthetic_snapshot(symbol: str, seed: int) -> Snapshot:
    dates = pd.date_range("2016-01-04", "2026-04-29", freq="B")
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates))))
    frame = validate_snapshot(
        pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices}),
        symbol, date(2016, 1, 1), date(2026, 5, 1),
    )
    return Snapshot(symbol, frame, Path(f"synthetic/{symbol}.csv"),
                    "synthetic", "smoke")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    symbols = ("ADRO.JK", "PTBA.JK", "ITMG.JK", "IDR=X")
    snapshots = {s: _synthetic_snapshot(s, SMOKE_SEED + i)
                 for i, s in enumerate(symbols)}
    checked = 0
    for target in config.target_symbols:
        for group in ("E0", "E1"):
            feature_config = replace(config.features, group=group)
            dataset, cols = build_feature_dataset(snapshots, target, feature_config)
            parts = split_dataset(dataset, cols, config)
            result = evaluate(parts, cols, config, "both")
            assert result.selected_model in ("zero_return_naive", "xgboost")
            assert any(m["split"] == "test" for m in result.metrics)
            assert all(np.isfinite(m["rmse"]) for m in result.metrics)
            checked += 1
            print(f"smoke {target} {group}: {result.selected_model}")
    print(f"smoke passed: {checked} target/group combinations, seed {SMOKE_SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
