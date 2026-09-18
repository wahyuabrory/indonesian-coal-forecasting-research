from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig


REQUIRED_SNAPSHOT_COLUMNS = ("Date", "Close", "Adj Close")


class DataValidationError(ValueError):
    """Raised when a Yahoo snapshot fails the data trust boundary."""


class DataDependencyError(RuntimeError):
    """Raised when downloading needs an unavailable optional dependency."""


@dataclass(frozen=True)
class Snapshot:
    symbol: str
    frame: pd.DataFrame
    path: Path
    source: str
    retrieved_at_utc: str

    def metadata(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "path": str(self.path),
            "source": self.source,
            "retrieved_at_utc": self.retrieved_at_utc,
        }


def snapshot_slug(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_").lower()


def _normalise_dates(values: pd.Series, field: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        bad_rows = int(parsed.isna().sum())
        raise DataValidationError(f"{field} contains {bad_rows} invalid date value(s).")
    return parsed.dt.tz_localize(None).dt.normalize()


def validate_snapshot(
    frame: pd.DataFrame,
    symbol: str,
    start_inclusive: date,
    end_exclusive: date,
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SNAPSHOT_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{symbol} snapshot is missing required columns: {', '.join(missing)}. "
            "Provide a snapshot with Date, Close, and Adj Close, then retry."
        )
    if frame.empty:
        raise DataValidationError(
            f"{symbol} snapshot is empty. Check the configured dates or provide a valid cache."
        )

    # A caller may supply an index level named like a required column. Clear it
    # before using label-based pandas operations at this trust boundary.
    clean = frame.loc[:, REQUIRED_SNAPSHOT_COLUMNS].copy().reset_index(drop=True)
    clean["Date"] = _normalise_dates(clean["Date"], f"{symbol}.Date")
    if clean["Date"].duplicated().any():
        raise DataValidationError(f"{symbol} snapshot contains duplicate dates.")

    for column in ("Close", "Adj Close"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        values = clean[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataValidationError(f"{symbol}.{column} contains missing or non-finite values.")
        if (values <= 0).any():
            raise DataValidationError(f"{symbol}.{column} must contain only positive prices.")

    clean = clean.sort_values("Date").reset_index(drop=True)
    start = pd.Timestamp(start_inclusive)
    end = pd.Timestamp(end_exclusive)
    clean = clean.loc[(clean["Date"] >= start) & (clean["Date"] < end)].reset_index(drop=True)
    if clean.empty:
        raise DataValidationError(
            f"{symbol} has no valid rows in [{start_inclusive.isoformat()}, {end_exclusive.isoformat()})."
        )
    return clean


def _download(symbol: str, config: AppConfig) -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataDependencyError(
            "Yahoo dependency is unavailable while downloading "
            f"{symbol}. Install the project dependencies or provide a validated cached "
            "snapshot with its JSON sidecar."
        ) from exc

    try:
        raw = yf.download(
            symbol,
            start=config.data.start_inclusive.isoformat(),
            end=config.data.end_exclusive.isoformat(),
            interval=config.data.interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            timeout=30,
        )
    except Exception as exc:
        raise DataValidationError(
            f"Yahoo download failed for {symbol}: {exc} Check the network or provide a "
            "validated cache, then retry."
        ) from exc
    if raw.empty:
        raise DataValidationError(
            f"Yahoo returned no rows for {symbol}. Check the configured dates or network, "
            "then retry."
        )

    columns: dict[str, pd.Series] = {}
    for field in ("Close", "Adj Close"):
        matches = [
            column
            for column in raw.columns
            if column == field or (isinstance(column, tuple) and field in column)
        ]
        if not matches:
            raise DataValidationError(
                f"Yahoo response for {symbol} is missing required column {field!r}. "
                "Check the Yahoo response or provide a validated cache."
            )
        selected = raw.loc[:, matches[0]]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        columns[field] = selected
    downloaded = pd.DataFrame({"Date": raw.index, **columns}).reset_index(drop=True)
    downloaded.index.name = None
    return downloaded, "Yahoo Finance"


def _read_cache(path: Path, symbol: str, config: AppConfig) -> tuple[pd.DataFrame, str, str]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(
            f"Could not read cache {path}: {exc} Run with --refresh to replace this cache."
        ) from exc

    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise DataValidationError(
            f"Cache {path} has no JSON sidecar at {metadata_path}. "
            "Run with --refresh to replace this cache with a reproducible snapshot."
        )
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(
            f"Cache sidecar {metadata_path} is unreadable: {exc} "
            "Run with --refresh to replace this cache."
        ) from exc

    expected = {
        "symbol": symbol,
        "requested_start_inclusive": config.data.start_inclusive.isoformat(),
        "requested_end_exclusive": config.data.end_exclusive.isoformat(),
        "interval": config.data.interval,
    }
    mismatches = [
        f"{field}={metadata.get(field)!r}, expected {value!r}"
        for field, value in expected.items()
        if metadata.get(field) != value
    ]
    if mismatches:
        details = "; ".join(mismatches)
        raise DataValidationError(
            f"Cache {path} metadata does not match the active config: {details}. "
            "Run with --refresh to replace this cache."
        )

    retrieved_at = str(metadata.get("retrieved_at_utc", ""))
    if not retrieved_at:
        raise DataValidationError(
            f"Cache sidecar {metadata_path} has no retrieval time. "
            "Run with --refresh to replace this cache."
        )
    return frame, "Yahoo Finance cached snapshot", retrieved_at


def load_snapshot(symbol: str, config: AppConfig, refresh: bool = False) -> Snapshot:
    cache_dir = config.data.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{snapshot_slug(symbol)}.csv"

    if path.exists() and not refresh:
        raw, source, retrieved_at = _read_cache(path, symbol, config)
    else:
        raw, source = _download(symbol, config)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        raw = validate_snapshot(
            raw,
            symbol,
            config.data.start_inclusive,
            config.data.end_exclusive,
        )
        raw.to_csv(path, index=False)
        metadata_path = path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "source": source,
                    "retrieved_at_utc": retrieved_at,
                    "requested_start_inclusive": config.data.start_inclusive.isoformat(),
                    "requested_end_exclusive": config.data.end_exclusive.isoformat(),
                    "interval": config.data.interval,
                    "rows": len(raw),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return Snapshot(symbol, raw, path, source, retrieved_at)

    try:
        clean = validate_snapshot(
            raw,
            symbol,
            config.data.start_inclusive,
            config.data.end_exclusive,
        )
    except DataValidationError as exc:
        raise DataValidationError(
            f"Cache {path} failed validation: {exc} Fix the cache or run with --refresh."
        ) from exc
    return Snapshot(symbol, clean, path, source, retrieved_at)


def load_snapshots(config: AppConfig, refresh: bool = False) -> dict[str, Snapshot]:
    return {
        symbol: load_snapshot(symbol, config, refresh=refresh)
        for symbol in config.data.symbols
    }
