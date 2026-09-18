from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data import DataValidationError, Snapshot


TARGET_COLUMN = "target_next_day_log_return"


@dataclass(frozen=True)
class SequenceDataset:
    dates: pd.Series
    features: np.ndarray
    targets: np.ndarray
    channel_names: tuple[str, ...]
    lookback: int

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise DataValidationError(
                "Sequence features must have shape (rows, lookback, channels)."
            )
        expected_shape = (len(self.dates), self.lookback, len(self.channel_names))
        if self.features.shape != expected_shape:
            raise DataValidationError(
                f"Sequence features have shape {self.features.shape}; expected {expected_shape}."
            )
        if self.targets.shape != (len(self.dates),):
            raise DataValidationError(
                f"Sequence targets have shape {self.targets.shape}; expected ({len(self.dates)},)."
            )
        if not np.isfinite(self.features).all() or not np.isfinite(self.targets).all():
            raise DataValidationError("Sequence features and targets must be finite.")


def _feature_frame(
    snapshot: Snapshot,
    prefix: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int] = (),
    availability_lag: int = 0,
) -> pd.DataFrame:
    frame = snapshot.frame[["Date", "Adj Close"]].copy()
    returns = np.log(frame["Adj Close"]).diff()
    available_returns = returns.shift(availability_lag)
    output = pd.DataFrame({"Date": frame["Date"]})
    for lag in lags:
        output[f"{prefix}_return_lag_{lag}"] = available_returns.shift(lag - 1)
    for window in rolling_windows:
        output[f"{prefix}_vol_{window}"] = available_returns.rolling(window, min_periods=window).std(ddof=0)
    return output


def _return_channel(snapshot: Snapshot, prefix: str, availability_lag: int) -> pd.DataFrame:
    channel = _feature_frame(
        snapshot,
        prefix=prefix,
        lags=(1,),
        availability_lag=availability_lag,
    )
    return channel.rename(columns={f"{prefix}_return_lag_1": f"{prefix}_return"})


def _add_asof_features(
    dataset: pd.DataFrame,
    source_features: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    aligned = pd.merge_asof(
        dataset[["Date"]].sort_values("Date"),
        source_features.sort_values("Date"),
        on="Date",
        direction="backward",
        allow_exact_matches=True,
    )
    for column in feature_columns:
        dataset[column] = aligned[column].to_numpy()


def build_feature_dataset(
    snapshots: dict[str, Snapshot],
    target_symbol: str,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, list[str]]:
    if target_symbol not in snapshots:
        raise DataValidationError(f"No validated snapshot is available for target {target_symbol}.")
    target = snapshots[target_symbol].frame
    dataset = pd.DataFrame({"Date": target["Date"]})
    log_close = np.log(target["Adj Close"])
    dataset[TARGET_COLUMN] = log_close.shift(-1) - log_close

    own = _feature_frame(
        snapshots[target_symbol],
        prefix="own",
        lags=feature_config.return_lags,
        rolling_windows=feature_config.rolling_windows,
    )
    feature_columns = [column for column in own.columns if column != "Date"]
    for column in feature_columns:
        dataset[column] = own[column].to_numpy()

    if feature_config.group == "E1":
        if "IDR=X" not in snapshots:
            raise DataValidationError("Feature group E1 requires an IDR=X snapshot.")
        fx = _feature_frame(
            snapshots["IDR=X"],
            prefix="usd_idr",
            lags=feature_config.return_lags,
            rolling_windows=feature_config.rolling_windows,
            availability_lag=feature_config.usd_idr_availability_lag,
        )
        fx_columns = [column for column in fx.columns if column != "Date"]
        _add_asof_features(dataset, fx, fx_columns)
        feature_columns.extend(fx_columns)

    if feature_config.group == "E2":
        peers = [peer for peer in feature_config.peers if peer != target_symbol]
        if not peers:
            raise DataValidationError(
                f"Feature group E2 has no peer symbols for target {target_symbol}."
            )
        for peer in peers:
            if peer not in snapshots:
                raise DataValidationError(f"Feature group E2 requires peer snapshot {peer}.")
            peer_key = peer.replace(".", "_").replace("=", "_")
            peer_features = _feature_frame(
                snapshots[peer],
                prefix=f"peer_{peer_key}",
                lags=feature_config.return_lags,
                availability_lag=feature_config.peer_availability_lag,
            )
            peer_columns = [column for column in peer_features.columns if column != "Date"]
            _add_asof_features(dataset, peer_features, peer_columns)
            feature_columns.extend(peer_columns)

    if dataset["Date"].duplicated().any():
        raise DataValidationError(f"{target_symbol} feature dataset contains duplicate dates.")
    dataset = dataset.sort_values("Date").reset_index(drop=True)
    dataset = dataset.dropna(subset=[*feature_columns, TARGET_COLUMN]).reset_index(drop=True)
    if dataset.empty:
        raise DataValidationError(
            f"{target_symbol} has no complete rows after target and feature construction."
        )
    values = dataset[[*feature_columns, TARGET_COLUMN]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError(f"{target_symbol} feature dataset contains non-finite values.")
    return dataset[["Date", *feature_columns, TARGET_COLUMN]], feature_columns


def build_sequence_dataset(
    snapshots: dict[str, Snapshot],
    target_symbol: str,
    feature_config: FeatureConfig,
    lookback: int,
) -> SequenceDataset:
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise DataValidationError("Sequence lookback must be a positive integer.")
    if lookback > 20:
        raise DataValidationError("Sequence lookback must be <= 20.")
    if target_symbol not in snapshots:
        raise DataValidationError(f"No validated snapshot is available for target {target_symbol}.")

    target = snapshots[target_symbol].frame
    dataset = pd.DataFrame({"Date": target["Date"]})
    log_close = np.log(target["Adj Close"])
    dataset[TARGET_COLUMN] = log_close.shift(-1) - log_close
    channel_columns: list[str] = []

    own = _return_channel(snapshots[target_symbol], "own", availability_lag=0)
    _add_asof_features(dataset, own, ["own_return"])
    channel_columns.append("own_return")

    if feature_config.group == "E1":
        if "IDR=X" not in snapshots:
            raise DataValidationError("Feature group E1 requires an IDR=X snapshot.")
        fx = _return_channel(
            snapshots["IDR=X"],
            "usd_idr",
            availability_lag=feature_config.usd_idr_availability_lag,
        )
        _add_asof_features(dataset, fx, ["usd_idr_return"])
        channel_columns.append("usd_idr_return")

    if feature_config.group == "E2":
        peers = [peer for peer in feature_config.peers if peer != target_symbol]
        if not peers:
            raise DataValidationError(
                f"Feature group E2 has no peer symbols for target {target_symbol}."
            )
        for peer in peers:
            if peer not in snapshots:
                raise DataValidationError(f"Feature group E2 requires peer snapshot {peer}.")
            peer_key = peer.replace(".", "_").replace("=", "_")
            channel = f"peer_{peer_key}_return"
            peer_returns = _return_channel(
                snapshots[peer],
                f"peer_{peer_key}",
                availability_lag=feature_config.peer_availability_lag,
            )
            _add_asof_features(dataset, peer_returns, [channel])
            channel_columns.append(channel)

    dataset = dataset.sort_values("Date").reset_index(drop=True)
    dataset = dataset.dropna(subset=[*channel_columns, TARGET_COLUMN]).reset_index(drop=True)
    if len(dataset) < lookback:
        raise DataValidationError(
            f"Insufficient sequence history for {target_symbol}: found {len(dataset)} complete "
            f"target rows, need at least {lookback} for lookback {lookback}."
        )
    values = dataset[channel_columns].to_numpy(dtype=np.float32)
    targets = dataset[TARGET_COLUMN].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise DataValidationError(
            f"{target_symbol} sequence data contains non-finite values after alignment."
        )

    end_indices = range(lookback - 1, len(dataset))
    windows = np.stack([values[end - lookback + 1 : end + 1] for end in end_indices])
    labels = np.asarray([targets[end] for end in end_indices], dtype=np.float32)
    dates = dataset["Date"].iloc[list(end_indices)].reset_index(drop=True)
    expected_shape = (len(dates), lookback, len(channel_columns))
    if windows.shape != expected_shape:
        raise DataValidationError(
            f"Sequence tensor has shape {windows.shape}; expected {expected_shape}."
        )
    return SequenceDataset(dates, windows, labels, tuple(channel_columns), lookback)
