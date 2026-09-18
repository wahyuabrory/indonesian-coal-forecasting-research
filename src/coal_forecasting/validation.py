"""Expanding-window temporal validation.

Single-split evaluation (train/validation/test from the config) remains the
CLI default. This module adds the research design upgrade: several expanding
development folds with one frozen final test.

Fold convention (half-open intervals, same as modeling.split_dataset):

    Fold 1  Train: [start, 2020-01-01)  Validate: [2020-01-01, 2021-01-01)
    Fold 2  Train: [start, 2021-01-01)  Validate: [2021-01-01, 2022-01-01)
    Fold 3  Train: [start, 2022-01-01)  Validate: [2022-01-01, 2023-01-01)
    Final test (untouched): [2023-01-01, test_end)

Fold dates can be overridden in config with [[expanding_folds]] tables.
Preprocessing inside every fold follows the same rule as the single split:
fit scalers on that fold's training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .config import AppConfig
from .features import TARGET_COLUMN, SequenceDataset

DEFAULT_FOLDS: tuple[tuple[str, str, str], ...] = (
    ("fold1", "2020-01-01", "2021-01-01"),
    ("fold2", "2021-01-01", "2022-01-01"),
    ("fold3", "2022-01-01", "2023-01-01"),
)


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: date
    train_end_exclusive: date
    validation_start_inclusive: date
    validation_end_exclusive: date


def build_expanding_folds(config: AppConfig) -> list[Fold]:
    raw_folds = getattr(config, "expanding_folds", ())
    folds: list[Fold] = []
    if raw_folds:
        for raw in raw_folds:
            folds.append(
                Fold(
                    name=raw.name,
                    train_start=config.data.start_inclusive,
                    train_end_exclusive=raw.train_end_exclusive,
                    validation_start_inclusive=raw.train_end_exclusive,
                    validation_end_exclusive=raw.validation_end_exclusive,
                )
            )
    else:
        for name, train_end, val_end in DEFAULT_FOLDS:
            folds.append(
                Fold(
                    name=name,
                    train_start=config.data.start_inclusive,
                    train_end_exclusive=date.fromisoformat(train_end),
                    validation_start_inclusive=date.fromisoformat(train_end),
                    validation_end_exclusive=date.fromisoformat(val_end),
                )
            )
    ordered = sorted(folds, key=lambda f: f.train_end_exclusive)
    for fold in ordered:
        if not (
            fold.train_start
            < fold.train_end_exclusive
            <= fold.validation_end_exclusive
            <= config.splits.validation_end_exclusive
        ):
            raise ValueError(
                f"Fold {fold.name!r} must satisfy start < train_end <= "
                "validation_end <= final-test origin "
                f"({config.splits.validation_end_exclusive.isoformat()})."
            )
    return ordered


def final_test_origin(config: AppConfig) -> date:
    return config.splits.validation_end_exclusive


def split_fold_tabular(
    dataset: pd.DataFrame, fold: Fold, min_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = dataset.loc[
        (dataset["Date"] >= pd.Timestamp(fold.train_start))
        & (dataset["Date"] < pd.Timestamp(fold.train_end_exclusive))
    ].reset_index(drop=True)
    validation = dataset.loc[
        (dataset["Date"] >= pd.Timestamp(fold.validation_start_inclusive))
        & (dataset["Date"] < pd.Timestamp(fold.validation_end_exclusive))
    ].reset_index(drop=True)
    for name, part in (("train", train), ("validation", validation)):
        if len(part) < min_rows:
            raise ValueError(
                f"Insufficient rows for {fold.name}/{name}: found {len(part)}, "
                f"need at least {min_rows}."
            )
    return train, validation


def split_fold_sequence(
    dataset: SequenceDataset, fold: Fold, min_rows: int
) -> tuple[SequenceDataset, SequenceDataset]:
    out: dict[str, SequenceDataset] = {}
    for name, start, end in (
        ("train", fold.train_start, fold.train_end_exclusive),
        (
            "validation",
            fold.validation_start_inclusive,
            fold.validation_end_exclusive,
        ),
    ):
        mask = (dataset.dates >= pd.Timestamp(start)) & (
            dataset.dates < pd.Timestamp(end)
        )
        import numpy as np

        indices = np.flatnonzero(mask.to_numpy())
        if len(indices) < min_rows:
            raise ValueError(
                f"Insufficient sequence rows for {fold.name}/{name}: found "
                f"{len(indices)}, need at least {min_rows}."
            )
        out[name] = SequenceDataset(
            dataset.dates.iloc[indices].reset_index(drop=True),
            dataset.features[indices],
            dataset.targets[indices],
            dataset.channel_names,
            dataset.lookback,
        )
    return out["train"], out["validation"]


def aggregate_fold_metrics(rows: list[dict]) -> pd.DataFrame:
    """Mean validation metric per (model, candidate) across folds."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["model", "candidate"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
        )
        .sort_values(["mean_rmse", "mean_mae", "model", "candidate"])
        .reset_index(drop=True)
    )
    return grouped


def select_by_folds(aggregated: pd.DataFrame, metric: str = "mean_rmse") -> dict:
    if aggregated.empty:
        raise ValueError("No fold metrics available for selection.")
    ordered = aggregated.sort_values(
        [metric, "model", "candidate"]
    ).reset_index(drop=True)
    return dict(ordered.iloc[0])
