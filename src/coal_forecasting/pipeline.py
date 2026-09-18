from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import AppConfig, FEATURE_GROUPS
from .data import Snapshot, load_snapshots, snapshot_slug
from .features import build_feature_dataset, build_sequence_dataset
from .modeling import EvaluationResult, evaluate, split_dataset, split_sequence_dataset
from .risk import calculate_historical_var


ModelMode = Literal["baseline", "xgboost", "both", "gru", "transformer", "all"]


def _split_metadata(config: AppConfig, result: EvaluationResult) -> dict[str, dict[str, object]]:
    configured = {
        "train": (config.data.start_inclusive, config.splits.train_end_exclusive),
        "validation": (
            config.splits.train_end_exclusive,
            config.splits.validation_end_exclusive,
        ),
        "test": (config.splits.validation_end_exclusive, config.splits.test_end_exclusive),
    }
    return {
        name: {
            "configured_start_inclusive": start.isoformat(),
            "configured_end_exclusive": end.isoformat(),
            **result.split_metadata[name],
        }
        for name, (start, end) in configured.items()
    }


def _write_result(
    output_dir: Path,
    target_symbol: str,
    feature_group: str,
    feature_columns: list[str],
    result: EvaluationResult,
    config: AppConfig,
    snapshots: dict[str, Snapshot],
    model_mode: ModelMode,
) -> Path:
    risk_payload = calculate_historical_var(
        snapshots[target_symbol],
        target_symbol,
        feature_group,
        model_mode,
        result.selected_model,
        result.selected_candidate,
        config,
    )
    target_dir = output_dir / snapshot_slug(target_symbol) / feature_group
    target_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.metrics).to_csv(target_dir / "metrics.csv", index=False)

    run_at = datetime.now(timezone.utc).isoformat()
    snapshot_metadata = {symbol: snapshot.metadata() for symbol, snapshot in snapshots.items()}
    metadata = {
        "run_at_utc": run_at,
        "source": "Yahoo Finance daily snapshots loaded from the configured cache or downloaded with yfinance.",
        "retrieval_times_utc": {
            symbol: snapshot.retrieved_at_utc for symbol, snapshot in snapshots.items()
        },
        "snapshot_metadata": snapshot_metadata,
        "target": target_symbol,
        "feature_group": feature_group,
        "feature_columns": feature_columns,
        "requested_start_inclusive": config.data.start_inclusive.isoformat(),
        "requested_end_exclusive": config.data.end_exclusive.isoformat(),
        "split_dates": _split_metadata(config, result),
        "seed": config.run.seed,
        "model_mode": model_mode,
        "selected_model": result.selected_model,
        "selected_candidate": result.selected_candidate,
        "selected_parameters": result.selected_parameters,
        "selected_framework": result.selected_framework,
        "framework_version": result.selected_framework_version,
        "best_epoch": result.selected_best_epoch,
        "parameter_count": result.selected_parameter_count,
        "model_metadata": result.model_metadata,
        "candidate_parameters": result.candidate_parameters,
        "selection_metric": config.selection_metric,
        "test_used_for_selection": False,
        "scaling_rule": result.scaling_rule,
        "gru_scaling_rule": (
            "GRU feature StandardScaler fit on training sequence values only and target "
            "StandardScaler fit on training labels only; validation and test are transform-only."
            if model_mode in ("gru", "all")
            else None
        ),
        "transformer_scaling_rule": (
            "Transformer feature StandardScaler fit on training sequence values only and target "
            "StandardScaler fit on training labels only; validation and test are transform-only."
            if model_mode in ("transformer", "all")
            else None
        ),
        "preprocessing": result.scaling_rule,
        "forecast_timestamp_assumption": config.research.forecast_timestamp_assumption,
        "adro_aadi_structural_break_caveat": config.research.adro_aadi_structural_break_caveat,
        "coal_feature_policy": config.research.coal_feature_policy,
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_payload = {
        "target": target_symbol,
        "feature_group": feature_group,
        "metrics": result.metrics,
    }
    (target_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (target_dir / "risk.json").write_text(json.dumps(risk_payload, indent=2), encoding="utf-8")
    return target_dir


def run(
    config: AppConfig,
    targets: tuple[str, ...],
    feature_group: str | None = None,
    model_mode: ModelMode = "both",
    refresh: bool = False,
    output_dir: Path | None = None,
) -> list[Path]:
    group = feature_group or config.features.group
    if group not in FEATURE_GROUPS:
        raise ValueError(f"feature group must be one of {FEATURE_GROUPS}.")
    unknown = [target for target in targets if target not in config.target_symbols]
    if unknown:
        raise ValueError(f"Unknown target symbol(s): {', '.join(unknown)}")
    if not targets:
        raise ValueError("At least one target symbol is required.")

    snapshots = load_snapshots(config, refresh=refresh)
    feature_config = replace(config.features, group=group)
    base_output_dir = output_dir or config.run.output_dir
    written: list[Path] = []
    for target in targets:
        dataset, feature_columns = build_feature_dataset(snapshots, target, feature_config)
        partitions = split_dataset(dataset, feature_columns, config)
        sequence_partitions = None
        if model_mode in ("gru", "transformer", "all"):
            sequence_partitions = {}
            sequence_candidates = (
                config.gru_candidates
                if model_mode == "gru"
                else config.transformer_candidates
                if model_mode == "transformer"
                else (*config.gru_candidates, *config.transformer_candidates)
            )
            for candidate in sequence_candidates:
                sequence_dataset = build_sequence_dataset(
                    snapshots,
                    target,
                    feature_config,
                    candidate.lookback,
                )
                sequence_partitions[candidate.name] = split_sequence_dataset(
                    sequence_dataset,
                    config,
                )
        result = evaluate(
            partitions,
            feature_columns,
            config,
            model_mode,
            sequence_partitions=sequence_partitions,
        )
        written.append(
            _write_result(
                base_output_dir,
                target,
                group,
                feature_columns,
                result,
                config,
                snapshots,
                model_mode,
            )
        )
    return written
