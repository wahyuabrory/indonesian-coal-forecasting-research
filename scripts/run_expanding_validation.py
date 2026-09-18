"""Expanding-window validation experiment.

For every (target, feature group):
  - evaluate naive + all XGBoost/GRU/Transformer candidates on 3 expanding
    development folds (validation years 2020, 2021, 2022),
  - aggregate by mean fold RMSE and select the winner,
  - refit the winner on the single-split development data and score the
    frozen final test [2023-01-01, 2026-05-01) exactly once.

Writes results/fold_metrics.csv, results/final_test_metrics.csv,
results/feature_ablation.csv, results/summary.csv, results/risk_summary.csv,
results/run_manifest.json, results/data_manifest.json.

Usage: python3 scripts/run_expanding_validation.py [--config PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coal_forecasting.config import AppConfig, load_config  # noqa: E402
from coal_forecasting.data import load_snapshots  # noqa: E402
from coal_forecasting.features import (  # noqa: E402
    TARGET_COLUMN,
    SequenceDataset,
    build_feature_dataset,
    build_sequence_dataset,
)
from coal_forecasting.modeling import (  # noqa: E402
    SequencePartitions,
    _metric_row,
    _metric_values,
    _sequence_metric_row,
    _train_gru_candidate,
    _train_transformer_candidate,
    _xgboost_model,
    split_dataset,
    split_sequence_dataset,
)
from coal_forecasting.risk import calculate_historical_var  # noqa: E402
from coal_forecasting.validation import (  # noqa: E402
    aggregate_fold_metrics,
    build_expanding_folds,
    select_by_folds,
    split_fold_sequence,
    split_fold_tabular,
)

GROUPS = ("E0", "E1", "E2", "E3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tabular_fold_metrics(
    dataset: pd.DataFrame,
    cols: list[str],
    config: AppConfig,
    target: str,
    group: str,
) -> list[dict]:
    rows: list[dict] = []
    for fold in build_expanding_folds(config):
        train, validation = split_fold_tabular(
            dataset, fold, config.splits.min_rows
        )
        scaler = StandardScaler().fit(train[cols].to_numpy(float))
        x_train = scaler.transform(train[cols].to_numpy(float))
        x_val = scaler.transform(validation[cols].to_numpy(float))
        y_train = train[TARGET_COLUMN].to_numpy(float)

        mae, rmse, da = _metric_values(
            validation[TARGET_COLUMN].to_numpy(float),
            np.zeros(len(validation)),
        )
        rows.append(
            {"target": target, "feature_group": group, "fold": fold.name,
             "model": "zero_return_naive", "candidate": "",
             "rmse": rmse, "mae": mae, "directional_accuracy": da,
             "rows": len(validation)}
        )
        for candidate in config.xgboost_candidates:
            model = _xgboost_model(candidate, config.run.seed)
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_val), dtype=float)
            mae, rmse, da = _metric_values(
                validation[TARGET_COLUMN].to_numpy(float), pred
            )
            rows.append(
                {"target": target, "feature_group": group, "fold": fold.name,
                 "model": "xgboost", "candidate": candidate.name,
                 "rmse": rmse, "mae": mae, "directional_accuracy": da,
                 "rows": len(validation)}
            )
    return rows


def _fold_seq_partitions(
    seq: SequenceDataset, config: AppConfig, fold_name: str,
    start, end, val_start, val_end,
) -> SequencePartitions:
    from coal_forecasting.validation import Fold

    fold = Fold(fold_name, start, end, val_start, val_end)
    train, validation = split_fold_sequence(
        seq, fold, config.splits.min_rows
    )
    return SequencePartitions(train, validation, validation)


def _deep_fold_metrics(
    snapshots, target: str, group: str, config: AppConfig,
    feature_config,
) -> list[dict]:
    rows: list[dict] = []
    folds = build_expanding_folds(config)
    for candidate in (*config.gru_candidates, *config.transformer_candidates):
        is_gru = candidate.name in {c.name for c in config.gru_candidates}
        seq = build_sequence_dataset(
            snapshots, target, feature_config, candidate.lookback
        )
        for fold in folds:
            train, validation = split_fold_sequence(
                seq, fold, config.splits.min_rows
            )
            parts = SequencePartitions(train, validation, validation)
            if is_gru:
                fitted = _train_gru_candidate(candidate, parts, config.run.seed)
                family = "gru"
            else:
                fitted = _train_transformer_candidate(
                    candidate, parts, config.run.seed
                )
                family = "transformer"
            row = fitted.validation_row
            rows.append(
                {"target": target, "feature_group": group, "fold": fold.name,
                 "model": family, "candidate": candidate.name,
                 "rmse": float(row["rmse"]), "mae": float(row["mae"]),
                 "directional_accuracy": float(row["directional_accuracy"]),
                 "rows": int(row["rows"])}
            )
    return rows


def _final_test_metrics(
    snapshots, target: str, group: str, config: AppConfig, feature_config,
    winner_model: str, winner_candidate: str,
) -> list[dict]:
    """Refit the fold-selected winner on single-split dev data; score test once."""
    dataset, cols = build_feature_dataset(snapshots, target, feature_config)
    parts = split_dataset(dataset, cols, config)
    scaler = StandardScaler().fit(parts.train[cols].to_numpy(float))
    out: list[dict] = []

    def row(model, candidate, actual_frame, pred, split):
        mae, rmse, da = _metric_values(
            actual_frame[TARGET_COLUMN].to_numpy(float), pred
        )
        return {"target": target, "feature_group": group, "model": model,
                "candidate": candidate or "", "split": split,
                "rows": len(actual_frame), "mae": mae, "rmse": rmse,
                "directional_accuracy": da,
                "selected_by": "mean_fold_rmse"}

    if winner_model == "zero_return_naive":
        out.append(row(winner_model, "", parts.test,
                       np.zeros(len(parts.test)), "test"))
    elif winner_model == "xgboost":
        candidate = next(
            c for c in config.xgboost_candidates if c.name == winner_candidate
        )
        model = _xgboost_model(candidate, config.run.seed)
        model.fit(scaler.transform(parts.train[cols].to_numpy(float)),
                  parts.train[TARGET_COLUMN].to_numpy(float))
        pred = np.asarray(
            model.predict(scaler.transform(parts.test[cols].to_numpy(float))),
            dtype=float,
        )
        out.append(row("xgboost", candidate.name, parts.test, pred, "test"))
    else:
        all_candidates = (*config.gru_candidates, *config.transformer_candidates)
        candidate = next(c for c in all_candidates if c.name == winner_candidate)
        seq = build_sequence_dataset(
            snapshots, target, feature_config, candidate.lookback
        )
        seq_parts = split_sequence_dataset(seq, config)
        if winner_model == "gru":
            fitted = _train_gru_candidate(candidate, seq_parts, config.run.seed)
        else:
            fitted = _train_transformer_candidate(
                candidate, seq_parts, config.run.seed
            )
        pred = fitted.predict(seq_parts.test)
        actual = seq_parts.test.targets.astype(float)
        mae, rmse, da = _metric_values(actual, pred)
        out.append({"target": target, "feature_group": group,
                    "model": winner_model, "candidate": candidate.name,
                    "split": "test", "rows": len(seq_parts.test.dates),
                    "mae": mae, "rmse": rmse, "directional_accuracy": da,
                    "selected_by": "mean_fold_rmse"})
    # Always report the naive test reference alongside the winner.
    if winner_model != "zero_return_naive":
        out.append(row("zero_return_naive", "", parts.test,
                       np.zeros(len(parts.test)), "test"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    snapshots = load_snapshots(config)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    try:
        import torch
        import xgboost

        torch_version = torch.__version__
        xgboost_version = xgboost.__version__
    except Exception:
        torch_version = xgboost_version = "unavailable"

    fold_rows: list[dict] = []
    final_rows: list[dict] = []
    risk_rows: list[dict] = []
    for target in config.target_symbols:
        for group in GROUPS:
            feature_config = replace(config.features, group=group)
            print(f"{target} {group}: tabular folds...", flush=True)
            dataset, cols = build_feature_dataset(
                snapshots, target, feature_config
            )
            fold_rows.extend(
                _tabular_fold_metrics(dataset, cols, config, target, group)
            )
            print(f"{target} {group}: deep folds...", flush=True)
            fold_rows.extend(
                _deep_fold_metrics(
                    snapshots, target, group, config, feature_config
                )
            )

            sub = [r for r in fold_rows
                   if r["target"] == target and r["feature_group"] == group]
            frame = pd.DataFrame(
                [{"model": r["model"], "candidate": r["candidate"],
                  "fold": r["fold"], "rmse": r["rmse"], "mae": r["mae"],
                  "directional_accuracy": r["directional_accuracy"]}
                 for r in sub]
            )
            agg = aggregate_fold_metrics(
                frame.to_dict(orient="records")
            )
            winner = select_by_folds(agg, "mean_rmse")
            print(f"{target} {group}: winner {winner['model']}/"
                  f"{winner['candidate']} mean_rmse={winner['mean_rmse']:.6f}",
                  flush=True)
            final_rows.extend(
                _final_test_metrics(
                    snapshots, target, group, config, feature_config,
                    winner["model"], winner["candidate"],
                )
            )
            risk = calculate_historical_var(
                snapshots[target], target, group, "expanding",
                winner["model"], winner["candidate"] or None, config,
            )
            risk_rows.append(
                {"target": target, "feature_group": group,
                 "confidence": risk["confidence"],
                 "var_log_return_magnitude": risk["var_log_return_magnitude"],
                 "simple_loss_fraction": risk["simple_loss_fraction"],
                 "estimated_loss_idr": risk["estimated_loss_idr"],
                 "final_test_breach_count": risk["final_test_breach_count"],
                 "final_test_breach_rate": risk["final_test_breach_rate"]}
            )

    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(results_dir / "fold_metrics.csv", index=False)

    final_frame = pd.DataFrame(final_rows)
    final_frame.to_csv(results_dir / "final_test_metrics.csv", index=False)

    # Ablation: fold-selected model test RMSE per group vs E0, per target.
    ablation_rows = []
    for target in config.target_symbols:
        base = final_frame[(final_frame.target == target)
                           & (final_frame.feature_group == "E0")]
        base = base[base["model"] != "zero_return_naive"]
        if base.empty:
            continue
        base_rmse = float(base.iloc[0]["rmse"])
        for group in GROUPS:
            row = final_frame[(final_frame.target == target)
                              & (final_frame.feature_group == group)]
            row = row[row["model"] != "zero_return_naive"]
            if row.empty:
                continue
            r = row.iloc[0]
            ablation_rows.append(
                {"target": target, "feature_group": group,
                 "model": r["model"], "candidate": r["candidate"],
                 "test_rmse": float(r["rmse"]),
                 "delta_vs_E0": float(r["rmse"]) - base_rmse}
            )
    pd.DataFrame(ablation_rows).to_csv(
        results_dir / "feature_ablation.csv", index=False
    )

    # Summary: fold winner + test RMSE per target/group.
    summary_rows = []
    for target in config.target_symbols:
        for group in GROUPS:
            sub = folds_frame[(folds_frame.target == target)
                              & (folds_frame.feature_group == group)]
            agg = aggregate_fold_metrics(
                sub[["model", "candidate", "fold", "rmse", "mae",
                     "directional_accuracy"]].to_dict(orient="records")
            )
            winner = select_by_folds(agg, "mean_rmse")
            test = final_frame[(final_frame.target == target)
                               & (final_frame.feature_group == group)
                               & (final_frame.model == winner["model"])]
            summary_rows.append(
                {"target": target, "feature_group": group,
                 "fold_selected_model": winner["model"],
                 "fold_selected_candidate": winner["candidate"],
                 "mean_fold_rmse": winner["mean_rmse"],
                 "test_rmse": float(test.iloc[0]["rmse"]) if len(test) else None}
            )
    pd.DataFrame(summary_rows).to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(results_dir / "risk_summary.csv", index=False)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    (results_dir / "run_manifest.json").write_text(json.dumps(
        {"run_at_utc": datetime.now(timezone.utc).isoformat(),
         "git_commit": commit,
         "config": str(args.config),
         "seed": config.run.seed,
         "selection": "mean_fold_rmse over fold1/fold2/fold3; final test evaluated once",
         "final_test": [config.splits.validation_end_exclusive.isoformat(),
                        config.splits.test_end_exclusive.isoformat()],
         "python": sys.version.split()[0],
         "versions": {"torch": torch_version, "xgboost": xgboost_version,
                      "pandas": pd.__version__, "numpy": np.__version__}},
        indent=2,
    ))
    (results_dir / "data_manifest.json").write_text(json.dumps(
        {"symbols": {
            symbol: {
                "path": str(snap.path),
                "source": snap.source,
                "retrieved_at_utc": snap.retrieved_at_utc,
                "rows": len(snap.frame),
                "sha256": _sha256(snap.path),
            } for symbol, snap in snapshots.items()}},
        indent=2,
    ))
    print("done:", sorted(p.name for p in results_dir.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
