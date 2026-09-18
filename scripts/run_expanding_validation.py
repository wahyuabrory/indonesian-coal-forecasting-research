"""Expanding-window validation experiment.

For every (target, feature group):
  1. Score naive plus every XGBoost/GRU/Transformer candidate on 3 expanding
     development folds (validation years 2020, 2021, 2022).
  2. Freeze one candidate per family using mean fold RMSE inside that family.
  3. Refit each frozen model on all pre-2023 development data. Deep models
     retrain for the median fold best epoch (folds only, never the test).
  4. Score Naive + XGBoost + GRU + Transformer once on the untouched final
     test [2023-01-01, 2026-05-01).

Writes results/fold_metrics.csv, results/validation_winners.csv,
results/final_test_metrics.csv, results/feature_ablation.csv,
results/summary.csv, results/risk_summary.csv, results/run_manifest.json,
results/data_manifest.json.

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
    _metric_values,
    _train_gru_candidate,
    _train_gru_final,
    _train_transformer_candidate,
    _train_transformer_final,
    _xgboost_model,
    split_dataset,
    split_sequence_dataset,
)
from coal_forecasting.risk import calculate_historical_var  # noqa: E402
from coal_forecasting.validation import (  # noqa: E402
    ablation_delta,
    aggregate_fold_metrics,
    build_expanding_folds,
    median_best_epoch,
    select_per_family,
    split_fold_sequence,
    split_fold_tabular,
)

GROUPS = ("E0", "E1", "E2", "E3")
FAMILIES = ("zero_return_naive", "xgboost", "gru", "transformer")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, cwd=ROOT
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


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
        y_val = validation[TARGET_COLUMN].to_numpy(float)

        mae, rmse, da = _metric_values(y_val, np.zeros(len(validation)))
        rows.append(
            {"target": target, "feature_group": group, "fold": fold.name,
             "model": "zero_return_naive", "candidate": "",
             "rmse": rmse, "mae": mae, "directional_accuracy": da,
             "rows": len(validation), "best_epoch": None}
        )
        for candidate in config.xgboost_candidates:
            model = _xgboost_model(candidate, config.run.seed)
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_val), dtype=float)
            mae, rmse, da = _metric_values(y_val, pred)
            rows.append(
                {"target": target, "feature_group": group, "fold": fold.name,
                 "model": "xgboost", "candidate": candidate.name,
                 "rmse": rmse, "mae": mae, "directional_accuracy": da,
                 "rows": len(validation), "best_epoch": None}
            )
    return rows


def _deep_fold_metrics(
    snapshots, target: str, group: str, config: AppConfig,
    feature_config,
) -> list[dict]:
    rows: list[dict] = []
    folds = build_expanding_folds(config)
    gru_names = {c.name for c in config.gru_candidates}
    for candidate in (*config.gru_candidates, *config.transformer_candidates):
        is_gru = candidate.name in gru_names
        family = "gru" if is_gru else "transformer"
        seq = build_sequence_dataset(
            snapshots, target, feature_config, candidate.lookback
        )
        for fold in folds:
            train, validation = split_fold_sequence(
                seq, fold, config.splits.min_rows
            )
            from coal_forecasting.modeling import SequencePartitions

            parts = SequencePartitions(train, validation, validation)
            if is_gru:
                fitted = _train_gru_candidate(candidate, parts, config.run.seed)
            else:
                fitted = _train_transformer_candidate(
                    candidate, parts, config.run.seed
                )
            row = fitted.validation_row
            rows.append(
                {"target": target, "feature_group": group, "fold": fold.name,
                 "model": family, "candidate": candidate.name,
                 "rmse": float(row["rmse"]), "mae": float(row["mae"]),
                 "directional_accuracy": float(row["directional_accuracy"]),
                 "rows": int(row["rows"]), "best_epoch": fitted.best_epoch}
            )
    return rows


def _dev_test_sequence(
    seq: SequenceDataset, config: AppConfig,
) -> tuple[SequenceDataset, SequenceDataset]:
    origin = pd.Timestamp(config.splits.validation_end_exclusive)
    dev_mask = seq.dates < origin
    test_mask = seq.dates >= origin
    dev_idx = np.flatnonzero(dev_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    if len(dev_idx) < config.splits.min_rows or len(test_idx) < config.splits.min_rows:
        raise ValueError("Insufficient dev/test sequence rows for final refit.")
    dev = SequenceDataset(
        seq.dates.iloc[dev_idx].reset_index(drop=True),
        seq.features[dev_idx],
        seq.targets[dev_idx],
        seq.channel_names,
        seq.lookback,
    )
    test = SequenceDataset(
        seq.dates.iloc[test_idx].reset_index(drop=True),
        seq.features[test_idx],
        seq.targets[test_idx],
        seq.channel_names,
        seq.lookback,
    )
    return dev, test


def _final_test_rows(
    snapshots, target: str, group: str, config: AppConfig, feature_config,
    winners: dict[str, dict], fold_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Refit each frozen family on pre-2023 dev; score the test once."""
    dataset, cols = build_feature_dataset(snapshots, target, feature_config)
    parts = split_dataset(dataset, cols, config)
    dev = pd.concat([parts.train, parts.validation], ignore_index=True)
    assert (dev["Date"] < pd.Timestamp(config.splits.validation_end_exclusive)).all()
    assert (parts.test["Date"] >= pd.Timestamp(config.splits.validation_end_exclusive)).all()

    test_rows: list[dict] = []
    winner_rows: list[dict] = []

    def row(model, candidate, actual, pred, extra=None):
        mae, rmse, da = _metric_values(actual, pred)
        record = {"target": target, "feature_group": group, "model": model,
                  "candidate": candidate or "", "split": "test",
                  "rows": len(actual) if isinstance(actual, np.ndarray) else len(pred),
                  "mae": mae, "rmse": rmse, "directional_accuracy": da,
                  "selected_by": "mean_fold_rmse", "refit_on": "dev_pre2023",
                  "final_epochs": None}
        if extra:
            record.update(extra)
        return record

    y_test = parts.test[TARGET_COLUMN].to_numpy(float)
    naive_pred = np.zeros(len(parts.test))
    test_rows.append(row("zero_return_naive", "", y_test, naive_pred))

    xgb_name = winners["xgboost"]["candidate"]
    xgb_candidate = next(
        c for c in config.xgboost_candidates if c.name == xgb_name
    )
    scaler = StandardScaler().fit(dev[cols].to_numpy(float))
    xgb_model = _xgboost_model(xgb_candidate, config.run.seed)
    xgb_model.fit(scaler.transform(dev[cols].to_numpy(float)),
                  dev[TARGET_COLUMN].to_numpy(float))
    xgb_pred = np.asarray(
        xgb_model.predict(scaler.transform(parts.test[cols].to_numpy(float))),
        dtype=float,
    )
    test_rows.append(row("xgboost", xgb_name, y_test, xgb_pred))

    gru_names = {c.name for c in config.gru_candidates}
    all_deep = (*config.gru_candidates, *config.transformer_candidates)
    for family in ("gru", "transformer"):
        name = winners[family]["candidate"]
        candidate = next(c for c in all_deep if c.name == name)
        epochs = median_best_epoch(
            [int(r["best_epoch"]) for r in fold_rows
             if r["model"] == family and r["candidate"] == name]
        )
        seq = build_sequence_dataset(
            snapshots, target, feature_config, candidate.lookback
        )
        seq_dev, seq_test = _dev_test_sequence(seq, config)
        if family == "gru":
            fitted = _train_gru_final(candidate, seq_dev, epochs, config.run.seed)
        else:
            fitted = _train_transformer_final(
                candidate, seq_dev, epochs, config.run.seed
            )
        pred = fitted.predict(seq_test)
        test_rows.append(row(family, name, seq_test.targets.astype(float), pred,
                             {"final_epochs": epochs}))

    for family in FAMILIES:
        winner_rows.append(
            {"target": target, "feature_group": group, "model": family,
             "candidate": winners[family]["candidate"],
             "mean_fold_rmse": winners[family]["mean_metric"],
             "folds": winners[family]["folds"],
             "final_epochs": next(
                 (r["final_epochs"] for r in test_rows
                  if r["model"] == family), None)}
        )
    return test_rows, winner_rows


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
    winner_rows: list[dict] = []
    risk_rows: list[dict] = []
    for target in config.target_symbols:
        for group in GROUPS:
            feature_config = replace(config.features, group=group)
            print(f"{target} {group}: tabular folds...", flush=True)
            dataset, cols = build_feature_dataset(
                snapshots, target, feature_config
            )
            combo_folds = _tabular_fold_metrics(dataset, cols, config, target, group)
            print(f"{target} {group}: deep folds...", flush=True)
            combo_folds.extend(
                _deep_fold_metrics(snapshots, target, group, config, feature_config)
            )
            fold_rows.extend(combo_folds)

            winners = select_per_family(combo_folds)
            print(f"{target} {group}: winners " +
                  ", ".join(f"{m}/{w['candidate'] or '-'}={w['mean_metric']:.6f}"
                            for m, w in winners.items()), flush=True)
            test_rows, combo_winners = _final_test_rows(
                snapshots, target, group, config, feature_config,
                winners, combo_folds,
            )
            final_rows.extend(test_rows)
            winner_rows.extend(combo_winners)
            risk = calculate_historical_var(
                snapshots[target], target, group, "expanding",
                "per_family", None, config,
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

    winners_frame = pd.DataFrame(winner_rows)
    winners_frame.to_csv(results_dir / "validation_winners.csv", index=False)

    final_frame = pd.DataFrame(final_rows)
    final_frame.to_csv(results_dir / "final_test_metrics.csv", index=False)

    # Same-family ablation: validation delta and test delta vs E0.
    val_means = (
        folds_frame.merge(
            winners_frame[["target", "feature_group", "model", "candidate"]],
            on=["target", "feature_group", "model", "candidate"],
        ).query("model != 'zero_return_naive'"
        ).groupby(["target", "model", "feature_group", "candidate"],
                  as_index=False)["rmse"].mean()
    )
    test_means = final_frame[final_frame["model"] != "zero_return_naive"][
        ["target", "model", "feature_group", "candidate", "rmse"]
    ]
    val_abl = ablation_delta(val_means, "rmse").rename(
        columns={"rmse": "val_rmse", "delta_vs_E0": "val_delta_vs_E0"})
    test_abl = ablation_delta(test_means, "rmse").rename(
        columns={"rmse": "test_rmse", "delta_vs_E0": "test_delta_vs_E0"})
    ablation = val_abl.merge(
        test_abl[["target", "model", "feature_group", "candidate",
                  "test_rmse", "test_delta_vs_E0"]],
        on=["target", "model", "feature_group", "candidate"],
        how="outer",
    ).sort_values(["target", "model", "feature_group"]).reset_index(drop=True)
    ablation.to_csv(results_dir / "feature_ablation.csv", index=False)

    summary = final_frame.merge(
        winners_frame[["target", "feature_group", "model", "mean_fold_rmse"]],
        on=["target", "feature_group", "model"],
    )[["target", "feature_group", "model", "candidate",
       "mean_fold_rmse", "rmse"]].rename(columns={"rmse": "test_rmse"})
    summary.to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(results_dir / "risk_summary.csv", index=False)

    dirty = _git(["status", "--porcelain"])
    (results_dir / "run_manifest.json").write_text(json.dumps(
        {"run_at_utc": datetime.now(timezone.utc).isoformat(),
         "code_commit": _git(["rev-parse", "HEAD"]),
         "worktree_dirty": bool(dirty),
         "worktree_status": dirty[:2000],
         "config": str(args.config),
         "config_sha256": _sha256(ROOT / "configs/baseline.toml"),
         "lockfile_sha256": _sha256(ROOT / "requirements.lock"),
         "script_sha256": _sha256(ROOT / "scripts/run_expanding_validation.py"),
         "seed": config.run.seed,
         "selection": "per-family best candidate by mean fold RMSE; all 4 families refit on dev and scored once on final test",
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
