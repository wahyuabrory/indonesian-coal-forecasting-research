"""Experiment-design tests: per-family selection, refit scope, artifacts."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coal_forecasting.config import GRUCandidate, TransformerCandidate
from coal_forecasting.data import Snapshot, validate_snapshot
from coal_forecasting.features import SequenceDataset, build_sequence_dataset
from coal_forecasting.modeling import (
    _train_gru_final,
    _train_transformer_final,
)
from coal_forecasting.validation import (
    ablation_delta,
    median_best_epoch,
    select_per_family,
)


def _fold_rows() -> list[dict]:
    return [
        {"model": "xgboost", "candidate": "a", "fold": "fold1", "rmse": 0.020},
        {"model": "xgboost", "candidate": "a", "fold": "fold2", "rmse": 0.022},
        {"model": "xgboost", "candidate": "b", "fold": "fold1", "rmse": 0.019},
        {"model": "xgboost", "candidate": "b", "fold": "fold2", "rmse": 0.021},
        {"model": "gru", "candidate": "g1", "fold": "fold1", "rmse": 0.018},
        {"model": "gru", "candidate": "g1", "fold": "fold2", "rmse": 0.030},
        {"model": "gru", "candidate": "g2", "fold": "fold1", "rmse": 0.023},
        {"model": "gru", "candidate": "g2", "fold": "fold2", "rmse": 0.023},
    ]


def test_select_per_family_picks_within_family():
    winners = select_per_family(_fold_rows())
    assert winners["xgboost"]["candidate"] == "b"
    # g1 has the single best fold but the worse mean; g2 wins on mean.
    assert winners["gru"]["candidate"] == "g2"
    assert winners["gru"]["mean_metric"] == pytest.approx(0.023)


def test_select_per_family_empty_raises():
    with pytest.raises(ValueError, match="No fold metrics"):
        select_per_family([])


def test_ablation_delta_stays_within_family():
    frame = pd.DataFrame([
        {"target": "ADRO.JK", "model": "gru", "feature_group": "E0", "rmse": 0.020},
        {"target": "ADRO.JK", "model": "gru", "feature_group": "E1", "rmse": 0.019},
        {"target": "ADRO.JK", "model": "xgboost", "feature_group": "E0", "rmse": 0.025},
        {"target": "ADRO.JK", "model": "xgboost", "feature_group": "E1", "rmse": 0.024},
    ])
    out = ablation_delta(frame, "rmse")
    gru_delta = out[(out.model == "gru") & (out.feature_group == "E1")].iloc[0]
    xgb_delta = out[(out.model == "xgboost") & (out.feature_group == "E1")].iloc[0]
    assert gru_delta["delta_vs_E0"] == pytest.approx(-0.001)
    assert xgb_delta["delta_vs_E0"] == pytest.approx(-0.001)
    # E0 baseline rows carry zero delta, never a cross-family mix.
    assert (out[out.feature_group == "E0"]["delta_vs_E0"] == 0).all()


def test_median_best_epoch():
    assert median_best_epoch([10, 30, 20]) == 20
    assert median_best_epoch([10, 20]) == 15
    with pytest.raises(ValueError, match="at least one"):
        median_best_epoch([])
    with pytest.raises(ValueError, match="positive integers"):
        median_best_epoch([5, 0])


def _tiny_sequence(lookback: int = 5, n: int = 400) -> SequenceDataset:
    dates = pd.date_range("2016-01-04", periods=n, freq="B")
    rng = np.random.default_rng(77)
    snaps = {}
    for i, symbol in enumerate(("ADRO.JK", "PTBA.JK", "ITMG.JK", "IDR=X")):
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
        frame = validate_snapshot(
            pd.DataFrame({"Date": dates, "Close": prices, "Adj Close": prices}),
            symbol, date(2016, 1, 1), date(2026, 5, 1),
        )
        snaps[symbol] = Snapshot(symbol, frame, Path("/tmp/x.csv"), "t", "n")
    from dataclasses import replace
    from coal_forecasting.config import load_config

    config = load_config(Path("configs/baseline.toml"))
    return build_sequence_dataset(
        snaps, "ADRO.JK", replace(config.features, group="E0"), lookback
    )


def _split_dev(seq: SequenceDataset, n_dev: int = 300):
    dev = SequenceDataset(
        seq.dates.iloc[:n_dev].reset_index(drop=True),
        seq.features[:n_dev], seq.targets[:n_dev],
        seq.channel_names, seq.lookback,
    )
    test = SequenceDataset(
        seq.dates.iloc[n_dev:].reset_index(drop=True),
        seq.features[n_dev:], seq.targets[n_dev:],
        seq.channel_names, seq.lookback,
    )
    return dev, test


def _gru_candidate() -> GRUCandidate:
    return GRUCandidate(name="tiny", lookback=5, hidden_size=4,
                        learning_rate=0.01, weight_decay=0.0,
                        max_epochs=5, patience=2)


def _transformer_candidate() -> TransformerCandidate:
    return TransformerCandidate(name="tiny", lookback=5, d_model=8, nhead=2,
                                dim_feedforward=16, learning_rate=0.01,
                                weight_decay=0.0, max_epochs=5, patience=2)


def test_gru_final_refit_uses_dev_only():
    seq = _tiny_sequence()
    dev, test = _split_dev(seq)
    fitted = _train_gru_final(_gru_candidate(), dev, epochs=3, seed=42)
    assert fitted.best_epoch == 3
    dev_mean = dev.features.reshape(-1, dev.features.shape[-1]).mean(axis=0)
    assert np.allclose(fitted.feature_scaler.mean_, dev_mean)
    full_mean = seq.features.reshape(-1, seq.features.shape[-1]).mean(axis=0)
    assert not np.allclose(fitted.feature_scaler.mean_, full_mean)
    pred = fitted.predict(test)
    assert pred.shape == (len(test.dates),) and np.isfinite(pred).all()
    with pytest.raises(Exception, match="at least 1 epoch"):
        _train_gru_final(_gru_candidate(), dev, epochs=0, seed=42)


def test_transformer_final_refit_uses_dev_only():
    seq = _tiny_sequence()
    dev, test = _split_dev(seq)
    fitted = _train_transformer_final(
        _transformer_candidate(), dev, epochs=2, seed=42
    )
    assert fitted.best_epoch == 2
    dev_mean = dev.features.reshape(-1, dev.features.shape[-1]).mean(axis=0)
    assert np.allclose(fitted.feature_scaler.mean_, dev_mean)
    assert np.isfinite(fitted.predict(test)).all()


def _results(path: str) -> pd.DataFrame:
    full = Path("results") / path
    assert full.exists(), f"Run scripts/run_expanding_validation.py first: {path} missing."
    return pd.read_csv(full)


def test_committed_final_test_has_four_families():
    final = _results("final_test_metrics.csv")
    expected_models = {"zero_return_naive", "xgboost", "gru", "transformer"}
    assert set(final["model"].unique()) == expected_models
    for target in ("ADRO.JK", "PTBA.JK", "ITMG.JK"):
        for group in ("E0", "E1", "E2", "E3"):
            sub = final[(final.target == target) & (final.feature_group == group)]
            assert set(sub["model"]) == expected_models, (target, group)


def test_committed_ablation_is_same_family():
    ablation = _results("feature_ablation.csv")
    final = _results("final_test_metrics.csv")
    for _, row in ablation.iterrows():
        sub = final[(final.target == row["target"])
                    & (final.model == row["model"])]
        e0 = sub[sub.feature_group == "E0"]["rmse"].iloc[0]
        group = sub[sub.feature_group == row["feature_group"]]["rmse"].iloc[0]
        assert row["test_delta_vs_E0"] == pytest.approx(group - e0)


def test_committed_validation_winners_match_folds():
    winners = _results("validation_winners.csv")
    folds = _results("fold_metrics.csv")
    for _, row in winners.iterrows():
        sub = folds[(folds.target == row["target"])
                    & (folds.feature_group == row["feature_group"])
                    & (folds.model == row["model"])
                    & (folds.candidate == row["candidate"])]
        assert len(sub) == row["folds"] == 3
        assert row["mean_fold_rmse"] == pytest.approx(sub["rmse"].mean())


def test_offline_smoke_passes():
    sys_path_insert = str(Path("scripts").resolve())
    import sys

    if sys_path_insert not in sys.path:
        sys.path.insert(0, str(Path(".").resolve() / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "smoke_offline", "scripts/smoke_offline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
