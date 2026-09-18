from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import Snapshot


class RiskCalculationError(ValueError):
    """Raised when a historical VaR calculation fails its data boundary."""


def _validated_returns(snapshot: Snapshot, target_symbol: str) -> tuple[pd.Series, np.ndarray]:
    if snapshot.symbol != target_symbol:
        raise RiskCalculationError(
            f"Risk snapshot symbol {snapshot.symbol!r} does not match target {target_symbol!r}."
        )
    missing = [column for column in ("Date", "Adj Close") if column not in snapshot.frame]
    if missing:
        raise RiskCalculationError(
            f"{target_symbol} risk input is missing required column(s): {', '.join(missing)}."
        )
    if snapshot.frame.empty:
        raise RiskCalculationError(f"{target_symbol} risk input has no rows.")

    frame = snapshot.frame.loc[:, ["Date", "Adj Close"]].copy()
    dates = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise RiskCalculationError(f"{target_symbol} risk input contains invalid dates.")
    dates = dates.dt.tz_localize(None).dt.normalize()
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise RiskCalculationError(
            f"{target_symbol} risk input dates must be unique and chronological."
        )

    prices = pd.to_numeric(frame["Adj Close"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise RiskCalculationError(
            f"{target_symbol} adjusted-close risk input must contain finite positive prices."
        )
    with np.errstate(over="raise", invalid="raise"):
        try:
            returns = np.diff(np.log(prices))
        except FloatingPointError as exc:
            raise RiskCalculationError(
                f"{target_symbol} adjusted-close log returns are not finite."
            ) from exc
    if not np.isfinite(returns).all():
        raise RiskCalculationError(
            f"{target_symbol} adjusted-close log returns are not finite."
        )
    return dates.iloc[1:].reset_index(drop=True), returns


def calculate_historical_var(
    snapshot: Snapshot,
    target_symbol: str,
    feature_group: str,
    model_mode: str,
    selected_model: str,
    selected_candidate: str | None,
    config: AppConfig,
) -> dict[str, Any]:
    data_start = pd.Timestamp(config.data.start_inclusive)
    data_end = pd.Timestamp(config.data.end_exclusive)
    train_end = pd.Timestamp(config.splits.train_end_exclusive)
    validation_end = pd.Timestamp(config.splits.validation_end_exclusive)
    test_end = pd.Timestamp(config.splits.test_end_exclusive)
    if not (
        data_start < train_end < validation_end < test_end <= data_end
    ):
        raise RiskCalculationError(
            "Risk calculation requires chronological data and split boundaries with "
            "test_end_exclusive within data.end_exclusive."
        )

    confidence = float(config.risk.confidence)
    notional = float(config.risk.portfolio_value_idr)
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise RiskCalculationError("risk.confidence must be finite and strictly between 0 and 1.")
    if not math.isfinite(notional) or notional <= 0:
        raise RiskCalculationError("risk.portfolio_value_idr must be finite and positive.")

    return_dates, returns = _validated_returns(snapshot, target_symbol)
    calibration_mask = (return_dates >= data_start) & (return_dates < validation_end)
    final_test_mask = (return_dates >= validation_end) & (return_dates < test_end)
    calibration = returns[calibration_mask.to_numpy()]
    final_test = returns[final_test_mask.to_numpy()]
    minimum_rows = config.splits.min_rows
    if len(calibration) < minimum_rows:
        raise RiskCalculationError(
            f"Insufficient finite calibration returns for {target_symbol}: found {len(calibration)}, "
            f"need at least {minimum_rows} before {config.splits.validation_end_exclusive.isoformat()}."
        )
    if len(final_test) < minimum_rows:
        raise RiskCalculationError(
            f"Insufficient finite final-test returns for {target_symbol}: found {len(final_test)}, "
            f"need at least {minimum_rows} from {config.splits.validation_end_exclusive.isoformat()}."
        )

    alpha = 1.0 - confidence
    quantile = float(np.quantile(calibration, alpha, method="linear"))
    var_log_return_magnitude = max(0.0, -quantile)
    simple_loss_fraction = max(0.0, 1.0 - math.exp(quantile)) if quantile < 0 else 0.0
    estimated_loss = notional * simple_loss_fraction
    breach_count = int(np.count_nonzero(final_test < quantile))

    calibration_dates = return_dates[calibration_mask]
    final_test_dates = return_dates[final_test_mask]
    return {
        "analysis": "secondary_descriptive_analysis",
        "method": "one_day_unconditional_historical_simulation_var",
        "target": target_symbol,
        "feature_group": feature_group,
        "model_mode": model_mode,
        "selected_model": selected_model,
        "selected_candidate": selected_candidate,
        "unconditional": True,
        "conditioned_on_forecasts": False,
        "risk_used_for_model_selection": False,
        "snapshot_symbol": snapshot.symbol,
        "snapshot_path": str(snapshot.path),
        "adjusted_close_field": "Adj Close",
        "return_definition": "log(Adj Close[d]) - log(Adj Close[previous observed target date])",
        "return_date_semantics": "The return date is the ending adjusted-close observation date.",
        "quantile_method": "numpy.quantile(method='linear')",
        "calibration_return_date_start_inclusive": config.data.start_inclusive.isoformat(),
        "calibration_return_date_end_exclusive": config.splits.validation_end_exclusive.isoformat(),
        "final_test_return_date_start_inclusive": config.splits.validation_end_exclusive.isoformat(),
        "final_test_return_date_end_exclusive": config.splits.test_end_exclusive.isoformat(),
        "calibration_rows": int(len(calibration)),
        "calibration_start_date": calibration_dates.min().date().isoformat(),
        "calibration_end_date": calibration_dates.max().date().isoformat(),
        "final_test_rows": int(len(final_test)),
        "final_test_start_date": final_test_dates.min().date().isoformat(),
        "final_test_end_date": final_test_dates.max().date().isoformat(),
        "confidence": confidence,
        "alpha": alpha,
        "log_return_quantile": quantile,
        "var_log_return_magnitude": var_log_return_magnitude,
        "simple_loss_fraction": simple_loss_fraction,
        "portfolio_value_idr": notional,
        "notional_idr": notional,
        "estimated_loss_idr": estimated_loss,
        "final_test_breach_count": breach_count,
        "final_test_breach_rate": breach_count / len(final_test),
    }
