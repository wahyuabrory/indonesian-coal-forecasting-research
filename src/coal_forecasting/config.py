from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
from typing import Any
import tomllib


class ConfigurationError(ValueError):
    """Raised when the run configuration cannot define a safe experiment."""


REQUIRED_SYMBOLS = ("ADRO.JK", "PTBA.JK", "ITMG.JK", "IDR=X")
FEATURE_GROUPS = ("E0", "E1", "E2", "E3")
DEFAULT_GRU_CANDIDATES = (
    {
        "name": "lookback5_hidden16",
        "lookback": 5,
        "hidden_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 100,
        "patience": 12,
    },
    {
        "name": "lookback10_hidden32",
        "lookback": 10,
        "hidden_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 100,
        "patience": 12,
    },
)
DEFAULT_TRANSFORMER_CANDIDATES = (
    {
        "name": "lookback5_d16_heads2_ff32",
        "lookback": 5,
        "d_model": 16,
        "nhead": 2,
        "dim_feedforward": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 100,
        "patience": 12,
    },
    {
        "name": "lookback10_d32_heads4_ff64",
        "lookback": 10,
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 100,
        "patience": 12,
    },
)


def _date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ConfigurationError(f"{field} must be an ISO date string.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(f"{field} is not a valid ISO date: {value!r}.") from exc


def _positive_int(value: Any, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{field} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{field} must be <= {maximum}.")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{field} must be a finite number.")
    return result


@dataclass(frozen=True)
class ExpandingFoldConfig:
    name: str
    train_end_exclusive: date
    validation_end_exclusive: date


@dataclass(frozen=True)
class ExternalConfig:
    usd_idr_symbol: str
    usd_idr_availability_lag: int


@dataclass(frozen=True)
class DataConfig:
    symbols: tuple[str, ...]
    start_inclusive: date
    end_exclusive: date
    interval: str
    cache_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    train_end_exclusive: date
    validation_end_exclusive: date
    test_end_exclusive: date
    min_rows: int


@dataclass(frozen=True)
class FeatureConfig:
    group: str
    return_lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    usd_idr_availability_lag: int
    peer_availability_lag: int
    peers: tuple[str, ...]


@dataclass(frozen=True)
class XGBoostCandidate:
    name: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    reg_lambda: float

    def parameters(self) -> dict[str, int | float]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
        }


@dataclass(frozen=True)
class GRUCandidate:
    name: str
    lookback: int
    hidden_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int

    def parameters(self) -> dict[str, int | float]:
        return {
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "num_layers": 1,
        }


@dataclass(frozen=True)
class TransformerCandidate:
    name: str
    lookback: int
    d_model: int
    nhead: int
    dim_feedforward: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int

    def parameters(self) -> dict[str, int | float | str]:
        return {
            "lookback": self.lookback,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "num_layers": 1,
            "positional_encoding": "learned_embedding",
        }


@dataclass(frozen=True)
class RiskConfig:
    confidence: float
    portfolio_value_idr: float


@dataclass(frozen=True)
class ResearchConfig:
    forecast_timestamp_assumption: str
    adro_aadi_structural_break_caveat: str
    coal_feature_policy: str


@dataclass(frozen=True)
class RunConfig:
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    splits: SplitConfig
    features: FeatureConfig
    risk: RiskConfig
    research: ResearchConfig
    run: RunConfig
    xgboost_candidates: tuple[XGBoostCandidate, ...]
    gru_candidates: tuple[GRUCandidate, ...]
    transformer_candidates: tuple[TransformerCandidate, ...]
    selection_metric: str
    expanding_folds: tuple[ExpandingFoldConfig, ...] = ()
    external: ExternalConfig | None = None

    @property
    def target_symbols(self) -> tuple[str, ...]:
        return tuple(symbol for symbol in self.data.symbols if symbol != "IDR=X")


def _check_dates(data: DataConfig, splits: SplitConfig) -> None:
    ends = (
        splits.train_end_exclusive,
        splits.validation_end_exclusive,
        splits.test_end_exclusive,
    )
    if not data.start_inclusive < splits.train_end_exclusive:
        raise ConfigurationError("start_inclusive must be before train_end_exclusive.")
    if not splits.train_end_exclusive < splits.validation_end_exclusive < splits.test_end_exclusive:
        raise ConfigurationError("Split end dates must be strictly chronological.")
    if splits.test_end_exclusive > data.end_exclusive:
        raise ConfigurationError("test_end_exclusive cannot be after data.end_exclusive.")
    if any(end <= data.start_inclusive for end in ends):
        raise ConfigurationError("Every split must contain dates after start_inclusive.")


def _parse_candidate(raw: Any, index: int) -> XGBoostCandidate:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"xgboost.candidates[{index}] must be a table.")
    prefix = f"xgboost.candidates[{index}]"
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError(f"{prefix}.name must be a non-empty string.")
    n_estimators = _positive_int(raw.get("n_estimators"), f"{prefix}.n_estimators", 500)
    max_depth = _positive_int(raw.get("max_depth"), f"{prefix}.max_depth", 8)
    min_child_weight = _positive_int(raw.get("min_child_weight"), f"{prefix}.min_child_weight", 100)
    learning_rate = raw.get("learning_rate")
    subsample = raw.get("subsample")
    colsample = raw.get("colsample_bytree")
    reg_lambda = raw.get("reg_lambda")
    floats = {
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "reg_lambda": reg_lambda,
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in floats.values()):
        raise ConfigurationError(f"{prefix} numeric parameters must be numbers.")
    if not 0 < float(learning_rate) <= 1:
        raise ConfigurationError(f"{prefix}.learning_rate must be in (0, 1].")
    if not 0 < float(subsample) <= 1 or not 0 < float(colsample) <= 1:
        raise ConfigurationError(f"{prefix} sampling rates must be in (0, 1].")
    if float(reg_lambda) < 0:
        raise ConfigurationError(f"{prefix}.reg_lambda cannot be negative.")
    return XGBoostCandidate(
        name=name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample),
        min_child_weight=min_child_weight,
        reg_lambda=float(reg_lambda),
    )


def _parse_gru_candidate(raw: Any, index: int) -> GRUCandidate:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"gru.candidates[{index}] must be a table.")
    prefix = f"gru.candidates[{index}]"
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError(f"{prefix}.name must be a non-empty string.")
    lookback = _positive_int(raw.get("lookback"), f"{prefix}.lookback", 20)
    hidden_size = _positive_int(raw.get("hidden_size"), f"{prefix}.hidden_size", 32)
    max_epochs = _positive_int(raw.get("max_epochs"), f"{prefix}.max_epochs", 100)
    patience = _positive_int(raw.get("patience"), f"{prefix}.patience", max_epochs)
    learning_rate = raw.get("learning_rate")
    weight_decay = raw.get("weight_decay")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (learning_rate, weight_decay)
    ):
        raise ConfigurationError(f"{prefix} numeric parameters must be numbers.")
    if not 0 < float(learning_rate) <= 1:
        raise ConfigurationError(f"{prefix}.learning_rate must be in (0, 1].")
    if not 0 <= float(weight_decay) <= 0.1:
        raise ConfigurationError(f"{prefix}.weight_decay must be in [0, 0.1].")
    return GRUCandidate(
        name=name,
        lookback=lookback,
        hidden_size=hidden_size,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        max_epochs=max_epochs,
        patience=patience,
    )


def _parse_transformer_candidate(raw: Any, index: int) -> TransformerCandidate:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"transformer.candidates[{index}] must be a table.")
    prefix = f"transformer.candidates[{index}]"
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError(f"{prefix}.name must be a non-empty string.")
    lookback = _positive_int(raw.get("lookback"), f"{prefix}.lookback", 20)
    d_model = _positive_int(raw.get("d_model"), f"{prefix}.d_model", 32)
    nhead = _positive_int(raw.get("nhead"), f"{prefix}.nhead", 32)
    dim_feedforward = _positive_int(
        raw.get("dim_feedforward"), f"{prefix}.dim_feedforward", 64
    )
    if nhead > d_model or d_model % nhead:
        raise ConfigurationError(f"{prefix}.nhead must divide d_model.")
    max_epochs = _positive_int(raw.get("max_epochs"), f"{prefix}.max_epochs", 100)
    patience = _positive_int(raw.get("patience"), f"{prefix}.patience", max_epochs)
    learning_rate = raw.get("learning_rate")
    weight_decay = raw.get("weight_decay")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (learning_rate, weight_decay)
    ):
        raise ConfigurationError(f"{prefix} numeric parameters must be numbers.")
    if not 0 < float(learning_rate) <= 1:
        raise ConfigurationError(f"{prefix}.learning_rate must be in (0, 1].")
    if not 0 <= float(weight_decay) <= 0.1:
        raise ConfigurationError(f"{prefix}.weight_decay must be in [0, 0.1].")
    return TransformerCandidate(
        name=name,
        lookback=lookback,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        max_epochs=max_epochs,
        patience=patience,
    )


def load_config(path: Path) -> AppConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"Config file not found: {path}. Pass an existing file with --config."
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(
            f"Invalid TOML in {path}: {exc} Fix the file or pass a different --config."
        ) from exc

    try:
        data_raw = raw["data"]
        splits_raw = raw["splits"]
        features_raw = raw["features"]
        research_raw = raw["research"]
        run_raw = raw["run"]
        xgb_raw = raw["xgboost"]
    except KeyError as exc:
        raise ConfigurationError(
            f"Config {path} is missing table {exc.args[0]!r}. Add it or pass a different --config."
        ) from exc

    symbols = tuple(data_raw.get("symbols", ()))
    if len(symbols) != len(set(symbols)):
        raise ConfigurationError("data.symbols must not contain duplicates.")
    missing = [symbol for symbol in REQUIRED_SYMBOLS if symbol not in symbols]
    unexpected = [symbol for symbol in symbols if symbol not in REQUIRED_SYMBOLS]
    if missing:
        raise ConfigurationError(f"data.symbols is missing required symbols: {', '.join(missing)}")
    if unexpected:
        raise ConfigurationError(f"data.symbols contains unsupported symbols: {', '.join(unexpected)}")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ConfigurationError("data.symbols must contain non-empty strings.")

    data = DataConfig(
        symbols=symbols,
        start_inclusive=_date(data_raw.get("start_inclusive"), "data.start_inclusive"),
        end_exclusive=_date(data_raw.get("end_exclusive"), "data.end_exclusive"),
        interval=data_raw.get("interval", "1d"),
        cache_dir=Path(data_raw.get("cache_dir", "data/yahoo_cache")),
    )
    if data.start_inclusive >= data.end_exclusive:
        raise ConfigurationError("data.start_inclusive must be before data.end_exclusive.")
    if data.interval != "1d":
        raise ConfigurationError("Only a daily interval is supported by this baseline.")

    splits = SplitConfig(
        train_end_exclusive=_date(splits_raw.get("train_end_exclusive"), "splits.train_end_exclusive"),
        validation_end_exclusive=_date(
            splits_raw.get("validation_end_exclusive"), "splits.validation_end_exclusive"
        ),
        test_end_exclusive=_date(splits_raw.get("test_end_exclusive"), "splits.test_end_exclusive"),
        min_rows=_positive_int(splits_raw.get("min_rows"), "splits.min_rows", 1_000_000),
    )
    _check_dates(data, splits)

    group = features_raw.get("group")
    if group not in FEATURE_GROUPS:
        raise ConfigurationError(f"features.group must be one of {FEATURE_GROUPS}.")
    lags = tuple(features_raw.get("return_lags", ()))
    windows = tuple(features_raw.get("rolling_windows", ()))
    if not lags or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in lags):
        raise ConfigurationError("features.return_lags must contain positive integers.")
    if len(lags) != len(set(lags)):
        raise ConfigurationError("features.return_lags must not contain duplicates.")
    if not windows or any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in windows):
        raise ConfigurationError("features.rolling_windows must contain integers >= 2.")
    if len(windows) != len(set(windows)):
        raise ConfigurationError("features.rolling_windows must not contain duplicates.")
    availability_lag = features_raw.get("usd_idr_availability_lag", 1)
    peer_lag = features_raw.get("peer_availability_lag", 1)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (availability_lag, peer_lag)):
        raise ConfigurationError("External availability lags must be non-negative integers.")
    peers = tuple(features_raw.get("peers", ()))
    if any(peer not in symbols or peer == "IDR=X" for peer in peers):
        raise ConfigurationError("features.peers must name configured equity symbols only.")

    features = FeatureConfig(
        group=group,
        return_lags=lags,
        rolling_windows=windows,
        usd_idr_availability_lag=availability_lag,
        peer_availability_lag=peer_lag,
        peers=peers,
    )

    if "risk" not in raw:
        risk_raw: Any = {"confidence": 0.95, "portfolio_value_idr": 1_000_000}
    else:
        risk_raw = raw["risk"]
    if not isinstance(risk_raw, dict):
        raise ConfigurationError("risk must be a table.")
    confidence = _finite_number(risk_raw.get("confidence"), "risk.confidence")
    if not 0 < confidence < 1:
        raise ConfigurationError("risk.confidence must be strictly between 0 and 1.")
    portfolio_value_idr = _finite_number(
        risk_raw.get("portfolio_value_idr"), "risk.portfolio_value_idr"
    )
    if portfolio_value_idr <= 0:
        raise ConfigurationError("risk.portfolio_value_idr must be positive.")
    risk = RiskConfig(confidence, portfolio_value_idr)

    research = ResearchConfig(
        forecast_timestamp_assumption=str(research_raw.get("forecast_timestamp_assumption", "")),
        adro_aadi_structural_break_caveat=str(research_raw.get("adro_aadi_structural_break_caveat", "")),
        coal_feature_policy=str(research_raw.get("coal_feature_policy", "")),
    )
    if not all((research.forecast_timestamp_assumption, research.adro_aadi_structural_break_caveat, research.coal_feature_policy)):
        raise ConfigurationError("research must state timestamp, ADRO/AADI, and coal-feature assumptions.")

    seed = _positive_int(run_raw.get("seed"), "run.seed", 2**31 - 1)
    candidates_raw = xgb_raw.get("candidates", ())
    if not isinstance(candidates_raw, list) or not candidates_raw or len(candidates_raw) > 8:
        raise ConfigurationError("xgboost.candidates must contain between 1 and 8 candidates.")
    candidates = tuple(_parse_candidate(candidate, index) for index, candidate in enumerate(candidates_raw))
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ConfigurationError("xgboost candidate names must be unique.")
    selection_metric = xgb_raw.get("selection_metric", "rmse")
    if selection_metric not in ("rmse", "mae", "directional_accuracy"):
        raise ConfigurationError(
            "xgboost.selection_metric must be rmse, mae, or directional_accuracy."
        )

    gru_raw = raw.get("gru")
    if gru_raw is None:
        candidates_raw = [dict(candidate) for candidate in DEFAULT_GRU_CANDIDATES]
    elif not isinstance(gru_raw, dict):
        raise ConfigurationError("gru must be a table.")
    else:
        candidates_raw = gru_raw.get("candidates", ())
    if not isinstance(candidates_raw, list) or len(candidates_raw) != 2:
        raise ConfigurationError("gru.candidates must contain exactly 2 candidates.")
    gru_candidates = tuple(
        _parse_gru_candidate(candidate, index)
        for index, candidate in enumerate(candidates_raw)
    )
    if len({candidate.name for candidate in gru_candidates}) != len(gru_candidates):
        raise ConfigurationError("gru candidate names must be unique.")

    transformer_raw = raw.get("transformer")
    if transformer_raw is None:
        candidates_raw = [dict(candidate) for candidate in DEFAULT_TRANSFORMER_CANDIDATES]
    elif not isinstance(transformer_raw, dict):
        raise ConfigurationError("transformer must be a table.")
    else:
        candidates_raw = transformer_raw.get("candidates", ())
    if not isinstance(candidates_raw, list) or len(candidates_raw) != 2:
        raise ConfigurationError("transformer.candidates must contain exactly 2 candidates.")
    transformer_candidates = tuple(
        _parse_transformer_candidate(candidate, index)
        for index, candidate in enumerate(candidates_raw)
    )
    if len({candidate.name for candidate in transformer_candidates}) != len(transformer_candidates):
        raise ConfigurationError("transformer candidate names must be unique.")
    all_candidates = (*candidates, *gru_candidates, *transformer_candidates)
    if len({candidate.name for candidate in all_candidates}) != len(all_candidates):
        raise ConfigurationError("Candidate names must be unique across model families.")

    expanding_raw = raw.get("expanding_folds", [])
    if expanding_raw is None:
        expanding_raw = []
    if not isinstance(expanding_raw, list):
        raise ConfigurationError("expanding_folds must be a list of tables.")
    if len(expanding_raw) > 8:
        raise ConfigurationError("expanding_folds must contain at most 8 folds.")
    expanding_folds: list[ExpandingFoldConfig] = []
    for index, item in enumerate(expanding_raw):
        if not isinstance(item, dict):
            raise ConfigurationError(f"expanding_folds[{index}] must be a table.")
        name = item.get("name") or f"fold{index + 1}"
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(f"expanding_folds[{index}].name must be a non-empty string.")
        train_end = _date(item.get("train_end_exclusive"), f"expanding_folds[{index}].train_end_exclusive")
        val_end = _date(item.get("validation_end_exclusive"), f"expanding_folds[{index}].validation_end_exclusive")
        if not train_end < val_end:
            raise ConfigurationError(f"expanding_folds[{index}] train_end must be before validation_end.")
        expanding_folds.append(ExpandingFoldConfig(str(name), train_end, val_end))
    if len({fold.name for fold in expanding_folds}) != len(expanding_folds):
        raise ConfigurationError("expanding_folds names must be unique.")

    external: ExternalConfig | None = None
    external_raw = raw.get("external", {})
    if not isinstance(external_raw, dict):
        raise ConfigurationError("external must be a table.")
    usd_raw = external_raw.get("usd_idr", None)
    if usd_raw is not None:
        if not isinstance(usd_raw, dict):
            raise ConfigurationError("external.usd_idr must be a table.")
        symbol = str(usd_raw.get("symbol", "IDR=X"))
        if symbol != "IDR=X":
            raise ConfigurationError("external.usd_idr.symbol must be IDR=X in this study.")
        lag = usd_raw.get("availability_lag", availability_lag)
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
            raise ConfigurationError("external.usd_idr.availability_lag must be a non-negative integer.")
        external = ExternalConfig(symbol, lag)

    return AppConfig(
        data=data,
        splits=splits,
        features=features,
        risk=risk,
        research=research,
        run=RunConfig(seed=seed, output_dir=Path(run_raw.get("output_dir", "outputs"))),
        xgboost_candidates=candidates,
        gru_candidates=gru_candidates,
        transformer_candidates=transformer_candidates,
        selection_metric=selection_metric,
        expanding_folds=tuple(expanding_folds),
        external=external,
    )
