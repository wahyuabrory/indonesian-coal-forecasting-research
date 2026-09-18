from __future__ import annotations

from dataclasses import dataclass
import copy
import random
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import AppConfig, GRUCandidate, TransformerCandidate, XGBoostCandidate
from .features import TARGET_COLUMN, SequenceDataset


class ModelingError(RuntimeError):
    """Raised when a configured experiment cannot be fit safely."""


@dataclass(frozen=True)
class Partitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True)
class SequencePartitions:
    train: SequenceDataset
    validation: SequenceDataset
    test: SequenceDataset

    def as_dict(self) -> dict[str, SequenceDataset]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True)
class EvaluationResult:
    metrics: list[dict[str, Any]]
    selected_model: str
    selected_candidate: str | None
    selected_parameters: dict[str, int | float | str]
    selected_framework: str | None
    selected_framework_version: str | None
    selected_best_epoch: int | None
    selected_parameter_count: int | None
    model_metadata: dict[str, dict[str, Any]]
    candidate_parameters: dict[str, dict[str, dict[str, int | float | str]]]
    scaling_rule: str
    scaler: StandardScaler
    split_metadata: dict[str, dict[str, str | int]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_dataset(dataset: pd.DataFrame, feature_columns: list[str], config: AppConfig) -> Partitions:
    required = {"Date", TARGET_COLUMN, *feature_columns}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ModelingError(
            f"Feature dataset is missing required columns: {', '.join(missing)}. "
            "Rebuild the feature dataset from the configured snapshots."
        )

    boundaries = (
        ("train", config.data.start_inclusive, config.splits.train_end_exclusive),
        (
            "validation",
            config.splits.train_end_exclusive,
            config.splits.validation_end_exclusive,
        ),
        ("test", config.splits.validation_end_exclusive, config.splits.test_end_exclusive),
    )
    parts: dict[str, pd.DataFrame] = {}
    for name, start, end in boundaries:
        mask = (dataset["Date"] >= pd.Timestamp(start)) & (dataset["Date"] < pd.Timestamp(end))
        part = dataset.loc[mask].reset_index(drop=True)
        if len(part) < config.splits.min_rows:
            raise ModelingError(
                f"Insufficient rows for {name}: found {len(part)}, need at least "
                f"{config.splits.min_rows} between {start.isoformat()} and {end.isoformat()} exclusive. "
                "Adjust the split dates or provide a longer valid snapshot."
            )
        parts[name] = part
    return Partitions(parts["train"], parts["validation"], parts["test"])


def split_sequence_dataset(dataset: SequenceDataset, config: AppConfig) -> SequencePartitions:
    if dataset.features.ndim != 3:
        raise ModelingError("Sequence features must have shape (rows, lookback, channels).")
    if len(dataset.dates) != len(dataset.targets):
        raise ModelingError("Sequence dates and targets must contain the same number of rows.")

    boundaries = (
        ("train", config.data.start_inclusive, config.splits.train_end_exclusive),
        (
            "validation",
            config.splits.train_end_exclusive,
            config.splits.validation_end_exclusive,
        ),
        ("test", config.splits.validation_end_exclusive, config.splits.test_end_exclusive),
    )
    parts: dict[str, SequenceDataset] = {}
    for name, start, end in boundaries:
        mask = (dataset.dates >= pd.Timestamp(start)) & (dataset.dates < pd.Timestamp(end))
        indices = np.flatnonzero(mask.to_numpy())
        if len(indices) < config.splits.min_rows:
            raise ModelingError(
                f"Insufficient sequence rows for {name}: found {len(indices)}, need at least "
                f"{config.splits.min_rows} between {start.isoformat()} and {end.isoformat()} exclusive."
            )
        parts[name] = SequenceDataset(
            dataset.dates.iloc[indices].reset_index(drop=True),
            dataset.features[indices],
            dataset.targets[indices],
            dataset.channel_names,
            dataset.lookback,
        )
    return SequencePartitions(parts["train"], parts["validation"], parts["test"])


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ModelingError("Predictions are not finite or do not match target rows.")
    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    directional_accuracy = float(np.mean(np.sign(actual) == np.sign(predicted)))
    return mae, rmse, directional_accuracy


def _metric_row(
    model: str,
    split_name: str,
    frame: pd.DataFrame,
    predicted: np.ndarray,
    selection_stage: str,
    candidate_name: str | None = None,
) -> dict[str, Any]:
    actual = frame[TARGET_COLUMN].to_numpy(dtype=float)
    mae, rmse, directional_accuracy = _metric_values(actual, predicted)
    return {
        "model": model,
        "candidate": candidate_name or "",
        "split": split_name,
        "selection_stage": selection_stage,
        "rows": len(frame),
        "start_date": frame["Date"].min().date().isoformat(),
        "end_date": frame["Date"].max().date().isoformat(),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }


def _xgboost_model(candidate: XGBoostCandidate, seed: int) -> Any:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise ModelingError(
            "XGBoost is unavailable. Install the project dependencies or run with --model baseline."
        ) from exc
    return XGBRegressor(
        **candidate.parameters(),
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def _selection_key(row: dict[str, Any], metric: str) -> tuple[float, float, float, str]:
    primary = float(row[metric])
    if metric == "directional_accuracy":
        primary = -primary
    return (primary, float(row["rmse"]), float(row["mae"]), str(row["candidate"]))


def _selection_value(row: dict[str, Any], metric: str) -> float:
    value = float(row[metric])
    return -value if metric == "directional_accuracy" else value


def _sequence_metric_row(
    model: str,
    split_name: str,
    dataset: SequenceDataset,
    predicted: np.ndarray,
    selection_stage: str,
    candidate_name: str | None = None,
) -> dict[str, Any]:
    actual = dataset.targets.astype(float)
    predicted = np.asarray(predicted, dtype=float)
    mae, rmse, directional_accuracy = _metric_values(actual, predicted)
    return {
        "model": model,
        "candidate": candidate_name or "",
        "split": split_name,
        "selection_stage": selection_stage,
        "rows": len(dataset.dates),
        "start_date": dataset.dates.min().date().isoformat(),
        "end_date": dataset.dates.max().date().isoformat(),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }


@dataclass
class _GRUFitted:
    candidate: GRUCandidate
    model: Any
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    validation_row: dict[str, Any]
    best_epoch: int
    parameter_count: int
    framework_version: str

    def predict(self, dataset: SequenceDataset) -> np.ndarray:
        import torch

        flat = dataset.features.reshape(-1, dataset.features.shape[-1])
        scaled_features = self.feature_scaler.transform(flat).reshape(dataset.features.shape)
        inputs = torch.from_numpy(scaled_features.astype(np.float32, copy=False))
        self.model.eval()
        with torch.no_grad():
            scaled_prediction = self.model(inputs).detach().cpu().numpy()
        return self.target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).ravel()


def _train_gru_candidate(
    candidate: GRUCandidate,
    partitions: SequencePartitions,
    seed: int,
) -> _GRUFitted:
    try:
        import torch
    except ImportError as exc:
        raise ModelingError(
            "PyTorch is required for --model gru/all. Install it with "
            "python -m pip install -e '.[deep]'"
        ) from exc

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    channels = partitions.train.features.shape[-1]
    feature_scaler = StandardScaler()
    train_flat = partitions.train.features.reshape(-1, channels)
    feature_scaler.fit(train_flat)

    def scale_features(dataset: SequenceDataset) -> np.ndarray:
        flat = dataset.features.reshape(-1, channels)
        return feature_scaler.transform(flat).reshape(dataset.features.shape).astype(
            np.float32, copy=False
        )

    target_scaler = StandardScaler()
    target_scaler.fit(partitions.train.targets.reshape(-1, 1))
    x_train = torch.from_numpy(scale_features(partitions.train))
    x_validation = torch.from_numpy(scale_features(partitions.validation))
    y_train = torch.from_numpy(
        target_scaler.transform(partitions.train.targets.reshape(-1, 1)).ravel().astype(
            np.float32, copy=False
        )
    )
    y_validation = torch.from_numpy(
        target_scaler.transform(partitions.validation.targets.reshape(-1, 1)).ravel().astype(
            np.float32, copy=False
        )
    )

    class GRUNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(
                input_size=channels,
                hidden_size=candidate.hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.output = torch.nn.Linear(candidate.hidden_size, 1)

        def forward(self, inputs: Any) -> Any:
            _, hidden = self.gru(inputs)
            return self.output(hidden[-1]).squeeze(-1)

    model = GRUNetwork()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, candidate.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_validation), y_validation).item())
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= candidate.patience:
                break

    if best_state is None:
        raise ModelingError(f"GRU candidate {candidate.name!r} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    fitted = _GRUFitted(
        candidate=candidate,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        validation_row={},
        best_epoch=best_epoch,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        framework_version=str(torch.__version__),
    )
    validation_prediction = fitted.predict(partitions.validation)
    fitted.validation_row = _sequence_metric_row(
        "gru_candidate",
        "validation",
        partitions.validation,
        validation_prediction,
        "candidate_selection",
        candidate.name,
    )
    return fitted


@dataclass
class _TransformerFitted:
    candidate: TransformerCandidate
    model: Any
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    validation_row: dict[str, Any]
    best_epoch: int
    parameter_count: int
    framework_version: str

    def predict(self, dataset: SequenceDataset) -> np.ndarray:
        import torch

        flat = dataset.features.reshape(-1, dataset.features.shape[-1])
        scaled_features = self.feature_scaler.transform(flat).reshape(dataset.features.shape)
        inputs = torch.from_numpy(scaled_features.astype(np.float32, copy=False))
        self.model.eval()
        with torch.no_grad():
            scaled_prediction = self.model(inputs).detach().cpu().numpy()
        return self.target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).ravel()


def _train_transformer_candidate(
    candidate: TransformerCandidate,
    partitions: SequencePartitions,
    seed: int,
) -> _TransformerFitted:
    try:
        import torch
    except ImportError as exc:
        raise ModelingError(
            "PyTorch is required for --model transformer/all. Install it with "
            "python -m pip install -e '.[deep]'"
        ) from exc

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")

    channels = partitions.train.features.shape[-1]
    feature_scaler = StandardScaler()
    train_flat = partitions.train.features.reshape(-1, channels)
    feature_scaler.fit(train_flat)

    def scale_features(dataset: SequenceDataset) -> np.ndarray:
        flat = dataset.features.reshape(-1, channels)
        return feature_scaler.transform(flat).reshape(dataset.features.shape).astype(
            np.float32, copy=False
        )

    target_scaler = StandardScaler()
    target_scaler.fit(partitions.train.targets.reshape(-1, 1))
    x_train = torch.from_numpy(scale_features(partitions.train))
    x_validation = torch.from_numpy(scale_features(partitions.validation))
    y_train = torch.from_numpy(
        target_scaler.transform(partitions.train.targets.reshape(-1, 1)).ravel().astype(
            np.float32, copy=False
        )
    )
    y_validation = torch.from_numpy(
        target_scaler.transform(partitions.validation.targets.reshape(-1, 1)).ravel().astype(
            np.float32, copy=False
        )
    )

    class TransformerNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(channels, candidate.d_model)
            self.position_embedding = torch.nn.Embedding(candidate.lookback, candidate.d_model)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=candidate.d_model,
                nhead=candidate.nhead,
                dim_feedforward=candidate.dim_feedforward,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.output = torch.nn.Linear(candidate.d_model, 1)

        def forward(self, inputs: Any) -> Any:
            positions = torch.arange(inputs.shape[1], device=inputs.device)
            encoded = self.input_projection(inputs) + self.position_embedding(positions)
            encoded = self.encoder(encoded)
            return self.output(encoded[:, -1, :]).squeeze(-1)

    model = TransformerNetwork().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, candidate.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_validation), y_validation).item())
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= candidate.patience:
                break

    if best_state is None:
        raise ModelingError(f"Transformer candidate {candidate.name!r} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    fitted = _TransformerFitted(
        candidate=candidate,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        validation_row={},
        best_epoch=best_epoch,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        framework_version=str(torch.__version__),
    )
    validation_prediction = fitted.predict(partitions.validation)
    fitted.validation_row = _sequence_metric_row(
        "transformer_candidate",
        "validation",
        partitions.validation,
        validation_prediction,
        "candidate_selection",
        candidate.name,
    )
    return fitted


def _model_details(
    framework: str | None,
    framework_version: str | None,
    candidate: str | None,
    parameters: dict[str, int | float | str],
    best_epoch: int | None,
    parameter_count: int | None,
) -> dict[str, Any]:
    return {
        "framework": framework,
        "framework_version": framework_version,
        "candidate": candidate,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
    }


def evaluate(
    partitions: Partitions,
    feature_columns: list[str],
    config: AppConfig,
    model_mode: Literal["baseline", "xgboost", "both", "gru", "transformer", "all"],
    sequence_partitions: dict[str, SequencePartitions] | None = None,
) -> EvaluationResult:
    if model_mode not in ("baseline", "xgboost", "both", "gru", "transformer", "all"):
        raise ModelingError(f"Unsupported model mode {model_mode!r}.")
    if model_mode in ("gru", "transformer", "all") and sequence_partitions is None:
        raise ModelingError(
            "GRU and Transformer modes require sequence partitions for every configured candidate."
        )

    seed_everything(config.run.seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(partitions.train[feature_columns])
    x_validation = scaler.transform(partitions.validation[feature_columns])
    y_train = partitions.train[TARGET_COLUMN].to_numpy(dtype=float)
    metrics: list[dict[str, Any]] = []
    candidate_parameters: dict[str, dict[str, dict[str, int | float | str]]] = {}

    baseline_validation_row: dict[str, Any] | None = None
    if model_mode in ("baseline", "both", "all"):
        baseline_validation_row = _metric_row(
            "zero_return_naive",
            "validation",
            partitions.validation,
            np.zeros(len(partitions.validation), dtype=float),
            "validation_selection",
        )
        metrics.append(baseline_validation_row)

    xgb_candidates: list[tuple[XGBoostCandidate, Any, dict[str, Any]]] = []
    best_xgb: tuple[XGBoostCandidate, Any, dict[str, Any]] | None = None
    xgb_framework_version: str | None = None
    if model_mode in ("xgboost", "both", "all"):
        candidate_parameters["xgboost"] = {
            candidate.name: candidate.parameters() for candidate in config.xgboost_candidates
        }
        try:
            import xgboost

            xgb_framework_version = str(xgboost.__version__)
        except Exception:
            xgb_framework_version = None
        for candidate in config.xgboost_candidates:
            model = _xgboost_model(candidate, config.run.seed)
            model.fit(x_train, y_train)
            validation_prediction = np.asarray(model.predict(x_validation), dtype=float)
            validation_row = _metric_row(
                "xgboost_candidate",
                "validation",
                partitions.validation,
                validation_prediction,
                "candidate_selection",
                candidate.name,
            )
            metrics.append(validation_row)
            xgb_candidates.append((candidate, model, validation_row))

        best_xgb = min(
            xgb_candidates,
            key=lambda item: _selection_key(item[2], config.selection_metric),
        )
        best_xgb_validation = {
            **best_xgb[2],
            "model": "xgboost",
            "selection_stage": "xgboost_validation_winner",
        }
        metrics.append(best_xgb_validation)

    best_gru: _GRUFitted | None = None
    torch_framework_version: str | None = None
    if model_mode in ("gru", "all"):
        if sequence_partitions is None:
            raise ModelingError("GRU mode requires sequence partitions.")
        candidate_parameters["gru"] = {
            candidate.name: candidate.parameters() for candidate in config.gru_candidates
        }
        gru_candidates: list[_GRUFitted] = []
        for candidate in config.gru_candidates:
            candidate_partitions = sequence_partitions.get(candidate.name)
            if candidate_partitions is None:
                raise ModelingError(
                    f"No sequence partitions were built for GRU candidate {candidate.name!r}."
                )
            fitted = _train_gru_candidate(candidate, candidate_partitions, config.run.seed)
            metrics.append(fitted.validation_row)
            gru_candidates.append(fitted)
            torch_framework_version = fitted.framework_version
        best_gru = min(
            gru_candidates,
            key=lambda fitted: _selection_key(fitted.validation_row, config.selection_metric),
        )
        metrics.append(
            {
                **best_gru.validation_row,
                "model": "gru",
                "selection_stage": "gru_validation_winner",
            }
        )

    best_transformer: _TransformerFitted | None = None
    transformer_framework_version: str | None = None
    if model_mode in ("transformer", "all"):
        if sequence_partitions is None:
            raise ModelingError("Transformer mode requires sequence partitions.")
        candidate_parameters["transformer"] = {
            candidate.name: candidate.parameters()
            for candidate in config.transformer_candidates
        }
        transformer_candidates: list[_TransformerFitted] = []
        for candidate in config.transformer_candidates:
            candidate_partitions = sequence_partitions.get(candidate.name)
            if candidate_partitions is None:
                raise ModelingError(
                    f"No sequence partitions were built for Transformer candidate {candidate.name!r}."
                )
            fitted = _train_transformer_candidate(
                candidate,
                candidate_partitions,
                config.run.seed,
            )
            metrics.append(fitted.validation_row)
            transformer_candidates.append(fitted)
            transformer_framework_version = fitted.framework_version
        best_transformer = min(
            transformer_candidates,
            key=lambda fitted: _selection_key(fitted.validation_row, config.selection_metric),
        )
        metrics.append(
            {
                **best_transformer.validation_row,
                "model": "transformer",
                "selection_stage": "transformer_validation_winner",
            }
        )

    if model_mode == "baseline":
        selected_model = "zero_return_naive"
        selected_candidate = None
        selected_parameters: dict[str, int | float | str] = {}
    elif model_mode == "xgboost":
        if best_xgb is None:
            raise ModelingError("No XGBoost candidates were available for validation selection.")
        selected_model = "xgboost"
        selected_candidate = best_xgb[0].name
        selected_parameters = best_xgb[0].parameters()
    elif model_mode == "gru":
        if best_gru is None:
            raise ModelingError("No GRU candidates were available for validation selection.")
        selected_model = "gru"
        selected_candidate = best_gru.candidate.name
        selected_parameters = best_gru.candidate.parameters()
    elif model_mode == "transformer":
        if best_transformer is None:
            raise ModelingError("No Transformer candidates were available for validation selection.")
        selected_model = "transformer"
        selected_candidate = best_transformer.candidate.name
        selected_parameters = best_transformer.candidate.parameters()
    elif model_mode == "both":
        if baseline_validation_row is None or best_xgb is None:
            raise ModelingError("Both-mode selection requires a baseline and at least one XGBoost candidate.")
        baseline_value = _selection_value(baseline_validation_row, config.selection_metric)
        xgb_value = _selection_value(best_xgb[2], config.selection_metric)
        if baseline_value <= xgb_value:
            selected_model = "zero_return_naive"
            selected_candidate = None
            selected_parameters = {}
        else:
            selected_model = "xgboost"
            selected_candidate = best_xgb[0].name
            selected_parameters = best_xgb[0].parameters()
    else:
        if (
            baseline_validation_row is None
            or best_xgb is None
            or best_gru is None
            or best_transformer is None
        ):
            raise ModelingError(
                "All-mode selection requires baseline, XGBoost, GRU, and Transformer "
                "validation results."
            )
        options = (
            (0, "zero_return_naive", None, baseline_validation_row, {}),
            (1, "xgboost", best_xgb[0].name, best_xgb[2], best_xgb[0].parameters()),
            (2, "gru", best_gru.candidate.name, best_gru.validation_row, best_gru.candidate.parameters()),
            (
                3,
                "transformer",
                best_transformer.candidate.name,
                best_transformer.validation_row,
                best_transformer.candidate.parameters(),
            ),
        )
        _, selected_model, selected_candidate, _, selected_parameters = min(
            options,
            key=lambda option: (
                _selection_value(option[3], config.selection_metric),
                option[0],
                float(option[3]["rmse"]),
                float(option[3]["mae"]),
                str(option[2] or ""),
            ),
        )

    if best_xgb is not None:
        x_test = scaler.transform(partitions.test[feature_columns])
        xgb_test_prediction = np.asarray(best_xgb[1].predict(x_test), dtype=float)
        metrics.append(
            _metric_row(
                "xgboost",
                "test",
                partitions.test,
                xgb_test_prediction,
                "frozen_test",
                best_xgb[0].name,
            )
        )
    if baseline_validation_row is not None:
        metrics.append(
            _metric_row(
                "zero_return_naive",
                "test",
                partitions.test,
                np.zeros(len(partitions.test), dtype=float),
                "frozen_test",
            )
        )
    if best_gru is not None and sequence_partitions is not None:
        gru_test = sequence_partitions[best_gru.candidate.name].test
        metrics.append(
            _sequence_metric_row(
                "gru",
                "test",
                gru_test,
                best_gru.predict(gru_test),
                "frozen_test",
                best_gru.candidate.name,
            )
        )
    if best_transformer is not None and sequence_partitions is not None:
        transformer_test = sequence_partitions[best_transformer.candidate.name].test
        metrics.append(
            _sequence_metric_row(
                "transformer",
                "test",
                transformer_test,
                best_transformer.predict(transformer_test),
                "frozen_test",
                best_transformer.candidate.name,
            )
        )

    model_metadata = {
        "zero_return_naive": _model_details(None, None, None, {}, None, None),
    }
    if best_xgb is not None:
        model_metadata["xgboost"] = _model_details(
            "xgboost",
            xgb_framework_version,
            best_xgb[0].name,
            best_xgb[0].parameters(),
            None,
            None,
        )
    if best_gru is not None:
        model_metadata["gru"] = _model_details(
            "torch",
            torch_framework_version,
            best_gru.candidate.name,
            best_gru.candidate.parameters(),
            best_gru.best_epoch,
            best_gru.parameter_count,
        )
    if best_transformer is not None:
        model_metadata["transformer"] = _model_details(
            "torch",
            transformer_framework_version,
            best_transformer.candidate.name,
            best_transformer.candidate.parameters(),
            best_transformer.best_epoch,
            best_transformer.parameter_count,
        )

    selected_details = model_metadata[selected_model]
    if selected_model == "gru":
        scaling_rule = (
            "GRU feature StandardScaler fit on training sequence values only and target "
            "StandardScaler fit on training labels only; validation and test are transform-only."
        )
    elif selected_model == "transformer":
        scaling_rule = (
            "Transformer feature StandardScaler fit on training sequence values only and target "
            "StandardScaler fit on training labels only; validation and test are transform-only."
        )
    else:
        scaling_rule = (
            "Tabular feature StandardScaler fit on training rows only; validation and test are "
            "transform-only."
        )

    split_metadata = {
        name: {
            "rows": len(frame),
            "start_date": frame["Date"].min().date().isoformat(),
            "end_date": frame["Date"].max().date().isoformat(),
        }
        for name, frame in partitions.as_dict().items()
    }
    return EvaluationResult(
        metrics=metrics,
        selected_model=selected_model,
        selected_candidate=selected_candidate,
        selected_parameters=selected_parameters,
        selected_framework=selected_details["framework"],
        selected_framework_version=selected_details["framework_version"],
        selected_best_epoch=selected_details["best_epoch"],
        selected_parameter_count=selected_details["parameter_count"],
        model_metadata=model_metadata,
        candidate_parameters=candidate_parameters,
        scaling_rule=scaling_rule,
        scaler=scaler,
        split_metadata=split_metadata,
    )
