"""Chronological training, validation-based selection, and artifact saving."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from energy_forecasting.config import (
    DAILY_DATE_COLUMN,
    DAILY_TARGET_COLUMN,
    FEATURE_DATA_PATH,
    FIGURE_DIR,
    MODEL_ARTIFACT_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    REPORT_DIR,
    SPLIT_METADATA_PATH,
    TEST_METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    VALIDATION_METRICS_PATH,
)
from energy_forecasting.evaluate import (
    PERSISTENCE_MODEL_NAME,
    evaluate_candidates,
    evaluate_selected_model,
    plot_residual_analysis,
    plot_test_predictions,
    plot_validation_comparison,
)
from energy_forecasting.features import FEATURE_COLUMNS


RANDOM_SEED = 42


@dataclass
class ChronologicalSplits:
    """Eligible training, validation, and test rows plus date metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any]


def _normalize_feature_data(
    feature_data: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    required = (DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN, *feature_columns)
    missing = [column for column in required if column not in feature_data]
    if missing:
        raise ValueError("Feature data is missing required column(s): " + ", ".join(missing))
    if feature_data.empty:
        raise ValueError("Feature data is empty.")

    data = feature_data.loc[:, required].copy()
    data[DAILY_DATE_COLUMN] = pd.to_datetime(data[DAILY_DATE_COLUMN], errors="coerce")
    if data[DAILY_DATE_COLUMN].isna().any():
        raise ValueError("Feature data contains invalid dates.")
    data[DAILY_DATE_COLUMN] = data[DAILY_DATE_COLUMN].dt.normalize()
    if data[DAILY_DATE_COLUMN].duplicated().any():
        raise ValueError("Feature data must contain at most one row per date.")

    for column in (DAILY_TARGET_COLUMN, *feature_columns):
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"Feature column '{column}' must be numeric or missing.")
        if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Feature column '{column}' contains infinite values.")
        data[column] = numeric
    return data.sort_values(DAILY_DATE_COLUMN, kind="stable").reset_index(drop=True)


def chronological_split(
    feature_data: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> ChronologicalSplits:
    """Split calendar dates chronologically, then exclude unusable rows.

    Calendar boundaries are selected before missing targets or unavailable
    feature rows are dropped, so gaps do not move observations across splits.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    data = _normalize_feature_data(feature_data, feature_columns)
    date_count = len(data)
    train_end = math.floor(date_count * train_fraction)
    validation_end = math.floor(date_count * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= date_count:
        raise ValueError("Not enough calendar dates for nonempty train/validation/test splits.")

    sections = {
        "train": data.iloc[:train_end].copy(),
        "validation": data.iloc[train_end:validation_end].copy(),
        "test": data.iloc[validation_end:].copy(),
    }
    eligible: dict[str, pd.DataFrame] = {}
    split_metadata: dict[str, Any] = {}
    for name, section in sections.items():
        missing_target = section[DAILY_TARGET_COLUMN].isna()
        missing_features = section.loc[:, feature_columns].isna().any(axis=1)
        keep = ~missing_target & ~missing_features
        eligible[name] = section.loc[keep].reset_index(drop=True)
        split_metadata[name] = {
            "calendar_start_date": section[DAILY_DATE_COLUMN].iloc[0].date().isoformat(),
            "calendar_end_date": section[DAILY_DATE_COLUMN].iloc[-1].date().isoformat(),
            "calendar_rows": int(len(section)),
            "eligible_rows": int(keep.sum()),
            "excluded_rows": int((~keep).sum()),
            "missing_target_rows": int(missing_target.sum()),
            "rows_with_unavailable_features": int(missing_features.sum()),
            "eligible_start_date": (
                eligible[name][DAILY_DATE_COLUMN].iloc[0].date().isoformat()
                if not eligible[name].empty
                else None
            ),
            "eligible_end_date": (
                eligible[name][DAILY_DATE_COLUMN].iloc[-1].date().isoformat()
                if not eligible[name].empty
                else None
            ),
        }
        if eligible[name].empty:
            raise ValueError(f"The {name} split has no rows with complete target and features.")

    metadata = {
        "strategy": "chronological calendar-date split before dropping incomplete rows",
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "total_calendar_rows": int(date_count),
        "feature_columns": list(feature_columns),
        "splits": split_metadata,
    }
    return ChronologicalSplits(
        train=eligible["train"],
        validation=eligible["validation"],
        test=eligible["test"],
        metadata=metadata,
    )


def build_candidate_models(random_seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Construct reproducible, modest model configurations."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_seed,
        ),
    }


def fit_candidate_models(
    models: dict[str, Any],
    training_data: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Fit candidate estimators using only the supplied training partition."""
    inputs = training_data.loc[:, feature_columns]
    target = training_data[DAILY_TARGET_COLUMN]
    for model in models.values():
        model.fit(inputs, target)
    return models


def select_model(validation_metrics: pd.DataFrame) -> str:
    """Select the first minimum-MAE candidate from validation results only."""
    if validation_metrics.empty or not {"model", "mae_kw"}.issubset(validation_metrics.columns):
        raise ValueError("Validation metrics must contain model and mae_kw columns.")
    scores = pd.to_numeric(validation_metrics["mae_kw"], errors="coerce")
    if scores.isna().all():
        raise ValueError("Validation metrics contain no usable MAE values.")
    return str(validation_metrics.loc[scores.idxmin(), "model"])


def save_model_artifacts(
    model_path: str | Path,
    metadata_path: str | Path,
    model_name: str,
    model: Any,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist the selected estimator bundle and readable JSON metadata."""
    artifact_path = Path(model_path)
    json_path = Path(metadata_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": model_name,
        "model": model,
        "feature_columns": metadata["feature_columns"],
    }
    joblib.dump(bundle, artifact_path)
    with json_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, allow_nan=False)
    return artifact_path, json_path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def train_models(
    feature_data: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR,
    report_dir: str | Path = REPORT_DIR,
    figure_dir: str | Path = FIGURE_DIR,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, Any]:
    """Fit candidates, select on validation MAE, then evaluate once on test."""
    splits = chronological_split(
        feature_data,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    models = fit_candidate_models(build_candidate_models(), splits.train)
    validation_metrics, _ = evaluate_candidates(
        models, feature_data, splits.validation, FEATURE_COLUMNS
    )
    selected_name = select_model(validation_metrics)
    selected_model = models.get(selected_name)
    test_metrics, test_predictions = evaluate_selected_model(
        selected_name,
        selected_model,
        feature_data,
        splits.test,
        FEATURE_COLUMNS,
    )

    validation_metrics = validation_metrics.assign(
        selected=validation_metrics["model"].eq(selected_name)
    )
    test_metrics_frame = pd.DataFrame([test_metrics])
    comparison = validation_metrics.copy()
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    test_metrics_frame.to_csv(output_dir / "test_metrics.csv", index=False)
    test_predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    model_parameters = (
        selected_model.get_params(deep=False) if selected_model is not None else {}
    )
    metadata = {
        "selected_model": selected_name,
        "target_column": DAILY_TARGET_COLUMN,
        "target_units": "kW",
        "feature_columns": list(FEATURE_COLUMNS),
        "selection_metric": "validation MAE (kW)",
        "training_rows": int(len(splits.train)),
        "random_seed": RANDOM_SEED,
        "model_parameters": model_parameters,
        "baseline_policy": (
            "For each forecast date, use the most recent nonmissing target strictly "
            "before that date; a missing previous calendar day may use an older "
            "already-observed target, never a later value."
        ),
        "evaluation_protocol": (
            "Rolling one-day-ahead predictions. Models are fitted once on training "
            "rows; observed earlier validation/test targets may be used only in "
            "past-only features for later forecast dates, never for fitting or selection."
        ),
        "missing_data_policy": (
            "No target imputation. Calendar splits are made before removing rows "
            "with missing targets or unavailable predictors. Comparisons use dates "
            "where every candidate, including persistence, can be scored."
        ),
        "split_metadata": splits.metadata,
        "validation_scored_start_date": validation_metrics["scored_start_date"].iloc[0],
        "validation_scored_end_date": validation_metrics["scored_end_date"].iloc[0],
        "test_scored_start_date": test_metrics["scored_start_date"],
        "test_scored_end_date": test_metrics["scored_end_date"],
    }

    model_path = Path(model_dir) / Path(MODEL_ARTIFACT_PATH).name
    metadata_path = Path(model_dir) / Path(MODEL_METADATA_PATH).name
    save_model_artifacts(model_path, metadata_path, selected_name, selected_model, metadata)
    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=_json_default, allow_nan=False)

    plot_validation_comparison(validation_metrics, selected_name, figure_dir)
    plot_test_predictions(test_predictions, selected_name, figure_dir)
    plot_residual_analysis(test_predictions, selected_name, figure_dir)
    return {
        "selected_model": selected_name,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics_frame,
        "split_metadata": metadata["split_metadata"],
        "model_path": model_path,
        "metadata_path": metadata_path,
        "report_dir": output_dir,
    }


def run_training_pipeline(
    feature_path: str | Path = FEATURE_DATA_PATH,
    **training_options: Any,
) -> dict[str, Any]:
    """Load an existing feature CSV and run Stage 3 without preprocessing."""
    source_path = Path(feature_path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Feature data not found at '{source_path}'. Run scripts/run_pipeline.py first."
        )
    feature_data = pd.read_csv(source_path, parse_dates=[DAILY_DATE_COLUMN])
    return train_models(feature_data, **training_options)