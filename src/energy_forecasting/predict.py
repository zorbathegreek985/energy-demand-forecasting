"""Load a selected artifact and forecast the next calendar day's average power."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from energy_forecasting.config import (
    DAILY_DATE_COLUMN,
    DAILY_TARGET_COLUMN,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
)
from energy_forecasting.evaluate import PERSISTENCE_MODEL_NAME, persistence_predictions
from energy_forecasting.features import FEATURE_COLUMNS, create_features


def load_selected_model(
    model_path: str | Path = MODEL_ARTIFACT_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate a locally generated model bundle and JSON metadata.

    Joblib uses pickle internally; only load artifacts created by this project
    or another source you explicitly trust.
    """
    artifact_path = Path(model_path)
    json_path = Path(metadata_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Selected model not found at '{artifact_path}'. Train models first.")
    if not json_path.is_file():
        raise FileNotFoundError(f"Model metadata not found at '{json_path}'. Train models first.")

    try:
        bundle = joblib.load(artifact_path)
    except (OSError, EOFError, ValueError) as error:
        raise ValueError(f"Could not load model artifact '{artifact_path}': {error}") from error
    try:
        with json_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read model metadata '{json_path}': {error}") from error

    if not isinstance(bundle, dict) or not isinstance(metadata, dict):
        raise ValueError("Model artifact and metadata must contain mappings.")
    if bundle.get("model_name") != metadata.get("selected_model"):
        raise ValueError("Saved model and metadata identify different selected models.")
    if bundle.get("feature_columns") != metadata.get("feature_columns"):
        raise ValueError("Saved model and metadata feature columns do not match.")
    model_name = bundle.get("model_name")
    model = bundle.get("model")
    if model_name == PERSISTENCE_MODEL_NAME:
        if model is not None:
            raise ValueError("Persistence artifact should not contain a fitted estimator.")
    elif model is None or not hasattr(model, "predict"):
        raise ValueError("Saved estimator is missing its prediction method.")
    return bundle, metadata


def forecast_next_day(
    history: pd.DataFrame,
    model_path: str | Path = MODEL_ARTIFACT_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
) -> dict[str, Any]:
    """Forecast the day after the final calendar date in daily history."""
    required = (DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN)
    missing = [column for column in required if column not in history]
    if missing:
        raise ValueError("History is missing required column(s): " + ", ".join(missing))
    if history.empty:
        raise ValueError("History is empty; a next-day forecast cannot be made.")

    daily_history = history.loc[:, required].copy()
    dates = pd.to_datetime(daily_history[DAILY_DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError("History contains invalid dates.")
    daily_history[DAILY_DATE_COLUMN] = dates.dt.normalize()
    if daily_history[DAILY_DATE_COLUMN].duplicated().any():
        raise ValueError("History must contain at most one row per calendar date.")
    numeric_target = pd.to_numeric(daily_history[DAILY_TARGET_COLUMN], errors="coerce")
    invalid_target = daily_history[DAILY_TARGET_COLUMN].notna() & numeric_target.isna()
    if invalid_target.any():
        raise ValueError("History targets must be numeric or missing.")
    daily_history[DAILY_TARGET_COLUMN] = numeric_target
    if daily_history[DAILY_TARGET_COLUMN].notna().sum() == 0:
        raise ValueError("History contains no observed target values.")

    bundle, metadata = load_selected_model(model_path, metadata_path)
    model_name = bundle["model_name"]
    forecast_date = daily_history[DAILY_DATE_COLUMN].max() + pd.Timedelta(days=1)
    extended_history = pd.concat(
        [
            daily_history,
            pd.DataFrame(
                {
                    DAILY_DATE_COLUMN: [forecast_date],
                    DAILY_TARGET_COLUMN: [np.nan],
                }
            ),
        ],
        ignore_index=True,
    )
    next_features = create_features(extended_history).iloc[-1]

    if model_name == PERSISTENCE_MODEL_NAME:
        prediction = persistence_predictions(daily_history, [forecast_date]).iloc[0]
        if not np.isfinite(prediction):
            raise ValueError("No observed target exists before the forecast date.")
    else:
        feature_columns = tuple(metadata.get("feature_columns", ()))
        if feature_columns != FEATURE_COLUMNS:
            raise ValueError("Saved model feature columns do not match the current feature generator.")
        unavailable = [
            column
            for column in feature_columns
            if pd.isna(next_features[column]) or not np.isfinite(float(next_features[column]))
        ]
        if unavailable:
            raise ValueError(
                "Cannot forecast because historical predictors are unavailable: "
                + ", ".join(unavailable)
            )
        model_input = pd.DataFrame(
            [[next_features[column] for column in feature_columns]],
            columns=feature_columns,
        )
        prediction = float(bundle["model"].predict(model_input)[0])

    return {
        "forecast_date": forecast_date.date().isoformat(),
        "predicted_average_active_power_kw": float(prediction),
        "units": "kW",
        "model": model_name,
    }