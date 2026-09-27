import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from energy_forecasting.config import DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN
from energy_forecasting.features import FEATURE_COLUMNS, create_features
from energy_forecasting.predict import forecast_next_day, load_selected_model
from energy_forecasting.train import save_model_artifacts


def save_test_model(tmp_path, model_name, model, feature_columns=FEATURE_COLUMNS):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    metadata = {
        "selected_model": model_name,
        "feature_columns": list(feature_columns),
    }
    save_model_artifacts(model_path, metadata_path, model_name, model, metadata)
    return model_path, metadata_path


def test_model_save_and_load_round_trip(tmp_path) -> None:
    model = LinearRegression().fit([[0], [1], [2]], [1.0, 2.0, 3.0])
    model_path, metadata_path = save_test_model(tmp_path, "linear_regression", model, ("value",))

    bundle, metadata = load_selected_model(model_path, metadata_path)

    assert metadata["selected_model"] == "linear_regression"
    assert bundle["model"].predict([[3]])[0] == pytest.approx(4.0)


def test_next_day_forecast_returns_date_value_and_units(tmp_path) -> None:
    daily = pd.DataFrame(
        {
            DAILY_DATE_COLUMN: pd.date_range("2023-01-01", periods=45, freq="D"),
            DAILY_TARGET_COLUMN: [2.0 + (index % 7) * 0.2 for index in range(45)],
        }
    )
    features = create_features(daily).dropna(subset=FEATURE_COLUMNS)
    model = LinearRegression().fit(features.loc[:, FEATURE_COLUMNS], features[DAILY_TARGET_COLUMN])
    model_path, metadata_path = save_test_model(tmp_path, "linear_regression", model)

    forecast = forecast_next_day(daily, model_path, metadata_path)

    assert forecast["forecast_date"] == "2023-02-15"
    assert np.isfinite(forecast["predicted_average_active_power_kw"])
    assert forecast["units"] == "kW"


def test_persistence_forecast_uses_latest_observation_before_forecast(tmp_path) -> None:
    history = pd.DataFrame(
        {
            DAILY_DATE_COLUMN: pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            DAILY_TARGET_COLUMN: [1.0, 2.0, np.nan],
        }
    )
    model_path, metadata_path = save_test_model(
        tmp_path, "naive_persistence", None, FEATURE_COLUMNS
    )

    forecast = forecast_next_day(history, model_path, metadata_path)

    assert forecast["forecast_date"] == "2023-01-04"
    assert forecast["predicted_average_active_power_kw"] == 2.0


def test_insufficient_history_and_missing_model_are_reported(tmp_path) -> None:
    model_path, metadata_path = save_test_model(
        tmp_path, "linear_regression", LinearRegression().fit([[0], [1]], [1, 2])
    )
    short_history = pd.DataFrame(
        {
            DAILY_DATE_COLUMN: pd.date_range("2023-01-01", periods=10, freq="D"),
            DAILY_TARGET_COLUMN: np.arange(10, dtype=float),
        }
    )

    with pytest.raises(ValueError, match="predictors are unavailable"):
        forecast_next_day(short_history, model_path, metadata_path)
    with pytest.raises(FileNotFoundError, match="Train models first"):
        forecast_next_day(short_history, tmp_path / "absent.joblib", metadata_path)


def test_missing_required_history_column_is_rejected(tmp_path) -> None:
    model_path, metadata_path = save_test_model(
        tmp_path, "naive_persistence", None, FEATURE_COLUMNS
    )

    with pytest.raises(ValueError, match="missing required"):
        forecast_next_day(pd.DataFrame({DAILY_DATE_COLUMN: ["2023-01-01"]}), model_path, metadata_path)