import numpy as np
import pandas as pd
import pytest

from energy_forecasting.evaluate import calculate_metrics, persistence_predictions


def test_metrics_match_known_example() -> None:
    metrics = calculate_metrics([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])

    assert metrics["mae_kw"] == pytest.approx(2 / 3)
    assert metrics["rmse_kw"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["r2"] == pytest.approx(0.0)
    assert metrics["n_observations"] == 3


def test_r_squared_is_missing_when_not_meaningful() -> None:
    assert calculate_metrics([1.0], [2.0])["r2"] is None
    assert calculate_metrics([1.0, 1.0], [1.0, 2.0])["r2"] is None


def test_persistence_uses_latest_prior_observation_and_never_future_values() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
            "daily_average_active_power_kw": [1.0, 2.0, np.nan, 99.0],
        }
    )
    forecast_dates = pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-05"])

    predictions = persistence_predictions(history, forecast_dates)

    assert predictions.tolist() == [2.0, 2.0, 99.0]


def test_persistence_reports_no_prediction_without_prior_target() -> None:
    history = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01"]), "daily_average_active_power_kw": [1.0]}
    )

    predictions = persistence_predictions(history, pd.to_datetime(["2020-01-01"]))

    assert np.isnan(predictions.iloc[0])


def test_metric_input_shape_must_match() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        calculate_metrics([1.0, 2.0], [1.0])