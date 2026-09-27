"""Metrics, scoring policies, and plots for one-day-ahead forecasts."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_forecasting.config import DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN, FIGURE_DIR


PERSISTENCE_MODEL_NAME = "naive_persistence"


def calculate_metrics(actual: Any, predicted: Any) -> dict[str, float | int | None]:
    """Calculate MAE, RMSE, and meaningful R-squared in kW."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have matching shapes.")
    if actual_values.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")

    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[valid]
    predicted_values = predicted_values[valid]
    if not len(actual_values):
        raise ValueError("No finite actual/predicted pairs are available to score.")

    errors = predicted_values - actual_values
    total_sum_squares = float(np.sum((actual_values - actual_values.mean()) ** 2))
    r_squared = None
    if len(actual_values) >= 2 and total_sum_squares > 0:
        r_squared = float(1.0 - np.sum(errors**2) / total_sum_squares)

    return {
        "mae_kw": float(np.mean(np.abs(errors))),
        "rmse_kw": float(np.sqrt(np.mean(errors**2))),
        "r2": r_squared,
        "n_observations": int(len(actual_values)),
    }


def persistence_predictions(
    history: pd.DataFrame,
    forecast_dates: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
) -> pd.Series:
    """Use the latest nonmissing target strictly before each forecast date.

    If the immediately preceding calendar day's target is missing, an older
    observed value is used only if it was already available at that origin.
    No target on or after the forecast date can be selected.
    """
    required = (DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN)
    missing = [column for column in required if column not in history]
    if missing:
        raise ValueError("History is missing required column(s): " + ", ".join(missing))

    history_dates = pd.to_datetime(history[DAILY_DATE_COLUMN], errors="coerce")
    if history_dates.isna().any():
        raise ValueError("History contains invalid dates.")
    history_dates = history_dates.dt.normalize()
    target = pd.to_numeric(history[DAILY_TARGET_COLUMN], errors="coerce")
    invalid_targets = history[DAILY_TARGET_COLUMN].notna() & target.isna()
    if invalid_targets.any():
        raise ValueError("History targets must be numeric or missing.")

    valid_history = pd.DataFrame(
        {DAILY_DATE_COLUMN: history_dates, DAILY_TARGET_COLUMN: target}
    ).dropna(subset=[DAILY_TARGET_COLUMN])
    valid_history = valid_history.sort_values(DAILY_DATE_COLUMN, kind="stable")

    dates = pd.DatetimeIndex(pd.to_datetime(forecast_dates, errors="coerce"))
    if dates.isna().any():
        raise ValueError("Forecast dates contain invalid values.")
    dates = dates.normalize()
    predictions: list[float] = []
    for forecast_date in dates:
        available = valid_history.loc[
            valid_history[DAILY_DATE_COLUMN] < forecast_date, DAILY_TARGET_COLUMN
        ]
        predictions.append(float(available.iloc[-1]) if not available.empty else np.nan)

    return pd.Series(predictions, index=dates, name="prediction_kw", dtype=float)


def _metric_row(
    model_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    metrics = calculate_metrics(actual, predicted)
    metrics.update(
        {
            "model": model_name,
            "scored_start_date": dates.min().date().isoformat(),
            "scored_end_date": dates.max().date().isoformat(),
        }
    )
    return metrics


def evaluate_candidates(
    models: dict[str, Any],
    history: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all candidates on one common set of eligible dates."""
    dates = pd.DatetimeIndex(evaluation_data[DAILY_DATE_COLUMN])
    actual = evaluation_data[DAILY_TARGET_COLUMN].to_numpy(dtype=float)
    raw_predictions: dict[str, np.ndarray] = {
        PERSISTENCE_MODEL_NAME: persistence_predictions(history, dates).to_numpy()
    }
    for model_name, model in models.items():
        raw_predictions[model_name] = np.asarray(
            model.predict(evaluation_data.loc[:, feature_columns]), dtype=float
        )

    common = np.isfinite(actual)
    for prediction in raw_predictions.values():
        common &= np.isfinite(prediction)
    if not common.any():
        raise ValueError("No common eligible dates are available for model comparison.")

    scored_dates = dates[common]
    scored_actual = actual[common]
    rows = [
        _metric_row(name, scored_actual, values[common], scored_dates)
        for name, values in raw_predictions.items()
    ]
    predictions = pd.DataFrame(
        {
            DAILY_DATE_COLUMN: scored_dates,
            "actual_kw": scored_actual,
            **{f"{name}_predicted_kw": values[common] for name, values in raw_predictions.items()},
        }
    )
    return pd.DataFrame(rows), predictions


def evaluate_selected_model(
    model_name: str,
    model: Any,
    history: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate only the already-selected model on its held-out period."""
    dates = pd.DatetimeIndex(evaluation_data[DAILY_DATE_COLUMN])
    if model_name == PERSISTENCE_MODEL_NAME:
        predictions = persistence_predictions(history, dates).to_numpy()
    else:
        predictions = np.asarray(
            model.predict(evaluation_data.loc[:, feature_columns]), dtype=float
        )
    actual = evaluation_data[DAILY_TARGET_COLUMN].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predictions)
    if not valid.any():
        raise ValueError(f"No eligible test dates can be scored for '{model_name}'.")

    scored_dates = dates[valid]
    scored_actual = actual[valid]
    scored_predictions = predictions[valid]
    metrics = _metric_row(model_name, scored_actual, scored_predictions, scored_dates)
    result = pd.DataFrame(
        {
            DAILY_DATE_COLUMN: scored_dates,
            "actual_kw": scored_actual,
            "predicted_kw": scored_predictions,
            "residual_kw": scored_predictions - scored_actual,
        }
    )
    return metrics, result


def _save_figure(figure: plt.Figure, output_dir: str | Path, filename: str) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_validation_comparison(
    metrics: pd.DataFrame,
    selected_model: str,
    output_dir: str | Path = FIGURE_DIR,
) -> Path:
    """Plot validation MAE for candidate models; selection is validation-only."""
    labels = [
        f"{name}{' *' if name == selected_model else ''}"
        for name in metrics["model"]
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, metrics["mae_kw"], label="Validation MAE")
    axis.set(
        title="Candidate Model Comparison (Validation Period)",
        xlabel="Model (* selected using validation MAE)",
        ylabel="Mean absolute error (kW)",
    )
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    return _save_figure(figure, output_dir, "validation_model_comparison.png")


def plot_test_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    output_dir: str | Path = FIGURE_DIR,
) -> Path:
    """Plot actual and selected-model predictions over the held-out test dates."""
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions[DAILY_DATE_COLUMN], predictions["actual_kw"], label="Actual", linewidth=1)
    axis.plot(
        predictions[DAILY_DATE_COLUMN],
        predictions["predicted_kw"],
        label=f"Predicted ({model_name})",
        linewidth=1,
    )
    axis.set(
        title="Held-Out Test: Actual vs Predicted Daily Average Active Power",
        xlabel="Date",
        ylabel="Daily average active power (kW)",
    )
    axis.legend()
    axis.grid(True, alpha=0.25)
    return _save_figure(figure, output_dir, "test_actual_vs_predicted.png")


def plot_residual_analysis(
    predictions: pd.DataFrame,
    model_name: str,
    output_dir: str | Path = FIGURE_DIR,
) -> Path:
    """Plot test residuals over time and their distribution."""
    figure, (time_axis, distribution_axis) = plt.subplots(2, 1, figsize=(11, 8))
    time_axis.plot(
        predictions[DAILY_DATE_COLUMN],
        predictions["residual_kw"],
        label="Prediction residual",
    )
    time_axis.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Zero error")
    time_axis.set(
        title=f"Held-Out Test Residual Analysis ({model_name})",
        xlabel="Date",
        ylabel="Residual (predicted - actual, kW)",
    )
    time_axis.legend()
    time_axis.grid(True, alpha=0.25)

    distribution_axis.hist(predictions["residual_kw"], bins=25, label="Test residuals")
    distribution_axis.set(
        title="Test Residual Distribution",
        xlabel="Residual (predicted - actual, kW)",
        ylabel="Number of days",
    )
    distribution_axis.legend()
    distribution_axis.grid(True, axis="y", alpha=0.25)
    return _save_figure(figure, output_dir, "test_residual_analysis.png")