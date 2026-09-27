"""Generate labeled exploratory plots from real daily and quality data."""

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from energy_forecasting.config import (
    DAILY_DATE_COLUMN,
    DAILY_TARGET_COLUMN,
    FIGURE_DIR,
)


def _validated_daily_data(daily_data: pd.DataFrame) -> pd.DataFrame:
    required_columns = (DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN)
    missing_columns = [column for column in required_columns if column not in daily_data]
    if missing_columns:
        raise ValueError(
            "Daily data is missing required column(s): " + ", ".join(missing_columns)
        )
    if daily_data.empty:
        raise ValueError("Daily data is empty; no EDA plots can be generated.")

    result = daily_data.loc[:, required_columns].copy()
    result[DAILY_DATE_COLUMN] = pd.to_datetime(
        result[DAILY_DATE_COLUMN], errors="coerce"
    )
    result[DAILY_TARGET_COLUMN] = pd.to_numeric(
        result[DAILY_TARGET_COLUMN], errors="coerce"
    )
    if result[DAILY_DATE_COLUMN].isna().any():
        raise ValueError("Daily data contains invalid dates.")
    if result[DAILY_TARGET_COLUMN].notna().sum() == 0:
        raise ValueError("Daily data has no valid active-power values to plot.")
    return result.sort_values(DAILY_DATE_COLUMN, kind="stable")


def _save_figure(figure: plt.Figure, output_dir: str | Path, filename: str) -> Path:
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_daily_average(daily_data: pd.DataFrame, output_dir: str | Path = FIGURE_DIR) -> Path:
    """Plot daily average active power over time."""
    data = _validated_daily_data(daily_data)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(data[DAILY_DATE_COLUMN], data[DAILY_TARGET_COLUMN], linewidth=0.8)
    axis.set(title="Daily Average Active Power", xlabel="Date", ylabel="Average active power (kW)")
    axis.grid(True, alpha=0.25)
    return _save_figure(figure, output_dir, "daily_average_active_power.png")


def plot_monthly_pattern(
    daily_data: pd.DataFrame, output_dir: str | Path = FIGURE_DIR
) -> Path:
    """Plot mean daily average power for each calendar month across years."""
    data = _validated_daily_data(daily_data)
    month_means = data.groupby(data[DAILY_DATE_COLUMN].dt.month)[DAILY_TARGET_COLUMN].mean()
    month_means = month_means.reindex(range(1, 13))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(range(1, 13), month_means, marker="o")
    axis.set_xticks(range(1, 13), [calendar.month_abbr[month] for month in range(1, 13)])
    axis.set(
        title="Mean Daily Average Active Power by Month",
        xlabel="Calendar month",
        ylabel="Mean daily average active power (kW)",
    )
    axis.grid(True, axis="y", alpha=0.25)
    return _save_figure(figure, output_dir, "monthly_average_active_power.png")


def plot_yearly_pattern(
    daily_data: pd.DataFrame, output_dir: str | Path = FIGURE_DIR
) -> Path:
    """Plot mean daily average power for each year represented in the data."""
    data = _validated_daily_data(daily_data)
    yearly_means = data.groupby(data[DAILY_DATE_COLUMN].dt.year)[DAILY_TARGET_COLUMN].mean()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(yearly_means.index.astype(str), yearly_means.values)
    axis.set(
        title="Mean Daily Average Active Power by Year",
        xlabel="Year",
        ylabel="Mean daily average active power (kW)",
    )
    axis.grid(True, axis="y", alpha=0.25)
    return _save_figure(figure, output_dir, "yearly_average_active_power.png")


def plot_power_distribution(
    daily_data: pd.DataFrame, output_dir: str | Path = FIGURE_DIR
) -> Path:
    """Plot the distribution of observed daily average active power."""
    data = _validated_daily_data(daily_data)
    values = data[DAILY_TARGET_COLUMN].dropna()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(values, bins=30, edgecolor="white")
    axis.set(
        title="Distribution of Daily Average Active Power",
        xlabel="Daily average active power (kW)",
        ylabel="Number of days",
    )
    axis.grid(True, axis="y", alpha=0.25)
    return _save_figure(figure, output_dir, "daily_average_power_distribution.png")


def plot_weekly_pattern(
    daily_data: pd.DataFrame, output_dir: str | Path = FIGURE_DIR
) -> Path:
    """Plot mean daily average power by day of the week."""
    data = _validated_daily_data(daily_data)
    weekdays = data.groupby(data[DAILY_DATE_COLUMN].dt.dayofweek)[DAILY_TARGET_COLUMN].mean()
    weekdays = weekdays.reindex(range(7))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(range(7), weekdays.values)
    axis.set_xticks(range(7), [calendar.day_name[day] for day in range(7)])
    axis.set(
        title="Mean Daily Average Active Power by Day of Week",
        xlabel="Day of week",
        ylabel="Mean daily average active power (kW)",
    )
    axis.grid(True, axis="y", alpha=0.25)
    return _save_figure(figure, output_dir, "weekly_average_active_power.png")


def plot_missing_data_overview(
    quality_summary: dict[str, object], output_dir: str | Path = FIGURE_DIR
) -> Path:
    """Plot missing source-value counts from the raw-data quality summary."""
    missing_counts = quality_summary.get("missing_values_by_column")
    if not isinstance(missing_counts, dict) or not missing_counts:
        raise ValueError("Quality summary must include missing_values_by_column.")

    labels = [str(column) for column in missing_counts]
    counts = [int(missing_counts[column]) for column in missing_counts]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels[::-1], counts[::-1])
    axis.set(
        title="Missing Values in Minute-Level Source Data",
        xlabel="Missing values (rows)",
        ylabel="Source field",
    )
    axis.grid(True, axis="x", alpha=0.25)
    return _save_figure(figure, output_dir, "missing_data_overview.png")


def generate_eda_plots(
    daily_data: pd.DataFrame,
    quality_summary: dict[str, object],
    output_dir: str | Path = FIGURE_DIR,
) -> dict[str, Path]:
    """Save the complete set of EDA figures and return their paths."""
    return {
        "daily_average": plot_daily_average(daily_data, output_dir),
        "monthly_pattern": plot_monthly_pattern(daily_data, output_dir),
        "yearly_pattern": plot_yearly_pattern(daily_data, output_dir),
        "distribution": plot_power_distribution(daily_data, output_dir),
        "weekly_pattern": plot_weekly_pattern(daily_data, output_dir),
        "missing_data": plot_missing_data_overview(quality_summary, output_dir),
    }