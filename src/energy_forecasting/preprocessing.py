"""Clean minute readings and aggregate valid active-power values by day."""

from pathlib import Path

import pandas as pd

from energy_forecasting.config import (
    DAILY_DATA_PATH,
    DAILY_TARGET_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from energy_forecasting.data_loader import load_minute_data


def data_quality_summary(minute_data: pd.DataFrame) -> dict[str, object]:
    """Summarize row counts, invalid timestamps, and missing source values."""
    if TIMESTAMP_COLUMN not in minute_data or TARGET_COLUMN not in minute_data:
        raise ValueError(
            f"Expected '{TIMESTAMP_COLUMN}' and '{TARGET_COLUMN}' columns."
        )

    return {
        "row_count": int(len(minute_data)),
        "invalid_timestamp_rows": int(minute_data[TIMESTAMP_COLUMN].isna().sum()),
        "missing_active_power_rows": int(minute_data[TARGET_COLUMN].isna().sum()),
        "valid_active_power_rows": int(minute_data[TARGET_COLUMN].notna().sum()),
        "missing_values_by_column": {
            str(column): int(count)
            for column, count in minute_data.isna().sum().items()
        },
    }


def clean_minute_data(minute_data: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and drop records that cannot be assigned a timestamp.

    Missing power values are deliberately retained here and excluded from the
    daily mean later. No readings are imputed or invented.
    """
    if TIMESTAMP_COLUMN not in minute_data or TARGET_COLUMN not in minute_data:
        raise ValueError(
            f"Expected '{TIMESTAMP_COLUMN}' and '{TARGET_COLUMN}' columns."
        )

    cleaned = minute_data.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    cleaned = cleaned.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No rows have a valid date and time.")
    return cleaned


def aggregate_daily_power(minute_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily mean active power in kW from valid minute readings."""
    cleaned = clean_minute_data(minute_data)
    daily = (
        cleaned.assign(date=cleaned[TIMESTAMP_COLUMN].dt.floor("D"))
        .groupby("date", sort=True)[TARGET_COLUMN]
        .agg(
            **{
                DAILY_TARGET_COLUMN: "mean",
                "valid_minute_readings": "count",
            }
        )
        .reset_index()
    )
    daily = daily.loc[daily["valid_minute_readings"] > 0].reset_index(drop=True)
    if daily.empty:
        raise ValueError("No valid active-power readings are available to aggregate.")
    return daily


def save_daily_data(
    daily_data: pd.DataFrame, path: str | Path = DAILY_DATA_PATH
) -> Path:
    """Save daily aggregates as CSV and return the destination path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    daily_data.to_csv(destination, index=False, date_format="%Y-%m-%d")
    return destination


def run_preprocessing(
    raw_path: str | Path,
    output_path: str | Path = DAILY_DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    """Load, summarize, aggregate, and save the daily dataset."""
    minute_data = load_minute_data(raw_path)
    quality = data_quality_summary(minute_data)
    daily_data = aggregate_daily_power(minute_data)
    output_file = save_daily_data(daily_data, output_path)
    return daily_data, quality, output_file