"""Create calendar-aligned predictors using only historical target values."""

from pathlib import Path

import pandas as pd

from energy_forecasting.config import (
    DAILY_DATE_COLUMN,
    DAILY_TARGET_COLUMN,
    FEATURE_DATA_PATH,
)


LAG_DAYS = (1, 7, 14, 30)
ROLLING_WINDOWS = (7, 30)
FEATURE_COLUMNS = (
    "lag_1_day",
    "lag_7_day",
    "lag_14_day",
    "lag_30_day",
    "rolling_mean_7_day",
    "rolling_mean_30_day",
    "day_of_week",
    "month",
    "is_weekend",
)


def create_features(daily_data: pd.DataFrame) -> pd.DataFrame:
    """Build one row per calendar day with leakage-safe one-day-ahead inputs.

    For the target on date ``t``, every lag and rolling mean uses observations
    strictly before ``t``. Calendar features describe ``t`` and are known at
    the forecast origin (the end of date ``t - 1``). Missing calendar dates
    are inserted with a missing target, not an imputed value, so a lag always
    means a calendar-day lag rather than a previous-observed-row lag.
    """
    required_columns = (DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN)
    missing_columns = [column for column in required_columns if column not in daily_data]
    if missing_columns:
        raise ValueError(
            "Daily data is missing required column(s): " + ", ".join(missing_columns)
        )
    if daily_data.empty:
        raise ValueError("Daily data is empty.")

    dates = pd.to_datetime(daily_data[DAILY_DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError("Daily data contains invalid dates.")
    dates = dates.dt.normalize()
    if dates.duplicated().any():
        raise ValueError("Daily data must contain at most one row per calendar date.")

    raw_target = daily_data[DAILY_TARGET_COLUMN]
    target_values = pd.to_numeric(raw_target, errors="coerce")
    invalid_values = raw_target.notna() & target_values.isna()
    if invalid_values.any():
        raise ValueError("Daily active-power values must be numeric or missing.")
    if target_values.notna().sum() == 0:
        raise ValueError("Daily data has no valid active-power observations.")

    ordered = pd.DataFrame(
        {DAILY_DATE_COLUMN: dates.to_numpy(), DAILY_TARGET_COLUMN: target_values.to_numpy()}
    ).sort_values(DAILY_DATE_COLUMN, kind="stable")
    calendar = pd.date_range(
        start=ordered[DAILY_DATE_COLUMN].iloc[0],
        end=ordered[DAILY_DATE_COLUMN].iloc[-1],
        freq="D",
        name=DAILY_DATE_COLUMN,
    )
    target = ordered.set_index(DAILY_DATE_COLUMN)[DAILY_TARGET_COLUMN].reindex(calendar)

    features = pd.DataFrame(index=calendar)
    for lag in LAG_DAYS:
        features[f"lag_{lag}_day"] = target.shift(lag)

    historical_values = target.shift(1)
    for window in ROLLING_WINDOWS:
        features[f"rolling_mean_{window}_day"] = historical_values.rolling(
            window=window,
            min_periods=window,
        ).mean()

    features["day_of_week"] = calendar.dayofweek
    features["month"] = calendar.month
    features["is_weekend"] = (calendar.dayofweek >= 5).astype("int8")

    result = pd.concat([target.rename(DAILY_TARGET_COLUMN), features], axis=1)
    return result.rename_axis(DAILY_DATE_COLUMN).reset_index()


def save_feature_data(
    feature_data: pd.DataFrame,
    path: str | Path = FEATURE_DATA_PATH,
) -> Path:
    """Save the calendar-aligned feature table as CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    feature_data.to_csv(destination, index=False, date_format="%Y-%m-%d")
    return destination