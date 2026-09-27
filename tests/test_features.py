import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from energy_forecasting.config import DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN
from energy_forecasting.features import FEATURE_COLUMNS, create_features


def sample_daily_data(periods: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            DAILY_DATE_COLUMN: pd.date_range("2024-01-01", periods=periods, freq="D"),
            DAILY_TARGET_COLUMN: [float(value) for value in range(1, periods + 1)],
        }
    )


def test_features_are_chronological_and_lags_align_to_calendar_days() -> None:
    daily_data = sample_daily_data().sample(frac=1, random_state=17)

    features = create_features(daily_data)

    assert features[DAILY_DATE_COLUMN].is_monotonic_increasing
    target_day = features.loc[
        features[DAILY_DATE_COLUMN] == pd.Timestamp("2024-01-31")
    ].iloc[0]
    assert target_day["lag_1_day"] == 30.0
    assert target_day["lag_7_day"] == 24.0
    assert target_day["lag_14_day"] == 17.0
    assert target_day["lag_30_day"] == 1.0


def test_rolling_means_exclude_the_target_day() -> None:
    features = create_features(sample_daily_data())
    target_day = features.loc[
        features[DAILY_DATE_COLUMN] == pd.Timestamp("2024-01-08")
    ].iloc[0]

    assert target_day[DAILY_TARGET_COLUMN] == 8.0
    assert target_day["rolling_mean_7_day"] == 4.0
    assert pd.isna(features.iloc[6]["rolling_mean_7_day"])
    assert pd.isna(features.iloc[29]["rolling_mean_30_day"])
    assert features.iloc[30]["rolling_mean_30_day"] == 15.5


def test_calendar_features_describe_the_target_date() -> None:
    features = create_features(sample_daily_data())
    saturday = features.loc[
        features[DAILY_DATE_COLUMN] == pd.Timestamp("2024-01-06")
    ].iloc[0]

    assert saturday["day_of_week"] == 5
    assert saturday["month"] == 1
    assert saturday["is_weekend"] == 1


def test_missing_calendar_day_is_not_treated_as_previous_observation() -> None:
    daily_data = sample_daily_data().drop(index=9)

    features = create_features(daily_data)

    assert len(features) == 40
    missing_day = features.loc[
        features[DAILY_DATE_COLUMN] == pd.Timestamp("2024-01-10")
    ].iloc[0]
    following_day = features.loc[
        features[DAILY_DATE_COLUMN] == pd.Timestamp("2024-01-11")
    ].iloc[0]
    assert pd.isna(missing_day[DAILY_TARGET_COLUMN])
    assert pd.isna(following_day["lag_1_day"])
    assert following_day["lag_7_day"] == 4.0
    assert pd.isna(following_day["rolling_mean_7_day"])


def test_future_target_changes_do_not_change_features_at_or_before_origin() -> None:
    daily_data = sample_daily_data()
    changed_future = daily_data.copy()
    changed_future.loc[25:, DAILY_TARGET_COLUMN] += 10000

    original_features = create_features(daily_data).set_index(DAILY_DATE_COLUMN)
    changed_features = create_features(changed_future).set_index(DAILY_DATE_COLUMN)
    forecast_origin = pd.Timestamp("2024-01-25")

    assert_frame_equal(
        original_features.loc[:forecast_origin, FEATURE_COLUMNS],
        changed_features.loc[:forecast_origin, FEATURE_COLUMNS],
    )


def test_duplicate_dates_and_invalid_input_are_rejected() -> None:
    duplicate_dates = sample_daily_data(2)
    duplicate_dates.loc[1, DAILY_DATE_COLUMN] = duplicate_dates.loc[0, DAILY_DATE_COLUMN]

    with pytest.raises(ValueError, match="at most one row"):
        create_features(duplicate_dates)

    with pytest.raises(ValueError, match="empty"):
        create_features(pd.DataFrame(columns=[DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN]))