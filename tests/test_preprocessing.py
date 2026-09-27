import pandas as pd
import pytest

from energy_forecasting.data_loader import load_minute_data, parse_minute_data
from energy_forecasting.preprocessing import (
    aggregate_daily_power,
    clean_minute_data,
    data_quality_summary,
    run_preprocessing,
    save_daily_data,
)


def sample_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["02/01/2007", "01/01/2007", "01/01/2007", "bad-date"],
            "Time": ["00:01:00", "00:01:00", "00:00:00", "00:02:00"],
            "Global_active_power": ["2.0", "?", "1.0", "3.0"],
            "Voltage": ["240.0", "241.0", "239.0", "238.0"],
        }
    )


def test_parse_dates_and_convert_missing_or_invalid_values() -> None:
    parsed = parse_minute_data(sample_raw_data())

    assert parsed.loc[0, "timestamp"] == pd.Timestamp("2007-01-02 00:01:00")
    assert pd.isna(parsed.loc[1, "Global_active_power"])
    assert pd.isna(parsed.loc[3, "timestamp"])


def test_cleaning_sorts_rows_and_drops_invalid_timestamps() -> None:
    cleaned = clean_minute_data(parse_minute_data(sample_raw_data()))

    assert cleaned["timestamp"].is_monotonic_increasing
    assert len(cleaned) == 3
    assert cleaned["Global_active_power"].isna().sum() == 1


def test_daily_aggregation_uses_valid_values_and_counts_them() -> None:
    daily = aggregate_daily_power(parse_minute_data(sample_raw_data()))

    assert daily["date"].tolist() == [pd.Timestamp("2007-01-01"), pd.Timestamp("2007-01-02")]
    assert daily["daily_average_active_power_kw"].tolist() == [1.0, 2.0]
    assert daily["valid_minute_readings"].tolist() == [1, 1]


def test_quality_summary_reports_invalid_timestamp_and_missing_target() -> None:
    parsed = parse_minute_data(sample_raw_data())

    summary = data_quality_summary(parsed)

    assert summary["row_count"] == 4
    assert summary["invalid_timestamp_rows"] == 1
    assert summary["missing_active_power_rows"] == 1


def test_missing_file_has_actionable_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="household_power_consumption.txt"):
        load_minute_data(tmp_path / "missing.txt")


def test_semicolon_file_runs_through_preprocessing(tmp_path) -> None:
    source = tmp_path / "household_power_consumption.txt"
    source.write_text(
        "Date;Time;Global_active_power;Voltage\n"
        "01/01/2007;00:00:00;1.5;240.0\n"
        "01/01/2007;00:01:00;?;?\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed" / "daily.csv"

    daily, quality, output_path = run_preprocessing(source, output)

    assert quality["row_count"] == 2
    assert quality["missing_active_power_rows"] == 1
    assert daily.loc[0, "daily_average_active_power_kw"] == 1.5
    assert output_path.is_file()


def test_empty_dataset_file_has_actionable_error(tmp_path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_minute_data(source)


def test_empty_or_malformed_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_minute_data(pd.DataFrame())

    with pytest.raises(ValueError, match="missing required column"):
        parse_minute_data(pd.DataFrame({"Date": ["01/01/2007"]}))


def test_all_invalid_timestamps_are_rejected() -> None:
    invalid_data = pd.DataFrame(
        {
            "timestamp": [pd.NaT],
            "Global_active_power": [1.0],
        }
    )

    with pytest.raises(ValueError, match="No rows have a valid"):
        clean_minute_data(invalid_data)


def test_saving_creates_parent_directory_and_csv(tmp_path) -> None:
    daily = aggregate_daily_power(parse_minute_data(sample_raw_data()))
    destination = save_daily_data(daily, tmp_path / "nested" / "daily.csv")

    loaded = pd.read_csv(destination)
    assert len(loaded) == 2
    assert "daily_average_active_power_kw" in loaded