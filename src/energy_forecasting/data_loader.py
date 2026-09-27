"""Read and validate the UCI minute-level household power data."""

from pathlib import Path

import pandas as pd

from energy_forecasting.config import (
    DATE_COLUMN,
    RAW_DATA_PATH,
    TIME_COLUMN,
    TIMESTAMP_COLUMN,
)


REQUIRED_COLUMNS = (DATE_COLUMN, TIME_COLUMN, "Global_active_power")


def parse_minute_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and numeric readings from a loaded UCI data table.

    Invalid dates become ``NaT`` and invalid numeric readings become ``NaN``;
    the preprocessing step reports and handles those rows explicitly.
    """
    if raw_data.empty:
        raise ValueError("The input dataset is empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw_data]
    if missing_columns:
        raise ValueError(
            "The input dataset is missing required column(s): "
            + ", ".join(missing_columns)
        )

    parsed = raw_data.copy()
    date_text = parsed[DATE_COLUMN].astype("string").str.strip()
    time_text = parsed[TIME_COLUMN].astype("string").str.strip()
    parsed[TIMESTAMP_COLUMN] = pd.to_datetime(
        date_text + " " + time_text,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    for column in parsed.columns:
        if column not in (DATE_COLUMN, TIME_COLUMN, TIMESTAMP_COLUMN):
            parsed[column] = pd.to_numeric(parsed[column], errors="coerce")

    return parsed


def load_minute_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the semicolon-separated source file and parse its values."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{source_path}'. Place "
            "household_power_consumption.txt in data/raw/ or pass its path."
        )

    try:
        raw_data = pd.read_csv(
            source_path,
            sep=";",
            na_values=["?"],
            low_memory=False,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Dataset file '{source_path}' is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"Could not parse dataset file '{source_path}': {error}") from error

    return parse_minute_data(raw_data)