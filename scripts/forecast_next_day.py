"""Generate a next-day forecast from the processed daily history."""

import argparse

import pandas as pd

from energy_forecasting.config import DAILY_DATA_PATH
from energy_forecasting.predict import forecast_next_day


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-path", default=DAILY_DATA_PATH, help="Daily aggregate CSV")
    arguments = parser.parse_args()
    try:
        history = pd.read_csv(arguments.history_path, parse_dates=["date"])
        forecast = forecast_next_day(history)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Forecast for {forecast['forecast_date']}: "
        f"{forecast['predicted_average_active_power_kw']:.4f} kW "
        f"({forecast['model']})"
    )


if __name__ == "__main__":
    main()