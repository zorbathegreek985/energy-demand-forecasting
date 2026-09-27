"""Preprocess the UCI data and generate Stage 2 features and EDA figures."""

import argparse

from energy_forecasting.config import DAILY_DATA_PATH, FEATURE_DATA_PATH, RAW_DATA_PATH
from energy_forecasting.features import create_features, save_feature_data
from energy_forecasting.preprocessing import run_preprocessing
from energy_forecasting.visualization import generate_eda_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default=RAW_DATA_PATH, help="Path to the UCI text file")
    parser.add_argument("--output-path", default=DAILY_DATA_PATH, help="Daily CSV output path")
    parser.add_argument(
        "--feature-output-path",
        default=FEATURE_DATA_PATH,
        help="Calendar-aligned feature CSV output path",
    )
    arguments = parser.parse_args()

    try:
        daily_data, quality, output_path = run_preprocessing(
            arguments.raw_path, arguments.output_path
        )
        feature_data = create_features(daily_data)
        feature_output_path = save_feature_data(feature_data, arguments.feature_output_path)
        figure_paths = generate_eda_plots(daily_data, quality)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print("Data quality summary:")
    for key, value in quality.items():
        print(f"  {key}: {value}")
    print(f"Daily rows saved: {len(daily_data)}")
    print(f"Output: {output_path}")
    print(f"Calendar-aligned feature rows saved: {len(feature_data)}")
    print(f"Feature output: {feature_output_path}")
    print("EDA figures saved:")
    for figure_path in figure_paths.values():
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()