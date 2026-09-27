"""Train, select, and evaluate forecasting models from processed features."""

import argparse

from energy_forecasting.train import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/processed/daily_forecasting_features.csv",
        help="Existing calendar-aligned feature CSV",
    )
    arguments = parser.parse_args()
    try:
        results = run_training_pipeline(arguments.feature_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Selected using validation MAE: {results['selected_model']}")
    print("Validation metrics:")
    print(results["validation_metrics"].to_string(index=False))
    print("Selected model held-out test metrics:")
    print(results["test_metrics"].to_string(index=False))
    print(f"Model: {results['model_path']}")
    print(f"Reports: {results['report_dir']}")


if __name__ == "__main__":
    main()