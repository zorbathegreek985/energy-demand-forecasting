"""Shared project paths and dataset column names."""

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("ENERGY_FORECASTING_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "household_power_consumption.txt"
DAILY_DATA_PATH = PROCESSED_DATA_DIR / "daily_average_active_power.csv"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "daily_forecasting_features.csv"
MODEL_ARTIFACT_PATH = MODEL_DIR / "selected_model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "selected_model_metadata.json"
VALIDATION_METRICS_PATH = REPORT_DIR / "validation_metrics.csv"
MODEL_COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
TEST_METRICS_PATH = REPORT_DIR / "test_metrics.csv"
TEST_PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
SPLIT_METADATA_PATH = REPORT_DIR / "split_metadata.json"

DATE_COLUMN = "Date"
TIME_COLUMN = "Time"
TARGET_COLUMN = "Global_active_power"
TIMESTAMP_COLUMN = "timestamp"
DAILY_DATE_COLUMN = "date"
DAILY_TARGET_COLUMN = "daily_average_active_power_kw"