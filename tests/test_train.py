import numpy as np
import pandas as pd
import pytest

from energy_forecasting.config import DAILY_DATE_COLUMN, DAILY_TARGET_COLUMN
from energy_forecasting.features import FEATURE_COLUMNS, create_features
import energy_forecasting.train as training
from energy_forecasting.train import (
    build_candidate_models,
    chronological_split,
    fit_candidate_models,
    select_model,
    train_models,
)


def sample_features(periods: int = 120) -> pd.DataFrame:
    daily = pd.DataFrame(
        {
            DAILY_DATE_COLUMN: pd.date_range("2020-01-01", periods=periods, freq="D"),
            DAILY_TARGET_COLUMN: [3.0 + (index % 13) * 0.1 for index in range(periods)],
        }
    )
    return create_features(daily)


def test_chronological_split_boundaries_are_date_based_and_disjoint() -> None:
    features = sample_features(100).sample(frac=1, random_state=8)

    splits = chronological_split(features)

    assert splits.metadata["total_calendar_rows"] == 100
    assert splits.metadata["splits"]["train"]["calendar_start_date"] == "2020-01-01"
    assert splits.metadata["splits"]["train"]["calendar_end_date"] == "2020-03-10"
    assert splits.metadata["splits"]["validation"]["calendar_start_date"] == "2020-03-11"
    assert splits.metadata["splits"]["validation"]["calendar_end_date"] == "2020-03-25"
    assert splits.metadata["splits"]["test"]["calendar_start_date"] == "2020-03-26"
    assert splits.train[DAILY_DATE_COLUMN].max() < splits.validation[DAILY_DATE_COLUMN].min()
    assert splits.validation[DAILY_DATE_COLUMN].max() < splits.test[DAILY_DATE_COLUMN].min()
    assert splits.train[DAILY_DATE_COLUMN].is_monotonic_increasing
    assert splits.validation[DAILY_DATE_COLUMN].is_monotonic_increasing
    assert splits.test[DAILY_DATE_COLUMN].is_monotonic_increasing


def test_missing_targets_and_features_are_excluded_after_boundaries() -> None:
    features = sample_features(100)
    missing_target_date = pd.Timestamp("2020-03-15")
    missing_feature_date = pd.Timestamp("2020-04-05")
    features.loc[features[DAILY_DATE_COLUMN] == missing_target_date, DAILY_TARGET_COLUMN] = np.nan
    features.loc[features[DAILY_DATE_COLUMN] == missing_feature_date, "lag_7_day"] = np.nan

    splits = chronological_split(features)

    assert missing_target_date not in set(splits.validation[DAILY_DATE_COLUMN])
    assert missing_feature_date not in set(splits.test[DAILY_DATE_COLUMN])
    assert splits.metadata["splits"]["validation"]["missing_target_rows"] == 1
    assert splits.metadata["splits"]["test"]["rows_with_unavailable_features"] == 1


def test_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="sum to less than 1"):
        chronological_split(sample_features(), train_fraction=0.8, validation_fraction=0.2)


def test_model_selection_uses_validation_mae_only() -> None:
    validation_metrics = pd.DataFrame(
        {
            "model": ["linear_regression", "random_forest"],
            "mae_kw": [0.4, 0.7],
            "test_mae_kw": [99.0, 0.01],
        }
    )

    assert select_model(validation_metrics) == "linear_regression"


def test_fit_helper_receives_only_the_training_partition() -> None:
    class RecordingModel:
        def fit(self, inputs, target):
            self.fit_inputs = inputs.copy()
            self.fit_target = target.copy()
            return self

    splits = chronological_split(sample_features())
    model = RecordingModel()

    fit_candidate_models({"recording": model}, splits.train)

    assert len(model.fit_target) == len(splits.train)
    pd.testing.assert_frame_equal(model.fit_inputs, splits.train.loc[:, FEATURE_COLUMNS])
    assert model.fit_target.index.equals(splits.train.index)


def test_random_forest_configuration_is_reproducible() -> None:
    features = sample_features(80)
    split = chronological_split(features)
    first = build_candidate_models(random_seed=17)["random_forest"]
    second = build_candidate_models(random_seed=17)["random_forest"]

    assert first.get_params()["random_state"] == second.get_params()["random_state"] == 17
    fit_candidate_models({"first": first}, split.train)
    fit_candidate_models({"second": second}, split.train)
    np.testing.assert_allclose(
        first.predict(split.test.loc[:, FEATURE_COLUMNS]),
        second.predict(split.test.loc[:, FEATURE_COLUMNS]),
        rtol=1e-14,
        atol=1e-14,
    )


def test_training_orchestrator_fits_only_training_rows(monkeypatch, tmp_path) -> None:
    class RecordingRegressor:
        def fit(self, inputs, target):
            self.fit_inputs = inputs.copy()
            self.fit_target = target.copy()
            self.mean_target = float(target.mean())
            return self

        def predict(self, inputs):
            return np.full(len(inputs), self.mean_target)

        def get_params(self, deep=False):
            return {}

    feature_data = sample_features(100)
    expected_training = chronological_split(feature_data).train
    estimator = RecordingRegressor()
    monkeypatch.setattr(training, "build_candidate_models", lambda: {"recording": estimator})

    train_models(
        feature_data,
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        figure_dir=tmp_path / "figures",
    )

    assert len(estimator.fit_target) == len(expected_training)
    pd.testing.assert_frame_equal(
        estimator.fit_inputs,
        expected_training.loc[:, FEATURE_COLUMNS],
    )
    assert estimator.fit_target.equals(expected_training[DAILY_TARGET_COLUMN])