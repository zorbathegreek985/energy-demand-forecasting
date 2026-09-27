# Energy Demand Forecasting & Analytics

## Executive Summary

This project builds a reproducible pipeline for analyzing the UCI Individual
Household Electric Power Consumption data and forecasting the next day's daily
average active power in kW. It includes raw-data validation, daily
aggregation, exploratory plots, calendar-aligned features, chronological
model selection, a held-out test evaluation, and saved-model inference.

Random Forest was selected by validation MAE (0.191176 kW over 185 dates).
On the separate 151-date test period, it recorded MAE 0.155183 kW, RMSE
0.199144 kW, and R-squared 0.571117. These results describe one household and
one chronological split; they do not establish deployment readiness or
future performance.

## Problem Definition

The target is `Global_active_power`, measured in kW at one-minute intervals.
The measurements are aggregated into each observed calendar day's average
active power, also in kW. The prediction target for date `t` is that day's
average, forecast using data available before date `t`. This is a power
forecast, not a daily energy estimate in kWh.

## Dataset and Usage

The source is the [UCI Individual Household Electric Power Consumption
dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
collected in Sceaux, France, from December 2006 through November 2010. It
contains 2,075,259 minute-level rows and nine fields. The UCI page lists the
dataset under Creative Commons Attribution 4.0 International (CC BY 4.0).
Attribute the creators and UCI record when sharing or adapting the data. The
unchanged raw file is stored locally in `data/raw/` and excluded from Git.

Citation: Hebrail, G. and Berard, A. (2006). *Individual Household Electric
Power Consumption* [Dataset]. UCI Machine Learning Repository. DOI:
[10.24432/C58K54](https://doi.org/10.24432/C58K54).

## Preprocessing and EDA

The loader parses the source's semicolon-separated records, combines the
`dd/MM/yyyy` date and time, converts `?` and invalid numeric values to missing
values, and reports missingness. Invalid timestamps are excluded because they
cannot be assigned to a date. Missing active-power readings are not imputed;
daily averages use valid readings only.

The inspected source contained 2,075,259 rows, including 25,979 missing
active-power readings and no invalid timestamps. Aggregation produced 1,433
observed daily rows from 2006-12-16 through 2010-11-26. The calendar spans
1,442 dates; nine dates had no valid daily target and remain missing in the
feature table.

The pipeline saves daily history, monthly and yearly patterns, a distribution,
day-of-week averages, and a raw missing-value overview. These charts are
generated from the local dataset, without filling gaps to create observations:

- [Daily average active power](figures/daily_average_active_power.png)
- [Monthly pattern](figures/monthly_average_active_power.png)
- [Yearly pattern](figures/yearly_average_active_power.png)
- [Daily-value distribution](figures/daily_average_power_distribution.png)
- [Day-of-week pattern](figures/weekly_average_active_power.png)
- [Raw-data missingness](figures/missing_data_overview.png)

## Features and Leakage Prevention

Predictors are `lag_1_day`, `lag_7_day`, `lag_14_day`, `lag_30_day`,
`rolling_mean_7_day`, `rolling_mean_30_day`, `day_of_week`, `month`, and
`is_weekend`. Lags refer to calendar days. Each rolling mean uses the complete
immediately preceding window and excludes the target date. Calendar features
are known for the forecast date. Missing historical targets are not filled;
lag or rolling features that cannot be calculated stay missing.

Training eligibility requires both a target and all feature values. Splits are
made on calendar dates before incomplete rows are removed. Thus the initial
30-day history and rows affected by gaps reduce eligible samples without
moving dates between partitions.

## Experimental Design

The 1,442 calendar dates are divided chronologically, approximately 70/15/15.
Model candidates are naive persistence, Linear Regression, Random Forest, and
HistGradientBoostingRegressor. Model settings are fixed; estimators are fit on
training rows only. Validation MAE chooses the model. Test data is scored only
after selection, and the selected model is not refit on validation or test.

| Partition | Calendar boundaries | Eligible rows | Eligible boundaries |
| --- | --- | ---: | --- |
| Train | 2006-12-16 to 2009-09-19 | 917 | 2007-01-15 to 2009-09-19 |
| Validation | 2009-09-20 to 2010-04-23 | 185 | 2009-09-20 to 2010-04-23 |
| Test | 2010-04-24 to 2010-11-26 | 151 | 2010-04-24 to 2010-11-26 |

Persistence predicts from the most recent nonmissing target strictly before
each forecast date. If the previous calendar day's target is missing, it may
use an older observation already available at that origin, never a future
value. Validation candidates share a common set of 185 score dates. The test
metric is for the selected model only.

Evaluation is rolling one-day-ahead: as each test date advances, earlier
observed targets, including earlier test-period targets, are available to its
past-only predictors. The model is not refit during this period. This is not a
fixed-origin multi-day forecast.

## Validation Comparison

All candidates were scored on the same 185 validation dates. MAE and RMSE are
in kW; R-squared is unitless.

| Model | Validation MAE (kW) | Validation RMSE (kW) | Validation R² |
| --- | ---: | ---: | ---: |
| Naive persistence | 0.269795 | 0.350407 | -0.547948 |
| Linear Regression | 0.193284 | 0.262159 | 0.133558 |
| Random Forest (selected) | 0.191176 | 0.254693 | 0.182204 |
| HistGradientBoostingRegressor | 0.207628 | 0.274556 | 0.049677 |

Random Forest had the lowest validation MAE, narrowly ahead of Linear
Regression. The validation comparison plot is
[here](figures/validation_model_comparison.png).

## Held-Out Test Results

The selected Random Forest was evaluated on 151 test dates, 2010-04-24 through
2010-11-26:

| Metric | Test result |
| --- | ---: |
| MAE | 0.155183 kW |
| RMSE | 0.199144 kW |
| R-squared | 0.571117 |

The actual-versus-predicted and residual figures are
[here](figures/test_actual_vs_predicted.png) and
[here](figures/test_residual_analysis.png). These are held-out results for one
historical split. A different period, data coverage, or forecasting setup may
produce different scores; the results are not a claim of general accuracy.

The saved model's smoke-test forecast is 1.4374 kW for 2010-11-27, the day
after the available dataset ends. It is not a forecast for the current date.

## Limitations and Future Improvements

- The data represents one household and ends in 2010; performance may not
  generalize to other households or present-day demand.
- Missing targets and unavailable rolling history remove dates from model
  fitting and evaluation. Missing values are not imputed.
- Rolling one-day-ahead evaluation uses earlier realized targets as they become
  available; it does not measure a multi-day forecast made once at a fixed
  origin.
- The split is a single chronological holdout, and the validation advantage
  over Linear Regression is small. Repeated rolling-origin evaluation and
  uncertainty intervals could provide a more robust assessment.
- The project is a portfolio/research implementation, not a deployed or
  production-validated forecasting service.

Possible future work includes additional rolling-origin splits, uncertainty
estimation, evaluating more households, and the separately planned Streamlit
dashboard. These are not part of the current implementation.