# Energy Demand Forecasting & Analytics

An end-to-end portfolio project that cleans household electricity readings,
explores real usage patterns, and compares models for forecasting the next
day's **average active power in kilowatts (kW)**. It demonstrates time-aware
feature engineering, chronological evaluation, reproducible Python scripts,
and automated tests. Daily average power is not daily energy in kWh.

## Dataset

Download the dataset manually from the
[UCI dataset page](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
Place the unchanged `household_power_consumption.txt` file in `data/raw/`.
The source is semicolon-separated, uses `?` for missing readings, and stores
dates as `dd/MM/yyyy`. The UCI page identifies the dataset license as Creative
Commons Attribution 4.0 International (CC BY 4.0). Follow its attribution
terms if sharing or adapting the dataset; cite the creators and UCI record
below. The raw file is excluded from Git and is not included in this
repository.

The pipeline converts invalid numeric readings to missing values and reports
them. Rows with invalid timestamps are reported and excluded because they
cannot be assigned to a day. Missing power readings are not imputed; the daily
mean uses only valid measurements and records their count. Days with no valid
power readings are omitted. Consequently, data gaps can affect daily coverage
and should be considered when interpreting later results.

## Setup (Windows PowerShell)

Run these commands from the project root in Windows PowerShell. They create
the project-local virtual environment and install the pinned requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Stage 1: preprocess data

After placing the downloaded file in `data/raw/`, run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

The script prints data-quality counts, writes
`data/processed/daily_average_active_power.csv` and
`data/processed/daily_forecasting_features.csv`, and saves six real-data EDA
figures in `reports/figures/`. Figures are generated from the dataset you
provide; the pipeline does not fabricate charts. Re-running this command
replaces the processed CSVs and Stage 2 EDA figures. You can override input
and output paths:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --raw-path "data/raw/household_power_consumption.txt" --output-path "data/processed/daily_average_active_power.csv" --feature-output-path "data/processed/daily_forecasting_features.csv"
```

The root path can also be changed with `ENERGY_FORECASTING_ROOT`; no machine-
specific absolute paths are stored in the code.

## Tests

With dependencies installed and the virtual environment active:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use small in-memory fixtures and temporary files; they do not need the
full dataset. The complete suite currently contains 33 tests.

## Project structure and data flow

- `src/energy_forecasting/config.py` defines relative project paths and shared
  dataset column names.
- `data_loader.py` reads the source format, parses timestamps, and converts
  invalid numeric values to missing values.
- `preprocessing.py` reports quality counts, orders valid timestamped records,
  computes daily mean power, and saves the processed CSV.
- `features.py` creates calendar-aligned daily lags, historical rolling means,
  and known calendar features. Missing dates remain missing rather than being
  forward-filled.
- `train.py` makes chronological partitions, fits the three ML estimators on
  training rows, selects by validation MAE, and saves the chosen artifact and
  date metadata.
- `evaluate.py` computes common-date metrics, applies the persistence baseline,
  and saves validation/test charts.
- `predict.py` loads the selected artifact and uses the shared feature builder
  to forecast the next calendar day.
- `visualization.py` saves daily, monthly, yearly, distribution, weekly, and
  raw missing-value plots from the supplied data.
- `scripts/run_pipeline.py` is the command-line entry point for preprocessing.
- `scripts/train_models.py` trains/evaluates from an existing feature CSV;
  `scripts/forecast_next_day.py` forecasts from the processed daily CSV.
- `tests/` verifies parsing, aggregation, feature alignment, chronological
  order, future-data leakage behavior, metrics, splits, model selection, saving,
  and prediction.
- `data/raw/`, `data/processed/`, `models/`, and `reports/figures/` hold local
  inputs or generated artifacts and are excluded from Git.

## Features and EDA

Each feature row predicts that date's daily average active power. `lag_1_day`,
`lag_7_day`, `lag_14_day`, and `lag_30_day` use the target from exactly that
many calendar days earlier. The 7- and 30-day rolling means include only the
complete immediately preceding calendar window; they exclude the target date.
The initial history, and windows containing missing daily observations, remain
missing instead of being filled. Day of week, month, and weekend status are
known from the forecast date in advance.

EDA charts summarize only available source observations. In particular, the
missing-value plot uses raw-data quality counts; a gap in daily measurements
is not imputed to make the time series look continuous.

## Stage 3: train and forecast

The target is `daily_average_active_power_kw`, the daily mean of minute-level
`Global_active_power` in kW. Each row forecasts that calendar date using only
historical lag/rolling values and known calendar fields. There is no target
imputation.

The dataset spans 1,442 calendar dates from 2006-12-16 through 2010-11-26.
Splits are made on those dates in order, before dropping rows with missing
targets or unavailable predictors:

| Partition | Calendar date boundaries | Eligible rows | Eligible date boundaries |
| --- | --- | ---: | --- |
| Train | 2006-12-16 to 2009-09-19 | 917 | 2007-01-15 to 2009-09-19 |
| Validation | 2009-09-20 to 2010-04-23 | 185 | 2009-09-20 to 2010-04-23 |
| Test | 2010-04-24 to 2010-11-26 | 151 | 2010-04-24 to 2010-11-26 |

The 70/15/15 proportions refer to calendar rows. Rows with a missing target or
any unavailable lag/rolling predictor are excluded after boundaries are fixed.
The reported missing-target and unavailable-feature counts can overlap. The
first 30 days lack full history; missing calendar observations can also make
lag/rolling inputs unavailable. Those dates are not fabricated or scored.

Candidates are naive persistence, linear regression, random forest, and
histogram gradient boosting. The models use fixed settings; random forest and
histogram gradient boosting use seed 42. Linear regression needs no fitted
scaler or imputer because incomplete rows are excluded.

For a forecast date, persistence uses the most recent nonmissing target with
a date strictly earlier than the forecast date. If yesterday is missing, it
uses the most recent older value already observed at that origin; it never
uses a later observation. Candidate metrics use the same dates where all four
approaches can be scored. Validation MAE selects the model. Only that selected
model is scored on test. The model is fitted on training rows only and is not
refit on validation or test. Test results are not used for selection.

Evaluation is rolling one-day-ahead: at each later date, actual targets from
earlier dates, including earlier test-period days, are past information and
may enter that date's lag/rolling predictors. Models are not refit during the
test period. This is not a fixed-origin, multi-day forecast.

Run training after preprocessing/features exist:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts/train_models.py
```

Training overwrites the selected model and Stage 3 reports/figures. It is not
needed to run tests or generate a forecast from the existing saved model.

Forecast one day after the latest processed history:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts/forecast_next_day.py
```

The commands can accept `--feature-path` or `--history-path` to use alternate
CSV files. Model files are stored under `models/` and ignored by Git. Reports
are written under `reports/`: `validation_metrics.csv`,
`model_comparison.csv`, `test_metrics.csv`, `test_predictions.csv`, and
`split_metadata.json`. Test/validation plots are saved under `reports/figures/`.
Joblib artifacts use pickle internally; load only artifacts you trust.

### Verified real-data results

These values were produced from the local UCI dataset and processed feature
CSV; they are not synthetic. Validation comparisons share 185 dates.
MAE and RMSE are in kW; R-squared has no units.

| Model | Validation MAE (kW) | Validation RMSE (kW) | Validation R² | Scored dates |
| --- | ---: | ---: | ---: | ---: |
| Naive persistence | 0.269795 | 0.350407 | -0.547948 | 185 |
| Linear regression | 0.193284 | 0.262159 | 0.133558 | 185 |
| Random forest (selected) | 0.191176 | 0.254693 | 0.182204 | 185 |
| Histogram gradient boosting | 0.207628 | 0.274556 | 0.049677 | 185 |

The random forest had the lowest validation MAE by a small margin over linear
regression. Its held-out test result, evaluated once after selection, was MAE
0.155183 kW, RMSE 0.199144 kW, R² 0.571117 over 151 dates. This is one
chronological split of historical household data and does not establish future
performance or production suitability. A saved-model smoke test produced
1.4374 kW for 2010-11-27, the day after this dataset's final date; it is not a
current-date demand forecast.

The persistence comparison is a rolling one-step baseline and can use an
older observed value when the previous calendar day's target is missing. The
included scoring rows all have a valid prior target; unavailable baseline
dates are excluded from the common validation comparison.

## Later stage (not implemented yet)

Stage 4 will add the Streamlit dashboard. It will use the saved processed data,
metrics, and model; it will not retrain on dashboard interactions.

The forecasting task is one-day-ahead prediction of daily average power in kW.
Results depend on data coverage, split dates, preprocessing, feature
availability, and modeling assumptions. Missing observations and this
household's historical usage limit generalization; historical metrics do not
guarantee future demand.

## Troubleshooting

- **`python` or `py` is not recognized:** install Python 3.12 and enable the
  Python Launcher/PATH option, then reopen PowerShell.
- **Activation is blocked:** use the process-scoped execution-policy command
  in the setup section; it does not change the machine-wide policy.
- **`ModuleNotFoundError: energy_forecasting`:** from the root, set
  `$env:PYTHONPATH = "src"` before running the script or tests.
- **Dataset not found:** verify the exact filename and location under
  `data/raw/`; the pipeline does not download it.
- **Malformed input:** verify that the original UCI file is intact and
  semicolon-separated. The original should not be edited.

## Attribution

Hebrail, G. and Berard, A. (2006). Individual Household Electric Power
Consumption [Dataset]. UCI Machine Learning Repository.
[DOI: 10.24432/C58K54](https://doi.org/10.24432/C58K54). The UCI record lists
the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See the [dataset record](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
for license details and attribution requirements.

## Project Report

See [reports/project_report.md](reports/project_report.md) for the problem
definition, experimental design, verified results, interpretation, and
limitations.

   the mean rather than replaced with fabricated measurements.
3. **How was leakage prevented?** Features use only observations
   available at each forecast origin, and data will be split chronologically.
4. **Why compare against persistence?** A simple last-value forecast provides
   a meaningful benchmark that complex models must beat on held-out data.
5. **What do MAE and RMSE measure?** Both measure prediction error in kW; MAE
  averages absolute error, while RMSE penalizes larger errors more strongly.
