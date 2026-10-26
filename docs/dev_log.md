# Decision log

## 2026-10-23 — Future actual generation in forecasting experiments

Using future actual generation assumes perfect generation forecasts and may
overstate forecasting performance. We will compare models with and without it
on the same chronological 168-hour evaluation windows.

## 2026-10-26 — Selected generation data sources

Use `solar`, `onshore wind`, `offshore wind`, `cogeneration`, `biomass`, `coal`, and `natural gas`. Exclude `other` due number of missing values. Exclude `waste` and `nuclear generation` following visual inspection of their time series. Keep the published CO₂ factor as the target and retain all raw data.

## 2026-10-26 — First processed dataset

Run `uv run python scripts/1_prepare_dataset_v1.py` to create
`data/processed/dataset_v1.csv` from `data/raw/dataset.csv`, retaining all rows
and seven selected generation volumes in kWh. Leave raw data unchanged.

- Fill offshore wind gaps by repeating the last known 24-hour profile;
  fill cogeneration gaps with the mean of values 168 and 336 hours earlier.
- Replace both wind volumes only at 2026-10-20 21:00 UTC with previous-hour
  values. Record all replacements in source flags and their union, `is_imputed`.
- Keep `emissionfactor (kg CO2/kWh)` aligned with same-hour generation,
  without shifting or duplicating it as `y`.
- Set `split` by UTC time: `train` before 2026-07-01, `valid` until
  2026-09-01, then `test`. Exclude `split` from model inputs.

Select parameters on validation and evaluate on test in successive 168-hour
windows. Supply generation forecasts for each window (future actuals assume
perfect forecasts); target history must end before the window starts.

## 2026-10-26 — Target-only dataset

Run `uv run python scripts/1_prepare_dataset_target_only_v1.py` to create
`data/processed/dataset_target_only_v1.csv` from raw target observations.
Retain the original target and add hourly lags 1, 2, 3, 24, 48, 168, 336,
and 504, with `y = target.shift(-168)`. Keep all rows, including missing
initial lags and the final 168 unknown labels.

Index rows by feature time; assign `split` using label time (index + 168 hours)
and the existing July/September boundaries. Exclude incomplete rows from
regression training and exclude `y` and `split` from features. A block of
168 feature rows ending at the forecast origin predicts the next 168 hours.
Use the original chronological target history for seasonal moving average.

## 2026-10-26 — Calendar features

Add boolean `is_holiday` (official Netherlands holidays, including Liberation
Day every year) and `is_weekend` (Saturday or Sunday) to both datasets using
the `holidays` package. Calculate dates in `Europe/Amsterdam` for the predicted
hour: row time in the generation dataset and row time + 168 hours in the
target-only dataset, including rows with unknown `y`. These calendar features
are known in advance; neither flag implies that every workplace is closed.

## 2026-10-26 — Row-wise target-only SMA experiment

Use the common model interface `fit(X, y)` / `predict(X)`, with each model
listing its input `feature_columns`. Predictions retain the feature-time index;
compare each prediction directly with the same row's `y`, 168 hours ahead.
SMA reads only the original target column and computes its own historical shifts.
For `season_length=168` and `n_seasons=2`, predict with the mean of the current
target and its value 168 hours earlier. One season repeats the current target.
The season length controls historical spacing, not the dataset's forecast lead.
This supersedes the earlier SMA interface that returned a future hourly path.

Run one fixed-parameter validation experiment with:

```bash
uv run python scripts/2_run_experiment.py --season-length 168 --n-seasons 2
```

The script saves `results/sma/predictions.csv` (feature time, target time, actual
and predicted values in kg CO2/kWh) and `metrics.json` (parameters and MAE/RMSE
in g CO2/kWh). Use `--output-dir` to retain separate runs and `--split test`
for final evaluation after selecting parameters on validation.

Fit once on complete train rows whose label times are no later than the first
evaluated feature time. This additional cutoff prevents regression training
from using labels unavailable at that time, since splits follow label time.
Pass preceding feature history to prediction, exclude unknown labels from
scoring, and require predictions for every otherwise eligible evaluation row.
SMA requires all requested seasonal observations; warmup predictions are missing.
This experiment evaluates forecasts issued every hour at a fixed 168-hour lead.
It does not evaluate all 168 lead hours from a single weekly forecast origin.

## 2026-10-26 — Target-only regression models and selectable features

Keep using `data/processed/dataset_target_only_v1.csv`. Add `RegressionModel`
with scikit-learn linear regression, and subclasses `LGBMRegressionModel` and
`XGBMRegressionModel` using LightGBM and XGBoost. All three share feature
selection, fitting, and indexed predictions. Each input row predicts its own
`y`, 168 hours ahead; the experiment's training-label cutoff remains in place.

Supply `feature_columns` when constructing a regression model. The CLI defaults
to all 11 target-only features: current target, eight target lags, and the two
calendar flags. Use `--features` to choose a subset. Models enforce that order
at both fit and predict and reject empty, duplicate, or leaking (`y`, `split`)
feature selections. Internally, selected columns get neutral names to avoid
library restrictions on feature names; output timestamps remain unchanged.

Install Python dependencies with `uv sync`. LightGBM also needs the OpenMP
runtime; on Ubuntu/Debian install it with `sudo apt-get install libgomp1`.
The boosting libraries are imported only when their model is instantiated,
so this runtime requirement does not prevent SMA or linear regression use.

Each command runs one experiment on validation by default:

```bash
uv run python scripts/2_run_experiment.py --model regression
uv run python scripts/2_run_experiment.py --model lgbm
uv run python scripts/2_run_experiment.py --model xgbm

uv run python scripts/2_run_experiment.py --model regression \
  --features 'emissionfactor (kg CO2/kWh)' target_lag_168 is_weekend \
  --output-dir results/regression_subset

uv run python scripts/2_run_experiment.py --model lgbm \
  --model-params '{"n_estimators": 200, "learning_rate": 0.05}' \
  --output-dir results/lgbm_200
```

Outputs default to `results/<model>/`. Metrics record the selected columns and
explicit estimator parameters (including wrapper defaults); library defaults
are fixed by `uv.lock`. Both boosting wrappers default to `random_state=42`
and `n_jobs=1`. These runs use fixed parameters without tuning or early stopping.
No scaling or feature engineering is added.

Initial validation results with all 11 features, 21,050 eligible training rows,
and the same 1,488 validation labels:

| Model | MAE, g CO2/kWh | RMSE, g CO2/kWh |
|---|---:|---:|
| Linear regression | 57.255 | 70.711 |
| LightGBM | 61.524 | 77.680 |
| XGBoost | 64.736 | 83.215 |

The earlier SMA run (168-hour season, two seasons) uses those same validation
labels and scores MAE 54.986 / RMSE 75.494. These are initial validation results,
not evidence that the default parameters or feature set are optimal. Test data
have not been used for model selection.

API references: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html),
[LGBMRegressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html),
and [XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor).
