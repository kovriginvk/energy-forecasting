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
