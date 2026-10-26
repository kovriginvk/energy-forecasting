# Decision log

## 2026-10-23 — Future actual generation in forecasting experiments

Using future actual generation assumes perfect generation forecasts and may
overstate forecasting performance. We will compare models with and without it
on the same chronological 168-hour evaluation windows.

## 2026-10-26 — Selected generation data sources

Use `solar`, `onshore wind`, `offshore wind`, `cogeneration`, `biomass`, `coal`, and `natural gas`. Exclude `other` due number of missing values. Exclude `waste` and `nuclear generation` following visual inspection of their time series. Keep the published CO₂ factor as the target and retain all raw data.
