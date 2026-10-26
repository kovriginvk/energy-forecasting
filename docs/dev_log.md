# Decision log

Entries follow decision order, oldest first. Append new decisions at the end;
keep usage instructions in README and implementation details in code.

## 2026-10-23 — Future actual generation in forecasting experiments

Future actual generation assumes perfect forecasts and may overstate performance.
Compare models with and without it on matching chronological 168-hour windows.

## 2026-10-26 — Selected generation data sources

Use solar, onshore/offshore wind, cogeneration, biomass, coal, and natural gas.
Exclude `other` for missingness and waste/nuclear after visual inspection.
Keep the published CO₂ factor as target and preserve all raw data.

## 2026-10-26 — First processed dataset

Create `dataset_v1.csv` with seven generation volumes (kWh) and the same-hour
`emissionfactor (kg CO2/kWh)` target, without shifting it or adding `y`.
Fill offshore wind gaps with the last 24-hour profile and cogeneration gaps
with the mean of values 168/336 hours earlier. Replace both wind values at
2026-10-20 21:00 UTC with previous-hour values; record imputation flags.
Split by UTC: train before July 1, valid before September 1, then test (2026).
Select on valid, evaluate on test; exclude `split` from features. Weekly
forecasts require generation forecasts and target history ending before the window.

## 2026-10-26 — Target-only dataset

Create `dataset_target_only_v1.csv` with the original target, lags
1/2/3/24/48/168/336/504 hours, and `y = target.shift(-168)`. Keep all rows;
exclude incomplete training rows and unknown labels from evaluation.
Index by feature time and assign splits by label time (index + 168 hours).
Exclude `y` and `split` from inputs. At origin T, feature rows T−167h…T
predict T+1h…T+168h without future targets or recursive feedback.
SMA uses the original target history, including history preceding this block.

## 2026-10-26 — Calendar features

Add `is_holiday` (official Netherlands holidays, including Liberation Day every
year, via `holidays`) and `is_weekend`. Use Europe/Amsterdam dates for the
predicted hour: row time for generation, row time + 168 hours for target-only.
These flags are known in advance, including for rows with unknown labels.

## 2026-10-26 — Row-wise target-only SMA experiment

Models expose `fit(X, y)`, `predict(X)`, and `feature_columns`; predictions keep
the feature-time index and align with each row's `y`. SMA averages the current
target and earlier values spaced by `season_length`; one season repeats the
current target. Season length controls history spacing, not forecast lead.
This replaces the earlier SMA interface returning a future hourly path.
Fit once using complete train rows with labels available by the first evaluated
feature time; pass preceding history and require predictions for every eligible
row. This scores a fixed 168-hour lead, without simulating weekly issuance/refits.
Initial SMA (season 168h, two seasons): valid MAE 54.986, RMSE 75.494 g CO2/kWh.

## 2026-10-26 — Target-only regression models and selectable features

Add linear regression, LightGBM, and XGBoost with explicit ordered feature
selection; reject empty, duplicate, or leaking (`y`, `split`) inputs. Use neutral
internal column names for library compatibility. Default to current target,
eight lags, and two calendar flags; preserve the training-label cutoff.
Boosting defaults: `random_state=42`, `n_jobs=1`; no tuning, early stopping,
scaling, or extra features. Lock dependencies; import boosting libraries lazily.
Initial valid scores on 1,488 labels (21,050 training rows), MAE/RMSE in g CO2/kWh:
linear 57.255/70.711, LightGBM 61.524/77.680, XGBoost 64.736/83.215.
These baseline results do not establish optimal parameters; test was not used
for selection.

## 2026-10-26 — Configured experiments and shared evaluation

Use `experiments/run.py` with one YAML config per dataset/model pair. Execute
only explicit `runs[].params` combinations overriding optional `base_params`.
Save the config and per-run summary in `<config_stem>/`; numbered runs contain
predictions, effective parameters, and metrics. Reruns replace prior results;
ignore generated results in Git. Report MAE/RMSE in g CO2/kWh and R², without
period-level metrics. Starter configs cover SMA and baseline linear regression.
Share training/alignment in `src/experiment.py` and metrics in `src/evaluator.py`;
the existing single-run CLI delegates to them. Configure target/offset as
`y`/168h for target-only or `emissionfactor (kg CO2/kWh)`/0h for generation.
Preserve fit-once evaluation and its label cutoff; future actual generation
still assumes perfect forecasts.

## 2026-10-26 — Combined experiment metrics

Use `experiments/collect_metrics.py` to rebuild `experiments/results/metrics.csv`
from experiment summaries, adding the config name as `experiment`. Preserve
individual run scores and metadata without averaging or training. Replace the
file on collection to avoid duplicates; refresh after rerunning experiments.

## 2026-10-26 — Separate validation and test experiments

Store configs in `experiments/configs/<split>/` and results in
`experiments/results/<split>/<config_stem>/` (`valid` or `test`). Require the
config directory to match its YAML split; reruns replace only that split/config.
Move existing experiments to valid. The collector reads both splits, preserving
split/experiment/run identity and checking directory/split consistency.
