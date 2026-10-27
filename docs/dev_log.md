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

## 2026-10-27 — Weekly expanding-window evaluation

Replace fit-once row scoring with non-overlapping 168-hour forecasts and refit at
each weekly origin using all complete rows whose labels are already observed,
including earlier validation/test observations. Keep hyperparameters fixed;
anchor origins one hour before the split's first target and step by 168 UTC
hours. Exclude incomplete windows and the partial final week, retaining SMA
history. Save origin, horizon, training sizes, per-week scores, and the protocol
name so the new results can be distinguished from historical fit-once scores.

## 2026-10-27 — Linear regression validation combinations

Compare all four combinations of `fit_intercept` and `positive` on the existing
target-only features, keeping the unconstrained model with an intercept first
as the baseline. Use validation to assess the constraints before test selection.

## 2026-10-27 — Selected linear regression for test evaluation

Select `fit_intercept=False` and `positive=True` by the lowest validation MAE
(49.530 g CO2/kWh across eight weekly origins), retaining the existing feature
list. Freeze this single combination in the test config and evaluate with the
same weekly expanding-window protocol.

## 2026-10-27 — Target-only boosting validation configs

Add LightGBM and XGBoost validation configs with the same features as linear
regression and eight explicit combinations each. Compare shallow-to-moderate
trees, paired learning rates/tree counts, stronger leaf/L2 regularization, and
row/feature sampling. Fix the seed and use one worker; enable LightGBM sampling
with `subsample_freq=1`. Keep squared-error objectives and select by validation
MAE under weekly refitting, without early stopping or test-based tuning.

## 2026-10-27 — Save XGBoost's missing-value default

Record NaN estimator defaults as the string `"NaN"` in saved parameter metadata
so XGBoost's default `missing` sentinel does not break strict JSON output. Keep
the in-memory estimator parameters unchanged and retain finite metric checks.

## 2026-10-27 — Simpler boosting models and absolute-error objectives

Append eight validation combinations per boosting model after depth two performed
best in the initial search: depth-one trees with 100/200/400 estimators, depth-two
trees with 50/100/400 estimators, and absolute-error objectives at depths one/two
with 200 estimators. Keep the learning rate at 0.05 and other parameters at their
initial best-run values to isolate tree count, depth, and objective effects;
preserve the first eight combinations and the weekly evaluation protocol.

## 2026-10-27 — Focus boosting configs on absolute-error training

Replace unsuccessful search branches with 13 combinations per boosting config:
retain the best squared-error reference and both evaluated absolute-error models;
cover depths one/two with 100/200/400/800 estimators at learning rate 0.05, then
vary leaf size/child weight and L2 separately around the current absolute-error
winner. Existing central values are included once. The regularization runs use
the current winner at 200 trees, before any new tree-count results are available.
Saved results retain their original config snapshots until the next launch.

## 2026-10-27 — Selected boosting models for test evaluation

Freeze each boosting model's lowest-validation-MAE combination in a single-run
test config: LightGBM run 005 (51.926 g CO2/kWh), absolute error, depth one, two
leaves, 400 trees; XGBoost run 006 (52.978 g CO2/kWh), absolute error, depth one,
800 trees. Both retain learning rate 0.05 and the remaining selected parameters;
evaluate with the same weekly expanding-window protocol as RM and SMA.

## 2026-10-27 — Random forest and CatBoost validation models

Add random forest and CatBoost through the shared row-wise regression interface,
with CatBoost as a locked dependency. Keep the existing target-only features and
weekly refitting protocol. Define 12 validation combinations per model: random
forest uses deeper trees (4/8/12/unlimited), leaf-size and feature-sampling
regularization, tree counts, and squared/absolute error; CatBoost compares MAE
at depths 1/2/4, iteration counts, L2, and an RMSE reference. Fix seeds and one
worker, disable CatBoost training files, and defer test configs until validation
selection.

## 2026-10-27 — Broader first-stage forest and CatBoost search

Expand validation to 42 random forest and 66 CatBoost combinations before any
results are available, to avoid centering the initial search on untested defaults.
Cross depth, leaf size, and objective for forests; cross depth, iteration count,
and objective for CatBoost. Add selected sampling, regularization, tree-count,
and learning-rate variants while avoiding a full cross-product of all controls.
Keep weekly evaluation unchanged and use validation results to narrow the next
iteration; the complete search entails 864 fits over eight origins.

## 2026-10-27 — Reduce and parallelize the remaining forest search

After 15 squared-error combinations completed, stop the one-worker forest run
and reduce the remaining 27 combinations to 14. Retain six absolute-error
variants with larger leaves, four feature-sampling variants, two half-size
bootstrap samples, and two 100-tree variants; drop the costly one-observation
MAE leaves and 600-tree variants. Preserve completed results and their explicit
one-worker parameters; run the remaining combinations with four workers. Keep
the original configuration snapshot beside the reduced 29-run result config.

## 2026-10-27 — Selected forest and CatBoost models for test evaluation

Freeze the lowest-validation-MAE configurations in single-run test configs:
random forest run 017 (54.065 g CO2/kWh), absolute error, depth 8, 300 trees,
leaf size 20, four workers; CatBoost run 031 (51.201 g CO2/kWh), MAE, depth 2,
1200 iterations, learning rate 0.01, L2 3, one worker. Retain all selected
parameters and features, using the same weekly expanding-window test protocol
as the previously evaluated models.

## 2026-10-27 — Canonical forest configuration

Keep the version-controlled forest validation config as one 29-combination
search with four workers for every combination, removing per-run worker
overrides left by continuation. Preserve the actual mixed-worker configuration
in the ignored result snapshot; move the superseded 42-run grid outside the
results directory. Existing scores and per-run parameter records remain intact.

## 2026-10-27 — Generation dataset target name

Rename the processed generation dataset's carbon-intensity target to `y` for
consistent experiment configuration. Preserve the raw source column name,
same-hour alignment (offset zero), target values, and all other dataset columns.
Regenerate `dataset_v1.csv` with the updated preparation script.

## 2026-10-27 — Compact generation validation grids

Add separate validation configs for the five regression models using all seven
generation volumes and two calendar flags, with `y` at offset zero under the
perfect-generation-forecast assumption. Compare four linear regression settings,
12 combinations per boosting model (objective, depth, tree count), and six forest
combinations (objective, depth, leaf size). Use a broader depth range than the
target-only winners because generation provides different predictors; limit
absolute-error forests to larger leaves for runtime. Keep weekly refitting and
fixed seeds, defer training and test selection, and preserve target-only configs.

## 2026-10-27 — Selected generation models for test evaluation

Freeze the lowest-validation-MAE runs from the saved generation experiment
configs: linear regression run 001 (23.760 g CO2/kWh), with an intercept and
unconstrained coefficients; XGBoost run 006 (5.561); LightGBM run 006 (5.446).
Both boosting models use squared error, depth six, 600 trees, and learning rate
0.05. Preserve all selected parameters and features in single-run test configs
and evaluate with weekly expanding-window refitting under the same assumption
of perfect future generation forecasts. Test labels enter training only after
their observation times; test scores do not determine hyperparameters.

## 2026-10-27 — Selected generation forest and CatBoost for test evaluation

Freeze the lowest-validation-MAE configurations from saved generation runs:
random forest run 003 (8.403 g CO2/kWh), squared error, 200 trees, unlimited
depth, leaf size five; CatBoost run 012 (4.450), MAE, depth six, 600 iterations,
learning rate 0.05, L2 three. Preserve selected features and remaining parameters
in single-run test configs. Evaluate with the existing weekly expanding-window
protocol and perfect-generation-forecast assumption, without test-based tuning.
