# Decision log

Entries follow decision order, oldest first. Record major decisions, their
reasons, and key constraints; keep usage instructions in README and
implementation details in code.

## 2026-10-23 — Future actual generation in forecasting experiments

Use future actual generation as a proxy for forecasts and compare models with
and without it on matching chronological 168-hour windows. This assumes perfect
generation forecasts and may overstate operational performance.

## 2026-10-26 — Selected generation data sources

Use solar, onshore/offshore wind, cogeneration, biomass, coal, and natural gas.
Exclude `other` for missingness and waste/nuclear after visual inspection.
Keep the published CO₂ factor as target and preserve all raw data.

## 2026-10-26 — Data preparation and chronological split

Keep generation inputs aligned with the same-hour carbon-intensity target.
Fill offshore wind gaps with the last 24-hour profile and cogeneration gaps
with the mean of values 168/336 hours earlier. Replace both wind values at
2026-10-20 21:00 UTC with previous-hour values; record imputation flags.
Split by UTC: train before July 1, valid before September 1, then test (2026).
Select hyperparameters on validation, reserving test for final evaluation.

## 2026-10-26 — Target-only forecasting alignment

Use the current target, lags 1/2/3/24/48/168/336/504 hours, and
`y = target.shift(-168)`. Index by feature time and assign splits by label time.
At origin T, feature rows T−167h…T predict T+1h…T+168h without future targets
or recursive feedback. Exclude incomplete training rows and unknown evaluation
labels; fit only on labels already observed. SMA receives preceding target
history to compute seasonal averages.

## 2026-10-26 — Calendar features

Add weekend and Netherlands holiday flags, including Liberation Day every year.
Use Europe/Amsterdam dates for the predicted hour: row time for generation,
row time + 168 hours for target-only. These features are known in advance.

## 2026-10-26 — Initial target-only baselines and evaluation

Start with seasonal moving averages, linear regression, LightGBM, and XGBoost
using a shared row-wise prediction interface and explicit feature selection.
Initially fit once and score a fixed 168-hour lead, respecting the training-label
cutoff; this does not yet simulate weekly forecast issuance or refitting.
Treat these scores as baselines rather than tuned-model results.

## 2026-10-26 — Reproducible experiments and shared metrics

Define explicit parameter combinations in version-controlled YAML configs and
share training, alignment, and evaluation across models. Save configuration
snapshots, effective parameters, predictions, and metrics; keep generated
experiment results outside Git. Report MAE/RMSE in g CO2/kWh and R².

## 2026-10-26 — Separate validation and test experiments

Keep validation and test configs/results separate to make selection and final
evaluation auditable. Collect scores without averaging, preserving each split,
experiment, and run identity.

## 2026-10-27 — Weekly expanding-window evaluation

Replace fit-once row scoring with non-overlapping 168-hour forecasts and refit at
each weekly origin using all complete rows whose labels are already observed,
including earlier validation/test observations. Keep hyperparameters fixed;
anchor origins one hour before the split's first target and step by 168 UTC
hours. Exclude incomplete windows and retain SMA history. Record the protocol,
forecast origins, horizons, training sizes, and weekly scores to distinguish
these results from historical fit-once scores.

## 2026-10-27 — Target-only model selection

Select linear regression and boosting configurations by the lowest validation
MAE under weekly refitting, then freeze them for test evaluation. Validation
favours nonnegative linear regression without an intercept and shallow
absolute-error LightGBM/XGBoost models. Keep the feature set fixed and avoid
test-based tuning or early stopping.

## 2026-10-27 — Random forest and CatBoost comparison

Add random forest and CatBoost to compare bagged trees and another boosting
method under the same features and weekly protocol. Search depth, model size,
objective, and regularization on validation with fixed seeds. Reduce the costly
forest search and parallelize remaining runs while preserving completed results
and the actual configuration snapshots.

## 2026-10-27 — Frozen target-only forest and CatBoost models

Freeze each model's lowest-validation-MAE configuration for test: an
absolute-error forest with depth 8 and leaf size 20, and an MAE CatBoost model
with depth 2. Preserve all selected parameters and use the same weekly protocol
as the other target-only models.

## 2026-10-27 — Full-generation comparison

Evaluate all five regression models using the seven generation volumes and
calendar flags with a same-hour target. Use separate, compact validation grids
with a broader depth range because these predictors differ from target history.
Freeze each lowest-validation-MAE configuration before test evaluation, retaining
weekly refitting and the perfect-generation-forecast assumption.

## 2026-10-27 — Renewable-only comparison

Compare solar, onshore/offshore wind, and biomass with the full generation mix,
retaining calendar flags and the same weekly evaluation windows. Exclude
cogeneration, coal, and natural gas. Select each model's configuration by
validation MAE and freeze it for test; future actual renewable generation still
assumes perfect forecasts.

## 2026-10-27 — Report evidence and figure selection

Report all 16 frozen model test results across the three scenarios. Illustrate
September 2026 using the three lowest-full-test-MAE models per scenario; this
ranking is descriptive and does not select hyperparameters. Distinguish monthly
plot metrics from full-test scores, and include the data example and weekly
evaluation protocol for context.

## 2026-10-27 — Report scope and interpretation

State that the assignment's forecast-generation setting remains only partially
validated because experiments use future actual generation. Explain the EDA
motivation for target lags, source exclusions, and imputation heuristics, whose
effect was not separately evaluated. Interpret the near-perfect full-generation
score as learning a same-hour relationship with the realized production mix,
without claiming equivalent operational forecasting accuracy.
