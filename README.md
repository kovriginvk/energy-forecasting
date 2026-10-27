# Energy forecasting

Forecast hourly electricity carbon intensity in the Netherlands. Project
requirements are in `docs/initial_assignment.md`; modeling decisions are in
`docs/dev_log.md`.

## Setup

Install the Python environment with `uv sync`. LightGBM also requires the
system OpenMP runtime (`libgomp1` on Ubuntu/Debian).

## Experiments

Run the starter configurations from the project root:

```bash
uv run python experiments/run.py experiments/configs/valid/target_only_v1_sma.yaml
uv run python experiments/run.py experiments/configs/valid/target_only_v1_rm.yaml
uv run python experiments/run.py experiments/configs/valid/target_only_v1_rf.yaml
uv run python experiments/run.py experiments/configs/valid/target_only_v1_catboost.yaml
```

Each YAML config describes one dataset/model pair. Set `dataset` (relative to
the project root, or absolute), `model` (`sma`, `regression`, `lgbm`, `xgbm`,
`random_forest`, `catboost`),
`target`, `target_offset_hours`, and `split` (`valid` or `test`). Store configs in
`experiments/configs/<split>/`; the directory must match the YAML `split`.
For final evaluation, copy the selected config into `configs/test/`, set
`split: test`, and retain only the selected parameter combination.
Regression models require an explicit `features` list; SMA uses its fixed
original carbon-intensity history column. Exclude the target and `split`
from inputs. Use validation to select parameters before evaluating on test.

The random forest config defines one 29-combination search with four workers.
It includes 15 squared-error combinations over depths 4/8/12/20/unlimited and
leaf sizes 1/20/100, six absolute-error combinations without single-observation
leaves, four feature-sampling variants, two half-size bootstrap samples, and two
100-tree variants. The shared baseline uses depth 8, leaf size 20, and 300 trees.

The CatBoost config contains 66 combinations. Its main block crosses MAE/RMSE,
depths 1/2/4/6/8, and 200/400/800 iterations at learning rate 0.05. Additional
runs compare learning schedules (0.01 with 1200 iterations; 0.1 with 400), L2
values 0.1/1/3/10/30, and split-score randomness, feature sampling, and Bernoulli
row sampling at depths 2/6. Randomness and sampling variants use MAE.

Both use the same target-only features and weekly evaluation as the existing
regression models, with a fixed seed. CatBoost uses one worker and disables auxiliary
training files. The configs contain explicit combinations rather than the full
cross-product of all parameters. The current search requires 760 fits across
eight validation origins, including the completed runs; narrow the next search
using validation results.

`runs` lists the exact parameter combinations to execute, in order. Each entry's
`params` overrides optional shared `base_params`; there is no Cartesian product:

```yaml
base_params:
  season_length: 168
runs:
  - params:
      n_seasons: 1
  - params:
      n_seasons: 2
```

Results are saved under
`experiments/results/<split>/<config_stem>/`. Each launch contains a
copy of `config.yaml`, an aggregate `metrics.csv` with one row per combination,
and numbered directories (`001`, `002`, ...). Each numbered directory contains
`params.json`, `predictions.csv`, `weekly_metrics.csv`, and `metrics.json`. Each launch replaces all
previous results for that split/config, including obsolete numbered directories.
Validation and test results with the same config name are stored separately.
Generated results are excluded from Git.
Result configs record the parameters used when those results were produced;
they may differ from the current configs under `experiments/configs/`.

Predictions contain feature time, target time, forecast origin, horizon (1–168
hours), training row count, `y` (the configured actual target), and `y_pred`,
in kg CO2/kWh. Metrics include MAE and RMSE in g CO2/kWh,
dimensionless R², row counts, and dataset/model metadata. `weekly_metrics.csv`
records scores and training size for each forecast origin. Run-level metrics
score all retained hourly predictions together. Aggregate metrics
collect the individual runs; they do not average across parameter combinations.
Regression parameter files use the estimator's `get_params()` output; CatBoost
reports explicitly supplied parameters, while sklearn-based estimators also
include defaults.
NaN parameter defaults, such as XGBoost's `missing` sentinel, are recorded as
the string `"NaN"` to keep JSON valid; model parameters are unchanged in memory.

Collect all saved experiment summaries without rerunning training:

```bash
uv run python experiments/collect_metrics.py
```

This rebuilds `experiments/results/metrics.csv` from the `metrics.csv` files in
both `results/valid/<config_stem>/` and `results/test/<config_stem>/`,
adding an `experiment` column with the config name. Each row identifies one
split/experiment/run combination and retains its metrics, parameters, and metadata.
Rerun collection after changing experiment results; the combined file is
replaced rather than appended. Use `--results-dir` for another results directory.
Compare scores on matching datasets, evaluation splits, and target times.

Open `notebooks/research/prediction_analysis.ipynb` and run all cells to inspect
saved forecasts and MAE-sorted metrics tables for validation and test. Widgets select the split, experiment, numbered run, and date
range. The plot compares actual and predicted carbon intensity by target time
in UTC, alongside saved full-split metrics and model parameters. Rerun the
notebook after generating new results to refresh the available selections.

For `dataset_target_only_v1.csv`, use `target: y` and
`target_offset_hours: 168`. For `dataset_v1.csv`, use
`target: y` and `target_offset_hours: 0`, with generation
and calendar features as inputs. The offset specifies label time relative to
the dataset index; it does not change or shift dataset values. Future actual
generation assumes perfect generation forecasts.

The runner uses weekly expanding-window backtesting. The first forecast origin
is one hour before the first target time in the chosen split; subsequent origins
are 168 hours apart in UTC. At each origin T, the model is fitted again using all
complete rows whose target time is at or before T, then predicts T+1 through
T+168. Training includes earlier validation/test observations once available;
hyperparameters stay fixed. Test begins with all available train and validation
history. No additional gap is imposed: label availability determines the cutoff.
For target-only, training requires feature time + 168 hours <= T, and the input
block spans T−167 through T. Preceding history is retained for SMA. Generation
models use the future generation inputs for the forecast week under the perfect
forecast assumption.

Only full 168-hour windows entirely within the selected split, with known labels
and complete features, are evaluated. An incomplete window is skipped without
shifting later origins; a partial final week is excluded. Insufficient SMA
history raises an error. Metrics record `evaluation_protocol: weekly_expanding`,
`n_origins`, first/last training sizes (`n_train`, `n_train_last`), and
`n_excluded_rows`. Select parameters using validation windows and freeze them
before test evaluation. Results from the former fit-once protocol must be rerun
before comparison with these scores.

## How target-only models forecast a full week

The target-only dataset aligns each historical feature row at time `t` with
the carbon intensity 168 hours later:

```python
y = target.shift(-168)
```

SMA and the regression models return one prediction per row. To forecast the
next week at time `T`, apply the model to the last 168 feature rows, shifting
their prediction times forward by 168 hours:

| Feature time | Predicted target time |
|---|---|
| `T - 167 hours` | `T + 1 hour` |
| `T - 166 hours` | `T + 2 hours` |
| ... | ... |
| `T` | `T + 168 hours` |

Together these predictions form the complete 168-hour forecast. Historical
target observations and lags for every input row are already available at `T`;
calendar features refer to the predicted hour and are known in advance. SMA
also needs history preceding the block to calculate its seasonal averages.
Fit the model using only labels available at `T`. This construction requires
neither future target observations nor feeding predictions back into the model.

The runner applies this alignment at each weekly origin and refits using only
labels available at that origin. Each complete week contributes 168 predictions
to the overall scores and one row to `weekly_metrics.csv`.

`src/experiment.py` handles training and temporal alignment;
`src/evaluator.py` calculates metrics. Model implementations are in
`src/models/`, and data preparation scripts are in `scripts/`.

Run checks with `uv run pytest`.

## Generation validation experiments

The `experiments/configs/valid/generation_v1_*.yaml` configs use `dataset_v1.csv`,
target `y`, offset zero, all seven generation volumes, and the two calendar flags.
Future actual generation assumes perfect generation forecasts. Run a config with,
for example:

```bash
uv run python experiments/run.py experiments/configs/valid/generation_v1_lgbm.yaml
```

Linear regression has four intercept/positive-coefficient combinations. Each
boosting model has 12 combinations: squared/absolute-error objectives, depths
2/4/6, and 200/600 trees (iterations for CatBoost), at learning rate 0.05.
LightGBM uses 4/16/32 leaves respectively. Random forest has six combinations:
squared error at depths 8/unlimited and leaf sizes 5/20, plus absolute error at
both depths with leaf size 20. Forests use 200 trees and four workers; boosting
uses one worker and all stochastic estimators use seed 42.
The 46 combinations require 368 fits across eight validation origins. Select
parameters on validation before creating single-run test configs.

## Renewable-only comparison

The `experiments/configs/valid/renewable_v1_*.yaml` configs use the same dataset,
calendar flags, and weekly evaluation as the generation experiments, but retain
only solar, onshore wind, offshore wind, and biomass generation. Future actual
renewable generation still assumes perfect forecasts; fossil generation and
cogeneration are excluded from inputs.

Each boosting model compares depths 2/6 and squared/absolute-error objectives
with 600 trees at learning rate 0.05 (four combinations). Random forest compares
depth 8/unlimited with 200 trees, squared error, and leaf size five (two).
Linear regression compares an intercept against no intercept, with unconstrained
coefficients (two). The 16 combinations require 128 fits across eight validation
origins. Run each config as for the generation experiments, for example:

```bash
uv run python experiments/run.py experiments/configs/valid/renewable_v1_catboost.yaml
```

Select the lowest-validation-MAE combination for each model before test evaluation.

The selected combinations are frozen in
`experiments/configs/test/renewable_v1_*.yaml`. Run them with the same command,
replacing `valid` with `test`, then rebuild the combined metrics with
`uv run python experiments/collect_metrics.py`.

The completed comparison uses eight validation weeks and seven test weeks.
All test models cover the same 1,176 hours, September 1–October 19, 2026 (UTC).
Scores below are in g CO2/kWh; each row uses its own validation-selected
parameters. Full-mix scores use the previously selected generation models.

| Model | Renewable valid MAE | Renewable test MAE | Renewable test RMSE | Full-mix test MAE |
|---|---:|---:|---:|---:|
| Linear regression | 36.420 | 34.164 | 40.979 | 20.171 |
| Random forest | 24.078 | 23.484 | 29.154 | 7.240 |
| LightGBM | 23.355 | 20.530 | 26.344 | 3.979 |
| XGBoost | 23.370 | 20.593 | 26.284 | 4.067 |
| CatBoost | 22.818 | 19.742 | 24.843 | 2.666 |

CatBoost has the lowest renewable-only validation and test MAE. Removing fossil
generation and cogeneration substantially increases error, showing the value of
knowing the realized production mix. This compares separately tuned feature
sets with different search budgets; it does not isolate individual sources'
contributions. Neither scenario measures operational accuracy with imperfect
generation forecasts.
