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
```

Each YAML config describes one dataset/model pair. Set `dataset` (relative to
the project root, or absolute), `model` (`sma`, `regression`, `lgbm`, `xgbm`),
`target`, `target_offset_hours`, and `split` (`valid` or `test`). Store configs in
`experiments/configs/<split>/`; the directory must match the YAML `split`.
For final evaluation, copy the selected config into `configs/test/`, set
`split: test`, and retain only the selected parameter combination.
Regression models require an explicit `features` list; SMA uses its fixed
original carbon-intensity history column. Exclude the target and `split`
from inputs. Use validation to select parameters before evaluating on test.

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

Predictions contain feature time, target time, forecast origin, horizon (1–168
hours), training row count, `y` (the configured actual target), and `y_pred`,
in kg CO2/kWh. Metrics include MAE and RMSE in g CO2/kWh,
dimensionless R², row counts, and dataset/model metadata. `weekly_metrics.csv`
records scores and training size for each forecast origin. Run-level metrics
score all retained hourly predictions together. Aggregate metrics
collect the individual runs; they do not average across parameter combinations.
Regression parameter files include estimator defaults.
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
`target: emissionfactor (kg CO2/kWh)` and `target_offset_hours: 0`, with generation
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
