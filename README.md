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
`params.json`, `predictions.csv`, and `metrics.json`. Each launch replaces all
previous results for that split/config, including obsolete numbered directories.
Validation and test results with the same config name are stored separately.
Generated results are excluded from Git.

Predictions contain feature time, target time, `y` (the configured actual
target), and `y_pred`, in kg CO2/kWh. Metrics include MAE and RMSE in g CO2/kWh,
dimensionless R², row counts, and dataset/model metadata. Aggregate metrics
collect the individual runs; they do not average across parameter combinations.
Regression parameter files include estimator defaults.

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

The runner fits once on train rows whose labels are available by the first
evaluated row time, then scores eligible rows in the chosen split. It preserves
preceding history for SMA. It does not simulate weekly refitting or forecast
issuance. A target-only block of 168 feature rows ending at time T can produce
predictions for T+1 through T+168 using only features available at T.

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

The existing runner evaluates rows individually using one fitted model. Its
metrics describe that evaluation protocol; it does not explicitly simulate
weekly forecast issuance or refitting, although the models support forming a
weekly forecast through the alignment above.

`src/experiment.py` handles training and temporal alignment;
`src/evaluator.py` calculates metrics. Model implementations are in
`src/models/`, and data preparation scripts are in `scripts/`.

Run checks with `uv run pytest`.
