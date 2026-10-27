# Electricity carbon intensity forecasting

Forecast the carbon intensity of electricity generation in the Netherlands for
seven days at hourly resolution: **168 predictions per forecast origin**. The
project uses historical Nederlandse Energie Data (NED) exports to compare
renewable-generation models, a broader generation-mix comparison, and baselines
using only carbon-intensity history.

The [assignment](docs/initial_assignment.md) defines the requirements. The
[PDF report](data/final_report/report.pdf) and its
[Markdown source](data/final_report/report.md) explain preprocessing, algorithm
choices, assumptions, all 16 selected model results, and prediction plots.

**Scope:** archived generation forecasts were unavailable. Generation models
receive future actual production as a proxy for perfect forecasts. Their scores
exclude generation forecast errors and do not establish operational seven-day
forecasting accuracy. Evaluation with real generation forecasts available at
each forecast origin remains unfinished. Target-only models use observed target
history and calendar features known in advance.

## Setup

Requirements: Git, [uv](https://docs.astral.sh/uv/), and Python 3.14 (pinned in
`.python-version`). On Ubuntu/Debian, LightGBM also requires the system OpenMP
runtime, available as `libgomp1`.

```bash
git clone https://github.com/kovriginvk/energy-forecasting.git
cd energy-forecasting
uv sync --locked
```

`uv` can provision the pinned Python version if needed. `pyproject.toml` defines
the dependencies; `uv.lock` fixes the resolved environment, including the pytest
development dependency. Run the commands below from the repository root.

To use notebooks, open them in VS Code with the Python/Jupyter extensions and
select `.venv/bin/python` as the kernel. The environment includes `ipykernel`
and `ipywidgets`; a standalone Jupyter server is not included.

## Project structure

```text
├── data/
│   ├── raw/                  # Original NED CSV exports and joined dataset.csv
│   ├── processed/            # Generation and target-only modeling datasets
│   └── final_report/         # English report (Markdown/PDF) and five PNG figures
├── scripts/                  # Raw-data assembly and dataset preparation
├── src/
│   ├── models/               # SMA, linear regression, RF, CatBoost, LightGBM, XGBoost
│   ├── experiment.py         # Weekly forecasts, refitting, and time alignment
│   └── evaluator.py          # MAE, RMSE, and R²
├── experiments/
│   ├── configs/valid/        # Explicit hyperparameter search combinations
│   ├── configs/test/         # Frozen validation-selected configurations
│   ├── run.py                # Execute one YAML configuration
│   ├── collect_metrics.py    # Combine saved experiment summaries
│   └── results/              # Generated predictions/metrics; excluded from Git
├── notebooks/research/
│   ├── eda/                  # Source-by-source exploratory notebooks
│   ├── eda_report.md         # EDA findings and preprocessing rationale
│   ├── prediction_analysis.ipynb
│   └── final_report_figures.ipynb
├── tests/                    # Data preparation, models, and experiment checks
├── docs/                     # Assignment and chronological decision log
├── pyproject.toml
└── uv.lock
```

## Data and preprocessing

The repository includes the raw exports and processed datasets, so experiments
can run without portal credentials or another download. The joined data cover
24,575 hourly observations from December 31, 2023, at 23:00 through October 20,
2026, at 21:00 UTC. Generation volumes are in kWh; carbon intensity is stored in
kg CO₂/kWh and converted to g CO₂/kWh for metrics and figures.

To obtain the source data independently, register at the
[NED data portal](https://ned.nl/nl/dataportaal/dataportaal-overzicht), download
hourly historical CSV exports for 2024–2026, and place them under `data/raw/`:

| Export | Subdirectory |
|---|---|
| Total electricity production, including the CO₂ emission factor | `co2/` |
| Solar | `sun/` |
| Onshore / offshore wind | `wind_land/`, `wind_sea/` |
| Biomass, cogeneration | `biomass/`, `cogeneration/` |
| Coal, natural gas | `coal/`, `natural_gas/` |
| Waste, nuclear | `waste/`, `nuclear/` |

Each subdirectory can contain multiple `.csv` files. The assembly script reads
`validfrom (UTC)` and `volume (kWh)` from generation exports, and
`validfrom (UTC)` and `emissionfactor (kg CO2/kWh)` from total-production
exports. Use non-overlapping exports matching the included files; preparation
requires a sorted, unique, complete hourly time grid. `other/` exports are
retained for EDA but are not used by the assembly script or models. There is no
automated portal downloader.

Rebuild the joined and processed data in this order:

```bash
uv run python scripts/0_build_raw_dataset.py
uv run python scripts/1_prepare_dataset_v1.py
uv run python scripts/1_prepare_dataset_target_only_v1.py
```

These commands overwrite `data/raw/dataset.csv`,
`data/processed/dataset_v1.csv`, and
`data/processed/dataset_target_only_v1.csv`, respectively.

Preprocessing retains the published intensity target. Offshore wind gaps repeat
the previous daily profile; cogeneration gaps use the mean of observations 168
and 336 hours earlier. Suspect wind values at the final export hour are replaced
with previous-hour values. Imputation flags are retained for inspection and
excluded from model inputs. Waste/nuclear are excluded after quality review;
`other` has extensive missing hours. Target-only rows with insufficient lag
history or unknown labels are excluded from training/evaluation.

Weekend and Netherlands holiday flags refer to the predicted hour in
Europe/Amsterdam time, including Liberation Day every year. See the
[EDA findings](notebooks/research/eda_report.md) and
[report](data/final_report/report.md) for the rationale and limitations of these
choices. Preparation scripts encode the fixed dates and corrections for this
export snapshot; a new snapshot requires reviewing those rules.

## Modeling and evaluation

Three scenarios separate the information supplied to the models:

| Scenario / config prefix | Inputs | Dataset |
|---|---|---|
| Renewable-only / `renewable_v1` | Solar, onshore/offshore wind, biomass, calendar | `dataset_v1.csv` |
| Full generation / `generation_v1` | Renewable-only inputs plus coal, natural gas, cogeneration | `dataset_v1.csv` |
| Target-only / `target_only_v1` | Current intensity, lags 1/2/3/24/48/168/336/504 hours, calendar; history alone for SMA | `dataset_target_only_v1.csv` |

Linear regression provides a simple reference; random forest and gradient
boosting (CatBoost, LightGBM, XGBoost) capture nonlinear relationships. A seasonal
moving average (SMA) is an additional target-only baseline. Hyperparameters are
selected separately for each model/scenario by lowest validation MAE and frozen
in the test configs. Search budgets differ between scenarios.

Splits are chronological by **target time in UTC**:

| Split | Period | Evaluated complete windows |
|---|---|---|
| Train | Before July 1, 2026 | Initial history |
| Validation | July 1–August 31, 2026 | Eight weeks, 1,344 hours, through August 25 |
| Test | September 1 onward | Seven weeks, 1,176 hours, through October 19 |

The runner issues non-overlapping weekly forecasts. At origin `T`, it refits on
all complete rows whose labels are observed by `T` and predicts `T+1` through
`T+168`. Earlier validation/test labels enter later training once available;
hyperparameters remain fixed. Incomplete windows and the partial final week are
excluded. MAE and RMSE are in g CO₂/kWh; R² is dimensionless. Overall metrics
score all retained hourly predictions together, rather than averaging weekly
scores. Historical publication delays and revisions are not modeled.

For generation models, inputs and targets describe the same hour
(`target_offset_hours: 0`); each weekly forecast therefore requires generation
inputs for all 168 future hours, supplied here as actuals. For target-only,
`y(t) = intensity(t + 168 hours)` (`target_offset_hours: 168`):

| Historical feature time | Predicted target time |
|---|---|
| `T - 167 hours` | `T + 1 hour` |
| … | … |
| `T` | `T + 168 hours` |

Applying the model to this 168-row historical block produces the full next week
without future target values or recursive prediction feedback. Training requires
`t + 168 hours <= T`; SMA also receives preceding history.

## Run experiments

For a focused reproduction of the main renewable-only model, run its validation
search and frozen test configuration, then collect metrics:

```bash
uv run python experiments/run.py experiments/configs/valid/renewable_v1_catboost.yaml
uv run python experiments/run.py experiments/configs/test/renewable_v1_catboost.yaml
uv run python experiments/collect_metrics.py
```

To reproduce all three scenarios and all 16 selected test models, run every
validation and test configuration. The full validation search involves many
weekly refits and can take substantial time:

```bash
for config in experiments/configs/valid/*.yaml; do
    uv run python experiments/run.py "$config" || break
done

for config in experiments/configs/test/*.yaml; do
    uv run python experiments/run.py "$config" || break
done

uv run python experiments/collect_metrics.py
```

Each config declares the dataset, model, target `y`, target-time offset, split,
and explicit parameter combinations in `runs`. Regression models also declare
an ordered `features` list; SMA uses its fixed target-history column. Each run's
`params` overrides optional `base_params`; combinations are not expanded into a
Cartesian product. The config must be in a `valid/` or `test/` directory matching
its `split`. The supplied test configurations already contain the selected
parameters. For new experiments, select on validation before changing test
configs; exclude `y`, `split`, and imputation flags from features.

Each launch **replaces previous results for the same split/config** under
`experiments/results/<split>/<config_stem>/`. The output contains the config
snapshot, a summary `metrics.csv`, and numbered run directories with:

- `predictions.csv`: feature time, target time, forecast origin, horizon 1–168,
  training size, actual `y`, and `y_pred` (both in kg CO₂/kWh).
- `metrics.json` and `weekly_metrics.csv`: overall and per-origin scores,
  training sizes, and evaluation metadata.
- `params.json`: effective estimator parameters.

The collector rebuilds `experiments/results/metrics.csv` from saved summaries
without retraining, preserving split/experiment/run identity. Generated results
are not committed; rerun experiments to recreate them. Saved config snapshots
record the parameters actually used for each launch.

## Results and visualizations

All results below cover the same 1,176 test hours. These are the lowest observed
test MAEs per scenario; hyperparameters were chosen on validation.

| Scenario | Lowest-test-MAE model | MAE, g CO₂/kWh | RMSE, g CO₂/kWh | R² |
|---|---|---:|---:|---:|
| Target-only | Random forest | 76.71 | 96.37 | 0.3852 |
| Renewable-only | CatBoost | 19.74 | 24.84 | 0.9591 |
| Full generation | CatBoost | 2.67 | 3.65 | 0.9991 |

CatBoost is the validation and test winner in both generation scenarios. For
target-only, SMA wins on validation but reaches test MAE 80.87, indicating that
model rankings are not stable across these periods. The near-perfect
full-generation score reflects a same-hour relationship with the realized mix,
not demonstrated operational forecast accuracy. Renewable-only CatBoost is the
main candidate for evaluation with real generation forecasts.

Open `notebooks/research/prediction_analysis.ipynb` after running experiments to
inspect saved actual/predicted series, metrics, and parameters interactively.
Run `notebooks/research/final_report_figures.ipynb` after reproducing all
validation/test runs to regenerate the five PNG report figures. It verifies
validation/test parameter consistency and trains no models. September forecast
panels show the three lowest-full-test-MAE models per scenario; panel metrics
cover only the plotted month, while report tables cover the full test period.
The committed figures and report are available without rerunning experiments.

To rebuild the PDF from `data/final_report/report.md` and its PNG figures:

```bash
uv run scripts/2_build_report_pdf.py
```

The script declares its pinned rendering dependencies separately from the ML
environment and overwrites `data/final_report/report.pdf`. It uses WeasyPrint,
which requires system Pango libraries, and DejaVu fonts for the report layout.

The [report](data/final_report/report.pdf) contains the complete comparison and
next steps: evaluate real forecast inputs and their availability, compare across
more seasons, and assess robustness before deployment. The
[decision log](docs/dev_log.md) records the major methodological choices.

## Checks

```bash
uv run pytest
```

Tests cover dataset preparation, calendar/target alignment, model behavior, and
experiment configuration and metric collection.
