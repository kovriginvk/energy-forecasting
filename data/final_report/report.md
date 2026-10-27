# Forecasting the Carbon Intensity of Electricity Generation in the Netherlands

## 1. Problem Statement

The objective is to forecast electricity-generation carbon intensity in the Netherlands for the next seven days at hourly resolution: 168 values per forecast origin. The target is the NED-published CO₂ emission factor. Intensity and errors are expressed in g CO₂/kWh.

Three scenarios with different available information are compared:

| Scenario | Input data |
|---|---|
| Target-only | Target history and calendar features; target history alone for SMA |
| Renewable-only | Solar, onshore and offshore wind generation, biomass, and calendar features |
| Full generation | All renewable-only features, plus coal, natural gas, and cogeneration |

“Full generation” refers to the complete set of seven sources in the modeling dataset; nuclear and waste generation are not included.

Target-only assesses accuracy without external generation forecasts, using intensity history and calendar features known in advance.

**Main assumption:** generation forecasts were unavailable, so both generation scenarios use future actual production as a proxy for perfect forecasts. Their scores exclude generation forecast errors and do not establish operational forecasting accuracy.

## 2. Data and Preparation

### 2.1. Source and Features

Historical hourly exports from Nederlandse Energie Data (NED), as specified in the [assignment](../../docs/initial_assignment.md), were joined by UTC time. The resulting hourly grid contains 24,575 observations, from December 31, 2023, at 23:00 to October 20, 2026, at 21:00 UTC. Generation is stored in kWh; intensity is converted from kg CO₂/kWh to grams for metrics and figures.

Generation features and targets describe the same hour. Calendar features indicate Netherlands weekends and holidays for the predicted hour in Europe/Amsterdam time, including public and optional holidays and Liberation Day every year.

Target-only regressors use current intensity and lags of 1, 2, 3, 24, 48, 168, 336, and 504 hours. These capture the short-term dependence and daily/weekly recurrence observed in exploratory analysis. Seasonal moving average (SMA) uses intensity history alone.

Source selection also reflects export quality: `other` is missing 2,605 hours, waste contains long nearly constant stretches, and nuclear has gaps and a suspicious boundary observation. These sources are excluded; their additional predictive value was not evaluated separately.

A sample of processed observations is shown below, with generation converted to GWh.

| UTC time | Intensity, g CO₂/kWh | Solar, GWh | Onshore wind, GWh | Offshore wind, GWh |
|---|---:|---:|---:|---:|
| 2026-09-01 00:00:00 | 295.79 | 0.000 | 0.761 | 2.006 |
| 2026-09-01 01:00:00 | 311.84 | 0.000 | 0.711 | 1.602 |
| 2026-09-01 02:00:00 | 322.54 | 0.000 | 0.656 | 1.332 |

![Hourly intensity and generation during the first test week](01_observed_data_week.png)

*Figure 1. Observed carbon intensity (above) and seven generation sources (below), September 1–7, 2026.*

### 2.2. Missing Values and Corrections

The target has no missing values. Offshore wind is missing 25 hours, filled from the same hour of the previous day; longer gaps repeat that profile sequentially. Cogeneration is missing 26 hours, filled with the mean of values 168 and 336 hours earlier. Both rules use preceding history.

These heuristics preserve the hourly grid and, for cogeneration, weekly timing. They do not guarantee recovery of actual production; sensitivity to the imputation method was not assessed.

Suspicious onshore/offshore wind values at the final export hour were replaced with previous-hour values. This hour lies outside the evaluated test windows. Overall, 27 rows contain at least one corrected feature, and no generation gaps remain. Correction flags are retained for quality control and excluded from model inputs.

Target-only rows with insufficient lag history or unknown future labels are excluded from training and evaluation.

## 3. Models and Evaluation Methodology

### 3.1. Algorithms and Parameter Selection

Linear regression provides a simple baseline; random forest and gradient boosting (CatBoost, LightGBM, XGBoost) capture nonlinear relationships and interactions. SMA is an additional target-only baseline.

Hyperparameters are selected separately for each model/scenario by validation MAE, considering tree depth, tree count, loss functions, and other settings. Only the selected configuration is evaluated on test. Search budgets differ across scenarios, so the comparison does not isolate individual sources’ contributions.

The selected SMA uses a 24-hour period and averages 14 historical values. In both generation scenarios, selected CatBoost uses 600 trees, depth 6, learning rate 0.05, and absolute-error loss.

### 3.2. Temporal Split and Weekly Forecasts

The data are split chronologically by target time:

| Split | UTC period | Purpose |
|---|---|---|
| Train | Before July 1, 2026 | Initial training history |
| Validation | July 1–August 31, 2026 | Hyperparameter selection |
| Test | From September 1 to the end of observations | Final evaluation of selected configurations |

At each weekly origin T, models are refitted on all complete rows with labels known by T, then predict T+1 through T+168. Origins advance by 168 hours. Earlier validation/test labels enter later training once available; hyperparameters remain fixed. The first test fit includes all available training and validation history.

For target-only, y(t) = intensity(t+168). Applying the model to historical rows T−167 through T produces predictions for T+1…T+168. Training requires t+168 ≤ T. No future intensity or recursive prediction feedback is needed.

![Temporal split and weekly evaluation](02_evaluation_protocol.png)

*Figure 2. Chronological split and first three test windows. Blue segments indicate available training labels; green segments indicate the next 168 forecast hours. Earlier training history extends beyond the lower panel.*

Only complete 168-hour windows are scored: eight validation weeks (1,344 hours, July 1–August 25) and seven test weeks (1,176 hours, September 1–October 19, 2026). All models share these test hours; incomplete tails are excluded.

### 3.3. Metrics

MAE (mean absolute error) determines hyperparameter selection. RMSE (root mean squared error) emphasizes large deviations; R² is dimensionless. Overall scores pool hourly predictions rather than averaging weekly metrics.

## 4. Results and Conclusions

Tables report all selected configurations on the same 1,176 test hours, ordered by test MAE. Lower MAE and RMSE indicate smaller errors.

Each prediction figure shows the three lowest-MAE models on the full test set, selected for illustration after evaluation; hyperparameters were selected on validation. All panels cover September 1–30 with identical axis limits. Panel metrics cover only the 720 plotted hours. Dashed lines separate weekly forecasts, rather than one month-long forecast.

### 4.1. Target-only

| Model | MAE, g CO₂/kWh | RMSE, g CO₂/kWh | R² |
|---|---:|---:|---:|
| Random forest | 76.71 | 96.37 | 0.3852 |
| CatBoost | 79.78 | 103.11 | 0.2962 |
| Linear regression | 79.92 | 102.81 | 0.3002 |
| Seasonal moving average (SMA) | 80.87 | 102.48 | 0.3048 |
| XGBoost | 82.56 | 105.29 | 0.2661 |
| LightGBM | 82.91 | 106.17 | 0.2537 |

Random forest achieves the lowest test MAE, but improves only moderately over SMA. CatBoost and linear regression perform similarly; greater complexity offers little advantage over the seasonal baseline.

SMA won on validation (MAE 47.26 versus 54.07 for random forest). Its higher test error and the changed ranking show limited stability across periods. The cause remains unverified; these experiments cannot attribute it conclusively to seasonality or distribution shifts.

![Three best target-only models for September](03_target_only_v1_2026_09.png)

*Figure 3. Target-only: random forest, CatBoost, and linear regression reproduce some regular fluctuations but show substantial deviations, including during low-intensity periods.*

Target-only produces a complete weekly forecast without generation inputs, but is substantially less accurate here than models given future actual generation.

### 4.2. Renewable-only

| Model | MAE, g CO₂/kWh | RMSE, g CO₂/kWh | R² |
|---|---:|---:|---:|
| CatBoost | 19.74 | 24.84 | 0.9591 |
| LightGBM | 20.53 | 26.34 | 0.9541 |
| XGBoost | 20.59 | 26.28 | 0.9543 |
| Random forest | 23.48 | 29.15 | 0.9437 |
| Linear regression | 34.16 | 40.98 | 0.8888 |

CatBoost wins on validation and test. LightGBM and XGBoost perform similarly, and all three boosting methods outperform random forest and linear regression.

![Three best renewable-only models for September](03_renewable_v1_2026_09.png)

*Figure 4. Renewable-only: CatBoost, LightGBM, and XGBoost track the main fluctuations, with noticeable deviations in some intervals.*

Renewable-only CatBoost is the most promising evaluated option for the main task, subject to testing with real generation forecasts.

### 4.3. Full Generation

| Model | MAE, g CO₂/kWh | RMSE, g CO₂/kWh | R² |
|---|---:|---:|---:|
| CatBoost | 2.67 | 3.65 | 0.9991 |
| LightGBM | 3.98 | 5.43 | 0.9981 |
| XGBoost | 4.07 | 5.54 | 0.9980 |
| Random forest | 7.24 | 9.75 | 0.9937 |
| Linear regression | 20.17 | 25.61 | 0.9566 |

CatBoost also wins in full generation, followed by LightGBM and XGBoost at approximately 4 g CO₂/kWh MAE. All three outperform random forest and linear regression.

![Three best full-generation models for September](03_generation_v1_2026_09.png)

*Figure 5. Full generation: CatBoost, LightGBM, and XGBoost closely match intensity when future production is known.*

Adding coal, natural gas, and cogeneration is associated with a marked error reduction. However, different tuning budgets and the combined feature change prevent attributing the improvement to individual sources.

CatBoost’s near-perfect R² reflects learning a same-hour relationship between realized production and published intensity. This explains the informative inputs, without establishing equivalent weekly accuracy under generation forecast errors.

### 4.4. Overall Conclusions

Available generation information has a larger observed effect than the choice among boosting implementations. The lowest test MAEs are 76.71 for target-only, 19.74 for renewable-only, and 2.67 for full generation. These compare best observed results, not causal feature contributions.

CatBoost wins on validation and test in both generation scenarios. Target-only rankings are less stable, and the best test model has only a small advantage over SMA.

Models producing 168 hourly predictions have been implemented and evaluated. The original task remains only partially validated because future actual generation substitutes for forecasts. Evaluation with generation forecasts available at issue time is still required.

## 5. Next Steps

1. **Establish data availability.** Identify generation sources, publication delays, and forecast horizons. Evaluate archived forecast versions available at each intensity forecast origin, using the available subset if seven-day forecasts for all sources cannot be obtained.

2. **Compare with target-only.** Evaluate models using real generation forecasts against target-only over identical periods, horizons, and refitting rules. Select an approach considering accuracy and dependence on external data.

3. **Develop the best approach.** Compare feature sets, history lengths, lags, imputation, algorithms, loss functions, and training settings. Select changes on validation and confirm them on a separate test period covering different seasons.

4. **Deploy with monitoring.** Automate data retrieval, 168-hour forecasting, and retraining. Monitor input availability, delays, quality, execution failures, runtime, and MAE/RMSE as observations arrive. Add alerts and retain data/model versions for troubleshooting and rollback.
