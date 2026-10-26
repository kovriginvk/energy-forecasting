# EDA report: inputs and preprocessing for CO₂ forecasting

## Modeling decision

All **11 EDA notebooks / 384 code cells** executed successfully in fresh kernels. Their conclusions were checked against an independent comparison of **33 raw CSVs**, and final `Next:` recommendations were updated. Raw data remain unchanged; notebook outputs follow the repository's `nbstripout` convention.

Use the published **total-mix `emissionfactor (kg CO2/kWh)` as the target**. Start with **solar, onshore wind, and offshore wind interval energy**, calendar features, and target history available at the forecast origin. Test other sources as optional additions.

Archived forecasts are unavailable, so the agreed experiment uses actual generation. Providing future actuals for all 168 hours **assumes known future production and excludes generation forecast error**. This does not establish operational forecast accuracy. The exports also do not establish publication delays or revision history; historical availability at an origin remains an assumption.

## Coverage and source comparison

Coverage spans **2023-12-31 23:00 to 2026-10-20 22:00 UTC, end exclusive**: **24,575 hourly intervals**, equivalent to 2024-01-01 through 2026-10-20 in Dutch local time. There are two complete local calendar years and a partial 2026.

All datasets have zero null raw cells, duplicate starts, invalid/non-hour intervals, non-finite numbers, or negative values. Five generation sources nevertheless omit entire hours.

Energy is **GWh per hourly interval** (`volume / 1e6`). Zero percentages exclude missing hours. Pearson correlations use each source's available observations paired with the total-mix target.

| Source | Observed hours | Missing hours | Min–max energy | Mean energy | Zero hours | Target correlation | Initial role |
|---|---:|---:|---:|---:|---:|---:|---|
| Solar (`sun`) | 24,575 | 0 | 0–20.176 | 2.971 | 45.45% | −0.630 | Core |
| Onshore wind | 24,575 | 0 | 0–6.671 | 1.982 | 1.92% | −0.663 | Core |
| Offshore wind | 24,550 | 25 | 0–4.574 | 1.747 | 0.31% | −0.426 | Core, with quality flags |
| Cogeneration | 24,549 | 26 | 0.476–3.940 | 1.630 | 0% | +0.444 | Optional |
| Biomass | 24,575 | 0 | 0–1.838 | 0.427 | 9.91% | +0.564 | Optional renewable extension |
| Coal | 24,575 | 0 | 0–3.140 | 1.069 | 16.17% | +0.689 | Optional |
| Natural gas | 24,575 | 0 | 0.346–10.786 | 3.643 | 0% | +0.791 | First optional mix comparison |
| Waste | 24,550 | 25 | 0.072–0.419 | 0.316 | 0% | +0.062 | Defer pending review |
| Nuclear | 24,551 | 24 | 0–0.491 | 0.395 | 17.12% | +0.078 | Secondary comparison |
| Other | 21,970 | 2,605 | 0.0445–0.221 | 0.190 | 0% | +0.006 | Exclude initially |

These are descriptive associations, not independent predictive value or causal effects. Solar/wind correlations remain negative in every observed year; gas, coal, cogeneration, and biomass remain positive. Waste and nuclear change sign; Other's 2024 correlation is undefined because production is constant.

Different notebook matrices use different complete-case samples, explaining small numerical differences. Restricting all sources to their **21,968 common hours** preserves the broad conclusions, but unnecessarily removes data from models that do not use Other.

Two relationships deserve separate feature comparisons:

- **Onshore/offshore wind: correlation 0.736** on 24,550 paired hours; both are below their respective paired-sample 10th percentiles for **1,249 hours (5.1%)**. Keep both inputs initially and test incremental value.
- **Biomass/coal: correlation 0.820**, with the same longest zero spell: **818 hours, 2024-06-08 05:00–2024-07-12 06:00 UTC inclusive**. Test separately and jointly; shared operation or reporting is unverified.

As a diagnostic, the sum of all ten source volumes divided by total-mix volume has median **1.000000**, mean **1.000519**, and range **0.805001–1.089836** on common hours. Typical agreement does not establish disjoint/exhaustive categories. Confirm definitions before constructing totals or shares.

## Target and temporal patterns

The target is complete, with range **18.22–460.16 g CO₂/kWh** and mean **210.10**. It has no IQR flags; **246** adjacent-hour changes exceed the 99th-percentile threshold of **93.32 g CO₂/kWh**. Retain these changes rather than clipping automatically.

Target autocorrelations at 1/24/168 hours are **0.970/0.697/0.479**; after retrospective detrending, 24/168-hour values remain **0.654/0.404**. This motivates historical target features and daily/weekly baselines, without demonstrating forecast skill.

Solar has strong daily recurrence (24-hour autocorrelation **0.899**) and **11,170 zero hours**, with a longest zero run of 16 hours. Preserve nighttime zeros. Wind's 168-hour persistence is weaker (**0.127 onshore / 0.111 offshore**). Cogeneration is higher in winter, with changing seasonal amplitudes. Long low/zero regimes in biomass, coal, and nuclear limit interpretation of pooled annual profiles. Compare matching calendar periods rather than full years against partial 2026.

## Quality issues and treatment

### Missing intervals

Bounds are UTC interval starts, inclusive.

| Source | Missing period | Hours |
|---|---|---:|
| Offshore wind, waste | 2026-08-22 07:00 – 2026-08-23 07:00 | 25 each |
| Cogeneration | 2026-08-22 07:00 – 2026-08-23 08:00 | 26 |
| Nuclear, Other | 2026-08-22 08:00 – 2026-08-23 07:00 | 24 each |
| Other | 2025-09-15 10:00 – 2025-12-31 22:00 | 2,581 |

Keep gaps missing, including Other's October–December 2025 totals. Their cause is unconfirmed. Before quality exclusions and window construction, solar + onshore yields **24,575 complete rows**; adding offshore leaves **24,550**. All sources except Other leave **24,549**; adding Other reduces this to **21,968**. These are hourly counts, not usable 168-hour window counts.

### Boundary observations

Most power–energy residuals are small: at most **2 kWh** for total mix, **10** for solar, **0** for cogeneration, and **3** for biomass/coal/gas. Five observations differ substantially:

| Source | Interval start UTC | Energy / one power-hour | Energy − power × hours, kWh |
|---|---|---:|---:|
| Onshore wind | 2026-10-20 21:00 | ≈2/3 | −916,383 |
| Offshore wind | 2026-10-20 21:00 | ≈1/2 | −1,919,790 |
| Waste | 2026-10-20 21:00 | ≈3/4 | −80,008 |
| Nuclear | 2026-08-22 07:00 | ≈1/4 | −348,952 |
| Other | 2026-08-22 07:00 | 1/4 | −133,500 |

Other also falls to **55,250 kWh at 2025-09-15 09:00 UTC**, one quarter of its surrounding regime, immediately before the long gap. Power falls proportionally, so arithmetic consistency does not flag it.

Partial aggregation is unconfirmed. Preserve raw observations and flag them without automatic rescaling. For the first model, mask suspect energy in the derived table and exclude affected windows. Truncate the renewable-core endpoint before **2026-10-20 21:00 UTC** pending review; a retained-boundary sensitivity run can assess the effect.

### Flat regimes and zeros

Waste repeats **320,024 kWh for 1,540 hours** from May 2026, followed by long runs at 320,020. Other has only **four distinct values**: 178,000 and 221,000 kWh dominate, with two quarter-level boundary observations. Such regimes inflate persistence; defer waste and exclude Other until reporting provenance is clarified.

Nuclear has **4,202 zero hours (17.1%)**, including a **1,093-hour** run from 2024-04-05 17:00 through 2024-05-21 05:00 UTC. Biomass/coal have **2,436/3,975 zero hours**. Preserve zeros and long spells; the files do not establish their operational cause. Percentile thresholds are descriptive: ties can flag far more than 10%, and Other's 99th-percentile hourly change is zero.

### CO₂ fields

Keep the published target rather than replacing it with `emission / volume`. Their maximum absolute difference is **4.334 g CO₂/kWh** at **2024-09-29 06:00 UTC**; mean absolute difference is **0.143**. Accounting or aggregation differences are possible but unverified.

Source factors round to zero for solar, wind, biomass, nuclear, and Other; to **0.298/0.798/0.396 kg CO₂/kWh** for cogeneration/coal/gas. Source emissions therefore add little beyond energy, and zero factors do not establish zero lifecycle emissions or renewable status.

Waste normally rounds to **0.22**, with three exceptions:

| Interval start UTC | Factor, kg CO₂/kWh | Emissions − energy × factor, kg CO₂ |
|---|---:|---:|
| 2023-12-31 23:00 | 0.27 | −113.500 |
| 2025-10-25 23:00 | 0.1375 | +17,038.550 |
| 2025-10-26 00:00 | 0.0825 | +17,652.495 |

The latter two fall on the Dutch autumn clock-change date; UTC remains unambiguous. Reported emissions imply factors close to 0.22, but the cause is unknown. Flag the accounting discrepancy without automatically repairing factors or rejecting interval energy.

## Preprocessing and evaluation

1. **Align in UTC.** Concatenate annual files per source, parse timestamps, sort, and verify unique starts and matching one-hour ends. Reindex to an hourly grid; retain filenames and quality flags for audit. Do not drop the first December 31 UTC observation because annual labels follow Dutch local years.
2. **Keep one energy feature per source.** Use `volume`, optionally in GWh, and the published target, optionally in g CO₂/kWh. Omit constant metadata, redundant power/source-emission fields, and undocumented `percentage`. Capacity does not establish installed capacity. Do not assume future total-mix emissions or volume are available.
3. **Preserve zeros, gaps, and valid extremes.** Initially exclude windows requiring missing or masked inputs; never impute target labels. Keep the hourly grid when computing lags: dropping rows first compresses time. Do not clip seasonal peaks or ramps merely because IQR flags them, including solar's **1,767 flagged observations**.
4. **Use origin-available history.** Add candidate target lags and trailing summaries relative to the forecast origin, respecting release delay. Derive local hour, weekday, and annual position in `Europe/Amsterdam`; retain UTC horizon indexing. No target information after the origin may enter predictors. The centered EDA smoother must not become a feature.
5. **Fit preprocessing on training data only.** Apply this to scaling, imputation, thresholds, and feature selection. Trees do not require standardization; other models may benefit. Avoid independent yearly normalization and report errors in physical units.

Start with the renewable core, comparing solar + onshore against all three sources. Test **biomass** as a renewable extension; **gas first, then coal and cogeneration** as optional mix inputs, separately and jointly. Reserve **nuclear** for a secondary comparison. Future actual fossil generation adds information about the realized mix, so identify that stronger assumption separately from renewable-only results.

Use chronological expanding/rolling validation with complete **168-hour target paths**. One practical split is 2024 initial training, 2025 rolling validation, and observed 2026 final testing. Account for overlapping label spans at split boundaries and freeze model choices before testing.

Compare alternatives on **identical valid forecast origins**, reporting excluded windows and seasonal coverage. Weekly non-overlapping origins provide a simple first evaluation. Baselines: persistence, repeated last-observed 24-hour profile, and last-observed 168-hour profile. Daily repetition must reuse only the profile known at the origin throughout the week. Report **MAE/RMSE in g CO₂/kWh**, overall and by lead day/season. Final feature selection requires these experiments; EDA alone has not demonstrated model superiority.
