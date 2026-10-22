# Technical Assignment: CO₂ Factor Forecasting Using Renewable Energy Data

## Objective

Build a machine learning model that estimates the CO₂ factor for the next 7 days, with 1 hour resolution, based on the forecasted renewable energy production in the Netherlands.

## Background

Nederlandse Energie Data (NED) provides open-access data on energy production in the Netherlands. In this assignment, you will use their data portal to download historical data and develop a predictive model.

Your task is to:

1. Collect and explore historical data on CO₂ emissions and renewable energy production.
2. Build and document a machine learning model that forecasts the CO₂ factor for the next 7 days with a resolution of 1 hour.
3. Evaluate and explain your modeling approach and results.

## Data Sources

You must create a free account at NED:

- [NED Data Portal Registration](https://ned.nl/nl/dataportaal/dataportaal-overzicht)

Once registered, download the following datasets:

- **CO₂ Factor (Historical):** [Totale Elektriciteitsproductie – CO₂ Factor](https://ned.nl/nl/dataportaal/energie-productie/elektriciteit/totale-elektriciteitsproductie)
- **Solar Energy Production (Historical):** [Zonne-energie](https://ned.nl/nl/dataportaal/energie-productie/elektriciteit/zonne-energie)
- **Other Renewable Sources (Optional but Recommended):** Wind, hydro, or biomass energy production from the same portal.
- You can use any other data source you may find useful.

## Assignment Requirements

You must:

### 1. Data Collection & Preprocessing

- Register and obtain access to the NED datasets.
- Clean, transform, and visualize the relevant datasets (optional).

### 2. Modeling

- Design a machine learning model (or multiple and compare them) to forecast the CO₂ factor for the next 7 days, with 1 hour resolution (i.e. 168 values), using renewable energy production as input features (and any other you find relevant, optional).
- Justify your choice of algorithms and pre-processing steps.
- You are free to use any Python library you find most relevant.

### 3. Documentation

- Clearly explain your methodology, model selection, and results. In a PDF report (max 5 pages, excluding figures).
- Include a well-documented README with setup instructions and project structure.

### 4. Code Quality

- Ensure readable, expandable, and commented code.
- Include any necessary file to reproduce your environment (e.g. `requirements.txt` or `environment.yml`).

## Deliverables

Please submit the following with a link to a GitHub repository:

- Source code (in Python, Notebook or scripts).
- README with setup and a brief explanation.
- Report explaining the approach, assumptions, and results (including visualizations of predictions vs actuals, and error metrics).

**Optional:**

- Error analysis or feature importance insights.
