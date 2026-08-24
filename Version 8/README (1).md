# Parts Usage Forecast Model — Version 8

## Overview

Version 8 tests whether **recency weighting improves the commitment-based rolling usage model**.

It is based on Version 5 and uses:

- Historical usage
- Engineered usage features
- Historical `Avg Commits/Month`
- Rolling one-month-ahead validation
- Recency weighting during LightGBM training

Version 8 does **not** use MPL features.

## Purpose

Version 8 isolates the effect of recency weighting on the commitment-based model.

The clean comparison is:

- **Version 5:** Usage + Commitments + Rolling Date
- **Version 8:** Usage + Commitments + Rolling Date + Recency Weighting

This helps determine whether recent historical observations should receive more influence than older observations.

## Input File

Default notebook input:

`V4 Usage Training.csv`

Required columns:

- `fpartno`
- `Date`
- `Monthly Inventory Issues`
- `Avg Commits/Month`

The notebook keeps observations from January 2023 onward.

## Model

The model uses `LGBMRegressor` with:

- Objective: Poisson
- Estimators: 250
- Learning rate: 0.05
- Maximum depth: 5
- Minimum child samples: 10
- L2 regularization (`reg_lambda`): 1.0
- Random state: 42

## Rolling Backtest

The validation period begins January 2026 and covers six months.

Each validation month is predicted using only information available before that month.

Examples:

- January 2026 uses history through December 2025
- February 2026 uses history through January 2026
- March 2026 uses history through February 2026

The model is retrained before each monthly prediction.

## Recency Weighting

Version 8 gives more recent training observations greater influence while retaining older history.

Weights increase gradually from:

- Oldest usable training row: `1.0`
- Newest usable training row: `2.0`

This is implemented using `sample_weight` in LightGBM.

Older observations still contribute to training and can provide seasonality, variability, intermittent-demand, and unusual-demand examples.

## Usage Features

The model includes:

### Date / Time

- Month
- Year
- Quarter
- Time index

### Usage Lags

- Lag 1
- Lag 2
- Lag 3
- Lag 6
- Lag 12

### Rolling Demand

- 3-, 6-, and 12-month rolling means
- 3-, 6-, and 12-month rolling medians
- 12-month rolling total

### Variability

- 3-, 6-, and 12-month rolling standard deviations
- 12-month minimum
- 12-month maximum
- 12-month coefficient of variation

### Demand Behavior

- Percentage of zero-usage months during the prior 12 months
- Recent 3-month demand versus 12-month demand
- 3-month trend
- 6-month trend

## Commitment Features

The commitment features are created from historical `Avg Commits/Month` values available before the predicted month.

Features include:

- `commits_lag_1`
- `commits_lag_2`
- `commits_lag_3`
- `commits_lag_6`
- `commits_rolling_mean_3`
- `commits_rolling_mean_6`
- `commits_rolling_mean_12`
- `commits_rolling_sum_3`
- `commits_rolling_sum_6`
- `commits_recent_vs_annual`

## Outputs

The backtest produces:

- Part number
- Prediction month
- Training-through month
- Predicted usage
- Rounded predicted usage
- Actual usage
- Error
- Absolute error

The notebook also contains an optional S3 export cell.

Default S3 output filename:

`version_8_commits_recency_weighting_results.csv`

## Evaluation

Version 8 should primarily be compared to Version 5 because recency weighting is the intended experimental change.

Recommended evaluation metrics include:

- Mean Absolute Error (MAE)
- WAPE
- Forecast bias
- Part-level percentage error
- Performance by part type or demand pattern

## Experiment Summary

Version 8 answers:

> Does giving recent observations more influence improve the Usage + Commitments rolling forecast?

MPL is intentionally excluded so the impact of recency weighting can be evaluated without the additional MPL signal.
