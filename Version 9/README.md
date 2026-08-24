# Parts Usage Forecast Model — Version 9

## Overview

Version 9 is a usage-only rolling forecasting model with **recency weighting**.

It is designed as a direct comparison to Version 3.

- **Version 3:** Usage + Rolling Date
- **Version 9:** Usage + Rolling Date + Recency Weighting

Version 9 intentionally excludes MPL and commitments so the effect of recency weighting can be isolated.

## Purpose

The purpose of Version 9 is to determine whether more recent historical observations should receive greater influence when predicting future monthly usage.

The model keeps older history because it may still contain useful examples of:

- Seasonality
- Demand spikes
- Intermittent usage
- Demand variability
- Longer-term patterns

However, newer training observations receive more weight because they may better represent current demand behavior.

## Input File

Default notebook input:

`V4 Usage Training.csv`

Required columns:

- `fpartno`
- `Date`
- `Monthly Inventory Issues`

The notebook keeps data from January 2023 onward.

## Model

The model uses `LGBMRegressor`.

Configuration:

- Objective: Poisson
- Estimators: 250
- Learning rate: 0.05
- Maximum depth: 5
- Minimum child samples: 10
- L2 regularization (`reg_lambda`): 1.0
- Random state: 42

## Rolling Backtest

The validation period begins January 2026 and covers six months.

Each month is predicted using only data available before the month being predicted.

Examples:

- January 2026 uses data through December 2025
- February 2026 uses data through January 2026
- March 2026 uses data through February 2026

The model is retrained before every monthly prediction.

## Recency Weighting

Version 9 applies sample weights during model training.

Weights increase gradually from:

- Oldest usable row: `1.0`
- Newest usable row: `2.0`

This means recent observations have greater influence without removing older observations from the training set.

## Features

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

- 3-month mean
- 6-month mean
- 12-month mean
- 3-month median
- 6-month median
- 12-month median
- 12-month total

### Variability

- 3-month standard deviation
- 6-month standard deviation
- 12-month standard deviation
- 12-month minimum
- 12-month maximum
- 12-month coefficient of variation

### Demand Behavior

- Percentage of zero-usage months during the prior 12 months
- Recent 3-month demand versus 12-month demand
- 3-month trend
- 6-month trend

## Outputs

The rolling backtest produces:

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

`version_9_usage_recency_weighting_results.csv`

## Evaluation

Version 9 should primarily be compared with Version 3.

Because both versions use the same usage-only feature structure and rolling validation approach, this comparison isolates the impact of recency weighting.

Recommended evaluation metrics:

- Mean Absolute Error (MAE)
- WAPE
- Forecast bias
- Part-level percentage error
- Performance by demand pattern

## Experiment Summary

Version 9 answers:

> Does giving recent historical usage observations greater influence improve the usage-only rolling forecasting model?
