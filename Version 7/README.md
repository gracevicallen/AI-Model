# Parts Usage Forecast Model - Version 7

## Overview

Version 7 is a monthly parts-usage forecasting experiment built with LightGBM. It combines the Version 6 feature set with **recency weighting**, so newer historical training observations receive more influence than older observations while still preserving the full available history from 2023 onward.

## Version 7 Configuration

- Rolling one-month-ahead backtest
- Historical usage features
- Historical commitment features (`Avg Commits/Month`)
- Historical missing-parts / stockout features (`MPL Qty`)
- Recency weighting during model training
- LightGBM Poisson regression

## Validation Design

The validation period begins on **2026-01-01** and covers six months.

For each part and each validation month, the notebook:

1. Uses only information available before the month being predicted.
2. Recreates all training features from that historical data.
3. Retrains a LightGBM model for that part.
4. Predicts one month ahead.
5. Compares the prediction with actual usage.
6. Stores error and absolute error for evaluation.

This rolling structure is intended to mimic how a future automated monthly forecasting process would behave.

## Recency Weighting

Version 7 gives newer training rows more influence during model fitting.

- Oldest usable training row: weight `1.0`
- Newest usable training row: weight `2.0`
- Intermediate rows receive gradually increasing weights

Older history is still retained for seasonality, volatility, intermittent demand, and unusual demand patterns; it simply receives less influence than more recent history.

## Core Usage Features

The model uses:

- Month, year, quarter, and time index
- Usage lags: 1, 2, 3, 6, and 12 months
- Rolling means: 3, 6, and 12 months
- Rolling medians: 3, 6, and 12 months
- Rolling 12-month usage total
- Rolling standard deviation: 3, 6, and 12 months
- Rolling 12-month minimum and maximum
- Zero-month percentage over the prior 12 months
- 12-month coefficient of variation
- Recent 3-month demand compared with annual demand
- 3-month and 6-month demand trends

## Commitment Features

`Avg Commits/Month` represents known future commitments for the part. Version 7 uses:

- Commitment lags: 1, 2, 3, and 6 months
- Rolling commitment means: 3, 6, and 12 months
- Rolling commitment totals: 3 and 6 months
- Recent commitment level compared with the annual commitment level

All commitment features are shifted so the model does not use information from the month it is trying to predict.

## MPL Features

`MPL Qty` represents historical missing-parts quantity / unfulfilled demand caused by stockouts. Version 7 uses:

- MPL lags: 1, 3, and 6 months
- Rolling MPL totals: 6 and 12 months
- Number of prior-year months with MPL activity
- Percentage of prior-year months with MPL activity
- `potential_demand_12`: historical issued usage plus historical MPL quantity over the prior 12 months

MPL is used only as an input feature. The prediction target remains actual monthly inventory issues.

## Model

The notebook uses `LGBMRegressor` with:

- Objective: `poisson`
- Estimators: `250`
- Learning rate: `0.05`
- Maximum depth: `5`
- Minimum child samples: `10`
- L2 regularization (`reg_lambda`): `1.0`
- Random state: `42`

## Input Data

The notebook expects a CSV named:

`V4 Usage Training.csv`

Required columns:

- `fpartno`
- `Date`
- `Monthly Inventory Issues`
- `Avg Commits/Month`
- `MPL Qty`

The notebook filters the training history to **2023 onward**.

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

The final notebook cell uploads results to the SageMaker project S3 path as:

`version_7_mpl_commitments_recency_weighting_results.csv`

## Project Context

Version 7 is one experiment in a sequence designed to determine which combination of historical usage, business-demand signals, stockout information, and recency treatment produces the most reliable monthly usage forecast.

Version 7 should be compared with Versions 3-9 using the same validation period and evaluation metrics so that feature changes can be assessed consistently.
