# Parts Usage Forecast Model — Version 1

## Overview

Version 1 is the original and simplest forecasting model created for the parts usage proof of concept. It uses historical monthly inventory usage to predict future monthly usage for each part with LightGBM.

## Purpose

Version 1 establishes a baseline that later model versions can be compared against.

## Required Input Columns

- `fpartno` — part number
- `Date` — month associated with the usage
- `Monthly Inventory Issues` — actual monthly usage quantity

Each part should have one row per month.

## Training and Validation

- Uses a fixed training cutoff date.
- Trains on all available history before the cutoff.
- Predicts the next six months recursively.
- Each prediction is added to the forecast history and used to help predict the following month.
- Compares the six predicted months with actual usage.

Example:

- Train through December 2025
- Predict January through June 2026
- Compare predictions with actual January through June 2026 usage

## Model

- Algorithm: `LGBMRegressor`
- Library: LightGBM
- Type: supervised machine learning / predictive AI

## Features

### Date and time

- Month
- Year
- Quarter
- Sequential time index

### Usage lags

- 1 month
- 2 months
- 3 months
- 6 months
- 12 months

### Rolling usage

- 3-month average
- 6-month average
- 12-month average
- 3-month standard deviation
- 6-month standard deviation

## Outputs

- Monthly prediction-versus-actual CSV files
- One validation chart per part
- Combined monthly validation results
- Summary results by part
- MAE and RMSE
- Optional S3 upload

## Primary Metric

Average Absolute Error (MAE) shows the average number of units between predicted usage and actual usage, regardless of whether the prediction was too high or too low.

## Recorded Result

- Approximate MAE: **5.29 units**

This result should only be compared with later versions when the same parts, date range, and validation period are used.

## Limitations

- Uses only historical usage.
- Does not include open demand, commitments, lead time, product family, or supplier behavior.
- Early forecast errors can influence later recursive predictions.
- All six forecast months are based on one fixed training cutoff.

## Role in the Project

Version 1 is the baseline model used to measure whether later feature and process changes improve forecasting accuracy.
