# Parts Usage Forecast Model — Version 2

## Overview

Version 2 expands Version 1 by adding features that describe typical annual usage, medians, variability, demand range, zero-use behavior, and trends.

It keeps the same fixed-cutoff validation approach as Version 1 so the effect of the new features can be measured fairly.

## Purpose

Version 2 tests whether richer usage-based features improve forecast accuracy without changing the LightGBM algorithm.

## Required Input Columns

- `fpartno` — part number
- `Date` — month associated with the usage
- `Monthly Inventory Issues` — actual monthly usage quantity

Each part should have one row per month.

The tested version kept data from January 2023 forward.

## Training and Validation

- Uses a fixed training cutoff date.
- Trains once on all history before the cutoff.
- Predicts the next six months recursively.
- Later forecast months use earlier predicted values.
- Compares the six predictions with actual usage.

Example:

- Train through December 2025
- Predict January through June 2026
- Compare with actual January through June 2026 usage

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

### Rolling averages

- 3 months
- 6 months
- 12 months

### Rolling medians

- 3 months
- 6 months
- 12 months

Medians were added because they are less affected by unusual usage spikes.

### Annual demand

- Rolling 12-month total

This helps the model recognize typical annual usage even when orders occur in different months each year.

### Demand variability

- 3-month standard deviation
- 6-month standard deviation
- 12-month standard deviation

### Demand range

- Lowest monthly usage in the previous 12 months
- Highest monthly usage in the previous 12 months

### Intermittent demand

- Percentage of the previous 12 months with zero usage

### Demand stability

- 12-month coefficient of variation

This measures volatility relative to the part's normal usage level.

### Recent versus annual demand

- Difference between the 3-month average and the 12-month average

### Trends

- 3-month trend
- 6-month trend

## Outputs

- Prediction-versus-actual results by part and month
- Rounded predictions for presentation
- Decimal predictions for accuracy calculations
- Error and absolute error
- Overall MAE
- MAE by part
- Six-month actual and predicted totals
- Optional S3 upload

## Recorded Result

- Version 1 MAE: **5.29 units**
- Version 2 MAE: **3.89 units**
- Improvement: **26.5% lower average forecasting error**

This comparison is valid only when both versions use the same parts, date range, and validation period.

## Why Version 2 Improved

The model received more context about:

- Typical monthly usage
- Typical annual usage
- Demand volatility
- Outliers
- Zero-use behavior
- Recent changes in demand
- Short- and medium-term trends

## Limitations

- Still relies only on historical usage.
- Does not include open demand, commitments, jobs, sales orders, lead time, supplier performance, or product family.
- Earlier recursive forecast errors can affect later months.
- More features may not improve every part.
- Highly intermittent parts may require a different forecasting method.

## Role in the Project

Version 2 is the strongest fixed-cutoff model tested so far and is the current benchmark for future improvements.
