# Parts Usage Forecast Model — Version 3

## Overview

Version 3 keeps the enhanced Version 2 feature set but changes the validation method to a rolling one-month-ahead backtest.

Instead of training once and predicting all six months from one cutoff, the model is retrained before each forecast month using all actual usage available through the prior month.

## Purpose

Version 3 tests how the model would behave in a future monthly refresh process where new actual usage is added before the next prediction.

## Required Input Columns

- `fpartno` — part number
- `Date` — month associated with the usage
- `Monthly Inventory Issues` — actual monthly usage quantity

Each part should have one row per month.

The tested version kept data from January 2023 forward.

## Rolling Validation Method

- January 2026 uses actual data through December 2025.
- February 2026 uses actual data through January 2026.
- March 2026 uses actual data through February 2026.
- April 2026 uses actual data through March 2026.
- May 2026 uses actual data through April 2026.
- June 2026 uses actual data through May 2026.

The model is retrained separately before every monthly prediction.

## Difference from Version 2

### Version 2

- One fixed cutoff
- One trained model
- Six recursive predictions
- Later months use earlier predicted values

### Version 3

- One validation start date
- A changing monthly cutoff
- Six separately trained models
- Each month can use the prior month's actual usage

## Model

- Algorithm: `LGBMRegressor`
- Library: LightGBM
- Type: supervised machine learning / predictive AI

## Features

Version 3 uses the same enhanced feature set as Version 2:

- Month, year, quarter, and time index
- Lags for 1, 2, 3, 6, and 12 months
- Rolling averages for 3, 6, and 12 months
- Rolling medians for 3, 6, and 12 months
- Rolling 12-month total
- Rolling standard deviations for 3, 6, and 12 months
- Rolling 12-month minimum and maximum
- Percentage of zero-use months over the prior 12 months
- 12-month coefficient of variation
- Recent 3-month average compared with the 12-month average
- 3-month and 6-month trends

## Outputs

- Part number
- Prediction month
- Training-through month
- Decimal predicted usage
- Rounded predicted usage
- Actual usage
- Error
- Absolute error
- Overall MAE
- MAE by part
- Optional S3 upload

The `Training Through` column verifies that the rolling logic is working.

## Evaluation Result

The hypothesis was that using the newest prior-month actual usage would improve accuracy.

In the historical test, Version 3 did not outperform Version 2.

This does not mean rolling retraining is incorrect. It means that the extra month of usage did not provide enough new predictive information to reduce MAE during this test period.

## Why Rolling Retraining May Not Improve MAE

- One new month is a small addition to the existing history.
- The newest month may be unusual and shift the model in the wrong direction.
- Existing lag and rolling features already summarize recent usage.
- Demand may be driven by business factors that are not yet included.
- Retraining LightGBM creates a new model each month, and the new model is not guaranteed to perform better.

## Production Relevance

Rolling retraining may still be appropriate for a future automated process:

- Add the newest completed month's actual usage
- Retrain the model
- Predict the next required forecast period

Version 3 is therefore useful as a process-design experiment even though Version 2 remains the stronger accuracy benchmark.

## Limitations

- Still uses only historical usage features.
- Tests one-month-ahead forecasting, which may not match required purchasing lead times.
- Does not include open demand, commitments, jobs, sales orders, or lead-time data.
- A one-month-ahead forecast may be too late for long-lead inventory decisions.

## Role in the Project

Version 3 evaluates a more realistic monthly retraining workflow. Its result suggests that future accuracy improvements are more likely to come from meaningful business features than from the retraining schedule alone.
