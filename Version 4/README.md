# Parts Usage Forecast Model — Version 4

## Overview

Version 4 builds on the Version 3 rolling monthly forecasting approach and introduces **MPL Qty** as a new business feature.

`MPL Qty` comes from the Missing Parts List and represents demand that existed but could not be fulfilled because the part was unavailable. This gives the model information about demand that may not be fully reflected in actual inventory issues.

## Purpose

The purpose of Version 4 is to test whether historical missing-parts activity can improve forecast accuracy beyond historical usage patterns alone.

Version 3 remains the rolling baseline. Version 4 changes the feature set while keeping the same rolling validation structure.

## Required Input Columns

- `fpartno` — part number
- `Date` — month associated with the data
- `Monthly Inventory Issues` — actual monthly usage
- `MPL Qty` — missing-parts quantity

Blank MPL values are treated as zero when zero means no missing-parts activity.

## Model

- Algorithm: `LGBMRegressor`
- Objective: Poisson
- Model type: supervised machine learning / predictive AI
- Validation style: rolling one-month-ahead backtest

## Rolling Validation Approach

The model is retrained separately for each forecast month using all information available before that month.

Example:

- January 2026 prediction uses data through December 2025
- February 2026 prediction uses data through January 2026
- March 2026 prediction uses data through February 2026
- This continues through the validation period

This structure is intended to more closely represent how the model could operate in an automated monthly process.

## Existing Usage Features

Version 4 keeps the enhanced usage features from Version 3:

- Month, year, quarter, and time index
- Usage lags: 1, 2, 3, 6, and 12 months
- Rolling means: 3, 6, and 12 months
- Rolling medians: 3, 6, and 12 months
- Rolling 12-month total
- Rolling standard deviations: 3, 6, and 12 months
- Rolling 12-month minimum and maximum
- Zero-month percentage over 12 months
- 12-month coefficient of variation
- Recent 3-month average versus 12-month average
- 3-month and 6-month trends

## MPL Features Tested

The expanded MPL feature set includes:

- `mpl_lag_1`
- `mpl_lag_3`
- `mpl_lag_6`
- `mpl_rolling_sum_6`
- `mpl_rolling_sum_12`
- `mpl_months_12`
- `mpl_percentage_12`
- `potential_demand_12`

`potential_demand_12` represents prior 12-month actual usage plus prior 12-month MPL quantity. It is used only as a feature; it does not change the prediction target.

## Why MPL Was Added

Actual inventory issues only show parts that were successfully issued.

If a part was needed but unavailable, actual usage can understate the true demand for that part. MPL data provides an additional signal that may help distinguish:

- Low usage because demand was genuinely low
- Low usage because the part was unavailable

The model target remains **actual monthly inventory issues**.

## Initial Findings

MPL was successfully incorporated into LightGBM and appeared in feature importance, confirming that the model was able to use it.

However, initial MPL features produced little or no visible change for some parts. This suggested that short-term MPL history alone was not a strong enough signal, which led to testing broader 6- and 12-month MPL features.

MPL activity is also relatively sparse for many parts, so its value may vary significantly by part.

## Outputs

Version 4 can produce:

- Part number
- Prediction month
- Training-through month
- Decimal predicted usage
- Rounded predicted usage
- Actual usage
- Error
- Absolute error
- Overall MAE
- MPL diagnostic summaries
- S3 backtest output

## Evaluation

Version 4 should be compared against the Version 3 rolling baseline using the same parts, historical range, validation period, model settings, and MAE calculation.

This isolates the effect of adding MPL-derived information.

## Key Takeaway

Version 4 tests whether unfulfilled historical demand adds predictive value. Even if MPL does not materially reduce overall MAE, it may still be useful for shortage-prone parts and for identifying where recorded usage understates actual demand.
