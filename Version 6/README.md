# Version 6 — Inventory Usage Forecast (Commitments + MPL)

Version 6 predicts **monthly inventory issues** (parts usage) for each part number using a LightGBM regression model. Compared with earlier versions, it uses both **production commitments** and **MPL (missed/part short) quantity** as demand signals, in addition to historical usage.

## What the model predicts

For each part (`fpartno`), the model estimates:

**`Monthly Inventory Issues`** — how many units are expected to be issued in a future month.

Predictions are kept as continuous values for accuracy checks, then also rounded for reporting.

## Input data (local only)

Place this file next to the notebook (it is **not** stored in GitHub):

- `V4 Usage Training.csv`

Required columns:

| Column | Role |
|--------|------|
| `fpartno` | Part number |
| `Date` | Month |
| `Monthly Inventory Issues` | Actual usage (target) |
| `Avg Commits/Month` | Commitment / planned demand pressure |
| `MPL Qty` | Stockout / missed-part quantity |

Blank commitment or MPL values are treated as zero. Data before 2023 is dropped. Duplicate part/month rows are rejected.

## How predictions are made

### 1. Train one model per part (and per validation month)

The notebook runs a **rolling one-month-ahead backtest**:

- Validation starts at `2026-01-01`
- Forecast horizon is **6 months**
- For each part and each validation month:
  1. Use only history **before** that month
  2. Build features from that history
  3. **Retrain** a LightGBM model
  4. Predict that next month
  5. Compare to actual usage when available

This avoids leaking future information into the forecast.

### 2. Features the model uses

All features are built so the month being predicted never sees its own actual usage, commitments, or MPL.

**Calendar / time**

- Month, year, quarter, and a time index along the part history

**Usage history**

- Lags at 1, 2, 3, 6, and 12 months
- Rolling means, medians, totals, std. deviations, min/max
- Trends (recent vs older lags)
- Share of zero-usage months and coefficient of variation
- Recent average vs annual average

**Commitments (`Avg Commits/Month`)**

- Commitment lags and rolling means/sums
- Whether recent commitment pressure is above the longer-term level

**MPL (`MPL Qty`) — Version 6 addition**

- MPL lags and rolling sums
- How often MPL occurred in the last year
- `potential_demand_12` — combines recent usage-side signal with MPL to reflect unmet demand

Together, these tell the model not only what was issued, but also what was committed and what demand was missed when stock was short.

### 3. Model type

```text
LightGBM LGBMRegressor
objective = poisson
```

Poisson is a common choice for count-like usage that cannot go below zero. After prediction, negative values (if any) are clipped to zero.

Key settings used in the notebook:

- `n_estimators=250`
- `learning_rate=0.05`
- `max_depth=5`
- `min_child_samples=10`
- `reg_lambda=1.0`
- `random_state=42`

### 4. Outputs

The notebook produces:

- Predicted usage (raw and rounded)
- Actual usage for validation months
- Error and absolute error
- A per-part demand profile (usage, commitment, and MPL summaries before validation start)
- Optional local export: `version_6_mpl_commitments_results.csv` (gitignored)

## What’s in this folder

| File | Purpose |
|------|---------|
| `Version 6.ipynb` | End-to-end training, backtest, and export notebook |
| `requirements.txt` | Python package list from the SageMaker environment |
| `.gitignore` | Keeps training CSVs, result CSVs, and virtualenvs out of Git |
| `README.md` | This overview |

## Run locally

1. Create/activate a Python 3.12 virtual environment.
2. Install dependencies (at minimum: `numpy`, `pandas`, `lightgbm`, `scikit-learn`, Jupyter).
3. Put `V4 Usage Training.csv` in the working directory.
4. Open `Version 6.ipynb` and select that environment as the kernel.
5. Run cells top to bottom.

Do not commit the training CSV or forecast result CSVs.
