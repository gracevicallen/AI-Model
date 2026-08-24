
"""Monthly parts usage forecast validation using LightGBM."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_FILE = "Usage Training.csv"
OUTPUT_FOLDER = Path("forecast_results")
TRAINING_CUTOFF = "2026-01-01"
FORECAST_MONTHS = 6
TARGET_COLUMN = "Monthly Inventory Issues"
PART_COLUMN = "fpartno"
DATE_COLUMN = "Date"
LAGS = [1, 2, 3, 6, 12]

UPLOAD_TO_S3 = False
S3_BUCKET = "YOUR_BUCKET_NAME"
S3_PREFIX = "forecast-results/all-parts"


def load_data(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    required = {PART_COLUMN, DATE_COLUMN, TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="raise")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="raise")
    data = (
        data.dropna(subset=[PART_COLUMN, DATE_COLUMN, TARGET_COLUMN])
        .sort_values([PART_COLUMN, DATE_COLUMN])
        .reset_index(drop=True)
    )

    duplicates = data.duplicated([PART_COLUMN, DATE_COLUMN], keep=False)
    if duplicates.any():
        duplicate_rows = data.loc[duplicates, [PART_COLUMN, DATE_COLUMN]].head(20)
        raise ValueError(
            "Duplicate part/month rows found. Each part should have one row per month.\n"
            f"{duplicate_rows}"
        )

    return data


def create_training_features(
    historical_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    feature_data = historical_data.copy()

    feature_data["month"] = feature_data[DATE_COLUMN].dt.month
    feature_data["year"] = feature_data[DATE_COLUMN].dt.year
    feature_data["quarter"] = feature_data[DATE_COLUMN].dt.quarter
    feature_data["time_idx"] = np.arange(len(feature_data))

    for lag in LAGS:
        feature_data[f"lag_{lag}"] = feature_data[TARGET_COLUMN].shift(lag)

    prior_usage = feature_data[TARGET_COLUMN].shift(1)
    feature_data["rolling_mean_3"] = prior_usage.rolling(3).mean()
    feature_data["rolling_mean_6"] = prior_usage.rolling(6).mean()
    feature_data["rolling_mean_12"] = prior_usage.rolling(12).mean()
    feature_data["rolling_std_3"] = prior_usage.rolling(3).std()
    feature_data["rolling_std_6"] = prior_usage.rolling(6).std()

    feature_columns = [
        "month",
        "year",
        "quarter",
        "time_idx",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_12",
        "rolling_std_3",
        "rolling_std_6",
    ]

    training_rows = feature_data.dropna(subset=feature_columns).reset_index(drop=True)
    return training_rows, feature_columns


def train_model(
    training_rows: pd.DataFrame,
    feature_columns: list[str],
) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
    )
    model.fit(training_rows[feature_columns], training_rows[TARGET_COLUMN])
    return model


def forecast_future_months(
    model: LGBMRegressor,
    historical_data: pd.DataFrame,
    feature_columns: list[str],
    forecast_months: int,
    part_number: str,
) -> pd.DataFrame:
    forecast_history = (
        historical_data[[DATE_COLUMN, TARGET_COLUMN]]
        .copy()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    predictions: list[dict[str, Any]] = []

    for _ in range(forecast_months):
        next_date = forecast_history[DATE_COLUMN].max() + pd.DateOffset(months=1)

        future_row: dict[str, float | int] = {
            "month": next_date.month,
            "year": next_date.year,
            "quarter": next_date.quarter,
            "time_idx": len(forecast_history),
        }

        for lag in LAGS:
            future_row[f"lag_{lag}"] = forecast_history[TARGET_COLUMN].iloc[-lag]

        recent_usage = forecast_history[TARGET_COLUMN]
        future_row["rolling_mean_3"] = recent_usage.iloc[-3:].mean()
        future_row["rolling_mean_6"] = recent_usage.iloc[-6:].mean()
        future_row["rolling_mean_12"] = recent_usage.iloc[-12:].mean()
        future_row["rolling_std_3"] = recent_usage.iloc[-3:].std()
        future_row["rolling_std_6"] = recent_usage.iloc[-6:].std()

        future_features = pd.DataFrame([future_row], columns=feature_columns)
        predicted_usage = max(0.0, float(model.predict(future_features)[0]))

        predictions.append(
            {
                PART_COLUMN: part_number,
                DATE_COLUMN: next_date,
                "Predicted Usage": predicted_usage,
            }
        )

        forecast_history = pd.concat(
            [
                forecast_history,
                pd.DataFrame(
                    {
                        DATE_COLUMN: [next_date],
                        TARGET_COLUMN: [predicted_usage],
                    }
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def save_validation_chart(
    comparison: pd.DataFrame,
    part_number: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        comparison[DATE_COLUMN],
        comparison[TARGET_COLUMN],
        marker="o",
        linewidth=2,
        label="Actual",
    )
    plt.plot(
        comparison[DATE_COLUMN],
        comparison["Predicted Usage"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Predicted",
    )
    plt.title(f"Forecast Validation - {part_number}")
    plt.xlabel("Month")
    plt.ylabel(TARGET_COLUMN)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def forecast_one_part(
    full_data: pd.DataFrame,
    part_number: str,
    training_cutoff: str,
    forecast_months: int,
    output_folder: Path,
) -> tuple[pd.DataFrame, dict[str, Any], LGBMRegressor]:
    part_data = (
        full_data[full_data[PART_COLUMN] == part_number]
        .copy()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    cutoff_date = pd.Timestamp(training_cutoff)
    forecast_end = cutoff_date + pd.DateOffset(months=forecast_months)

    historical_data = part_data[part_data[DATE_COLUMN] < cutoff_date].copy()
    actual_data = part_data[
        (part_data[DATE_COLUMN] >= cutoff_date)
        & (part_data[DATE_COLUMN] < forecast_end)
    ][[DATE_COLUMN, TARGET_COLUMN]].copy()

    if len(historical_data) < 24:
        raise ValueError(f"{part_number} has fewer than 24 historical months.")
    if len(actual_data) != forecast_months:
        raise ValueError(
            f"{part_number} has {len(actual_data)} actual validation months; "
            f"{forecast_months} were expected."
        )

    training_rows, feature_columns = create_training_features(historical_data)
    if training_rows.empty:
        raise ValueError(f"{part_number} has no usable rows after feature creation.")

    model = train_model(training_rows, feature_columns)
    forecast = forecast_future_months(
        model,
        historical_data,
        feature_columns,
        forecast_months,
        part_number,
    )

    comparison = actual_data.merge(forecast, on=DATE_COLUMN, how="inner")
    comparison["Difference"] = comparison["Predicted Usage"] - comparison[TARGET_COLUMN]
    comparison["Absolute Error"] = comparison["Difference"].abs()
    comparison = comparison[
        [
            PART_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
            "Predicted Usage",
            "Difference",
            "Absolute Error",
        ]
    ].sort_values(DATE_COLUMN)

    mae = mean_absolute_error(comparison[TARGET_COLUMN], comparison["Predicted Usage"])
    rmse = np.sqrt(
        mean_squared_error(comparison[TARGET_COLUMN], comparison["Predicted Usage"])
    )
    average_actual = comparison[TARGET_COLUMN].mean()
    average_predicted = comparison["Predicted Usage"].mean()
    mae_percent_of_average = (
        mae / average_actual * 100 if average_actual != 0 else np.nan
    )

    safe_part = safe_filename(part_number)
    monthly_csv_path = output_folder / f"Forecast_Validation_{safe_part}.csv"
    chart_path = output_folder / f"Forecast_Validation_{safe_part}.png"

    comparison.to_csv(monthly_csv_path, index=False)
    save_validation_chart(comparison, part_number, chart_path)

    summary_row = {
        PART_COLUMN: part_number,
        "Training Through": historical_data[DATE_COLUMN].max(),
        "Forecast Months": len(comparison),
        "Average Actual Usage": average_actual,
        "Average Predicted Usage": average_predicted,
        "MAE": mae,
        "RMSE": rmse,
        "MAE Percent of Average Usage": mae_percent_of_average,
        "Actual Forecast-Period Total": comparison[TARGET_COLUMN].sum(),
        "Predicted Forecast-Period Total": comparison["Predicted Usage"].sum(),
        "Monthly CSV File": str(monthly_csv_path),
        "Chart File": str(chart_path),
    }

    return comparison, summary_row, model


def upload_folder_to_s3(local_folder: Path, bucket: str, prefix: str) -> None:
    import boto3

    s3 = boto3.client("s3")
    for local_path in sorted(local_folder.iterdir()):
        if not local_path.is_file():
            continue
        s3_key = f"{prefix.rstrip('/')}/{local_path.name}"
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"Uploaded: s3://{bucket}/{s3_key}")


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    data = load_data(INPUT_FILE)

    print(f"Rows loaded: {len(data):,}")
    print(f"Unique parts: {data[PART_COLUMN].nunique():,}")

    all_comparisons: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for part_number in sorted(data[PART_COLUMN].dropna().unique()):
        print(f"Processing: {part_number}")
        try:
            comparison, summary_row, _ = forecast_one_part(
                full_data=data,
                part_number=part_number,
                training_cutoff=TRAINING_CUTOFF,
                forecast_months=FORECAST_MONTHS,
                output_folder=OUTPUT_FOLDER,
            )
            all_comparisons.append(comparison)
            summary_rows.append(summary_row)
            print(f"Finished {part_number} | MAE: {summary_row['MAE']:.2f}")
        except Exception as error:
            errors.append({PART_COLUMN: str(part_number), "Error": str(error)})
            print(f"Skipped {part_number}: {error}")

    if not all_comparisons:
        raise RuntimeError("No parts completed successfully. Review the printed errors.")

    monthly_results = (
        pd.concat(all_comparisons, ignore_index=True)
        .sort_values([PART_COLUMN, DATE_COLUMN])
        .reset_index(drop=True)
    )
    summary = pd.DataFrame(summary_rows).sort_values("MAE").reset_index(drop=True)

    monthly_results.to_csv(
        OUTPUT_FOLDER / "All_Parts_Monthly_Forecast_Validation.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_FOLDER / "All_Parts_Forecast_Summary.csv",
        index=False,
    )
    if errors:
        pd.DataFrame(errors).to_csv(
            OUTPUT_FOLDER / "Forecast_Errors.csv",
            index=False,
        )

    print("\nProcessing complete.")
    print(
        "Monthly output:",
        OUTPUT_FOLDER / "All_Parts_Monthly_Forecast_Validation.csv",
    )
    print(
        "Summary output:",
        OUTPUT_FOLDER / "All_Parts_Forecast_Summary.csv",
    )

    if UPLOAD_TO_S3:
        upload_folder_to_s3(OUTPUT_FOLDER, S3_BUCKET, S3_PREFIX)


if __name__ == "__main__":
    main()
