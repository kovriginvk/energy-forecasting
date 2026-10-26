"""Prepare target history and lags for predictions 168 hours ahead."""

from pathlib import Path

import holidays
import pandas as pd

TARGET = "emissionfactor (kg CO2/kWh)"
HORIZON = 168
LAGS = (1, 2, 3, 24, 48, 168, 336, 504)
VALID_START = pd.Timestamp("2026-07-01")
TEST_START = pd.Timestamp("2026-09-01")


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep all rows; index features by observation time, split by label time.

    A block of 168 feature rows ending at the forecast origin predicts the
    following 168 hours. Exclude rows with incomplete features or unknown y
    from regression training. Never use y or split as input features.
    """
    if not raw.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Input must be sorted on a unique, complete hourly grid.")

    dataset = raw[[TARGET]].copy()
    for lag in LAGS:
        dataset[f"target_lag_{lag}"] = raw[TARGET].shift(lag)
    dataset["y"] = raw[TARGET].shift(-HORIZON)
    label_time = dataset.index + pd.Timedelta(hours=HORIZON)
    local_time = label_time.tz_localize("UTC").tz_convert("Europe/Amsterdam")
    # Include Liberation Day every year, even when it is not a general day off.
    calendar = holidays.country_holidays(
        "NL", years=local_time.year.unique(), categories=("public", "optional")
    )
    dataset["is_holiday"] = [day in calendar for day in local_time.date]
    dataset["is_weekend"] = local_time.dayofweek >= 5
    dataset["split"] = "train"
    dataset.loc[label_time >= VALID_START, "split"] = "valid"
    dataset.loc[label_time >= TEST_START, "split"] = "test"
    return dataset


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    raw = pd.read_csv(
        data_dir / "raw" / "dataset.csv",
        index_col="validfrom (UTC)",
        parse_dates=True,
    )
    dataset = prepare_dataset(raw)
    output_path = data_dir / "processed" / "dataset_target_only_v1.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path)
    print(f"Saved {len(dataset):,} rows to {output_path}")


if __name__ == "__main__":
    main()
