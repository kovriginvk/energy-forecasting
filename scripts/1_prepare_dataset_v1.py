"""Prepare dataset v1 with generation and carbon intensity for the same hour."""

from pathlib import Path

import holidays
import pandas as pd

FEATURES = [
    f"{source}_volume (kWh)"
    for source in (
        "sun",
        "wind_land",
        "wind_sea",
        "cogeneration",
        "biomass",
        "coal",
        "natural_gas",
    )
]
TARGET = "emissionfactor (kg CO2/kWh)"
VALID_START = pd.Timestamp("2026-07-01")
TEST_START = pd.Timestamp("2026-09-01")


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep hourly features and the published target aligned at time t.

    Fill offshore wind by repeating the last available daily profile, and
    cogeneration by averaging observations 168 and 336 hours earlier.
    Replace suspect wind volumes at 2026-10-20 21:00 UTC with the previous hour.
    """
    if not raw.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Input must be sorted on a unique, complete hourly grid.")

    dataset = raw[FEATURES].copy()
    wind = dataset["wind_sea_volume (kWh)"].copy()
    cogeneration = dataset["cogeneration_volume (kWh)"]
    dataset["wind_land_is_imputed"] = False
    dataset["wind_sea_is_imputed"] = wind.isna()
    dataset["cogeneration_is_imputed"] = cogeneration.isna()

    # Updating in order repeats the known daily profile across gaps over 24h.
    for position in wind.isna().to_numpy().nonzero()[0]:
        if position < 24:
            raise ValueError("Offshore wind imputation needs 24 hours of history.")
        wind.iloc[position] = wind.iloc[position - 24]
    dataset["wind_sea_volume (kWh)"] = wind
    dataset["cogeneration_volume (kWh)"] = cogeneration.fillna(
        (cogeneration.shift(168) + cogeneration.shift(336)) / 2
    )
    # Repair the known export boundary, not the last row of arbitrary inputs.
    suspect_hour = pd.Timestamp("2026-10-20 21:00:00")
    if suspect_hour in dataset.index:
        for source in ("wind_land", "wind_sea"):
            column = f"{source}_volume (kWh)"
            dataset.loc[suspect_hour, column] = dataset.loc[
                suspect_hour - pd.Timedelta(hours=1), column
            ]
            dataset.loc[suspect_hour, f"{source}_is_imputed"] = True
    if dataset[FEATURES].isna().any().any():
        raise ValueError("Missing features remain after historical imputation.")

    dataset["is_imputed"] = dataset[
        ["wind_land_is_imputed", "wind_sea_is_imputed", "cogeneration_is_imputed"]
    ].any(axis=1)
    dataset["y"] = raw[TARGET]
    local_time = dataset.index.tz_localize("UTC").tz_convert("Europe/Amsterdam")
    # Include Liberation Day every year, even when it is not a general day off.
    calendar = holidays.country_holidays(
        "NL", years=local_time.year.unique(), categories=("public", "optional")
    )
    dataset["is_holiday"] = [day in calendar for day in local_time.date]
    dataset["is_weekend"] = local_time.dayofweek >= 5
    dataset["split"] = "train"
    dataset.loc[dataset.index >= VALID_START, "split"] = "valid"
    dataset.loc[dataset.index >= TEST_START, "split"] = "test"
    return dataset


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    raw = pd.read_csv(
        data_dir / "raw" / "dataset.csv",
        index_col="validfrom (UTC)",
        parse_dates=True,
    )
    dataset = prepare_dataset(raw)
    output_path = data_dir / "processed" / "dataset_v1.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path)
    print(f"Saved {len(dataset):,} rows to {output_path}")


if __name__ == "__main__":
    main()
