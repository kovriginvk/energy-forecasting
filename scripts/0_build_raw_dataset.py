"""Combine raw hourly generation volumes and the published carbon intensity."""

from pathlib import Path

import pandas as pd


def main():
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    sources = (
        "sun",
        "wind_land",
        "wind_sea",
        "cogeneration",
        "biomass",
        "coal",
        "natural_gas",
        "waste",
        "nuclear",
        "co2",
    )
    tables = []
    for source in sources:
        value_column = (
            "emissionfactor (kg CO2/kWh)" if source == "co2" else "volume (kWh)"
        )
        output_column = (
            "emissionfactor (kg CO2/kWh)"
            if source == "co2"
            else f"{source}_volume (kWh)"
        )
        table = pd.concat(
            pd.read_csv(path, usecols=["validfrom (UTC)", value_column])
            for path in sorted((raw_dir / source).glob("*.csv"))
        )
        tables.append(
            table.set_index("validfrom (UTC)").rename(
                columns={value_column: output_column}
            )
        )

    dataset = pd.concat(tables, axis=1).sort_index()
    output_path = raw_dir / "dataset.csv"
    dataset.to_csv(output_path)
    print(f"Saved {len(dataset):,} rows to {output_path}")


if __name__ == "__main__":
    main()
