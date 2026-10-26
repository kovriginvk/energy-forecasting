"""Collect saved experiment summaries into one CSV without training models."""

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def collect_metrics(results_dir: Path = RESULTS_DIR) -> Path:
    """Rebuild the combined table, identifying rows by experiment and run."""
    tables = []
    paths = [
        path
        for split in ("valid", "test")
        for path in (results_dir / split).glob("*/metrics.csv")
    ]
    for path in sorted(paths):
        table = pd.read_csv(path)
        if not table["split"].eq(path.parent.parent.name).all():
            raise ValueError(f"Metrics split does not match its directory: {path}")
        table.insert(0, "experiment", path.parent.name)
        tables.append(table)
    if not tables:
        raise ValueError(f"No experiment metrics found in {results_dir}.")
    output = results_dir / "metrics.csv"
    pd.concat(tables, ignore_index=True).to_csv(output, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    print(collect_metrics(args.results_dir))


if __name__ == "__main__":
    main()
