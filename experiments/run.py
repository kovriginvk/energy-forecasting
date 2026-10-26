"""Run explicit parameter combinations from a YAML configuration."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment import run_experiment
from models import (
    LGBMRegressionModel,
    RegressionModel,
    SeasonalMovingAverage,
    XGBMRegressionModel,
)

MODELS = {
    "sma": SeasonalMovingAverage,
    "regression": RegressionModel,
    "lgbm": LGBMRegressionModel,
    "xgbm": XGBMRegressionModel,
}


def run_config(config_path: Path, results_dir: Path | None = None) -> Path:
    """Replace the config's results with numbered runs and a summary."""
    config_path = Path(config_path)
    config_text = config_path.read_text()
    config = yaml.safe_load(config_text)
    split = config["split"]
    if split not in {"valid", "test"} or config_path.parent.name != split:
        raise ValueError("Store the config in a valid/test directory matching its split.")
    combinations = config["runs"]
    if not isinstance(combinations, list) or not combinations or not all(
        isinstance(run, dict) and isinstance(run.get("params"), dict)
        for run in combinations
    ):
        raise ValueError("runs must be a non-empty list of entries with params mappings.")
    dataset_path = ROOT / config["dataset"]
    dataset = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    model_class = MODELS[config["model"]]
    target = config["target"]
    features = config.get("features")
    if model_class is not SeasonalMovingAverage and not features:
        raise ValueError("Regression models require an explicit features list.")
    if model_class is SeasonalMovingAverage and features is not None:
        raise ValueError("SMA uses its fixed target-history feature; omit features.")
    output = (
        (results_dir if results_dir is not None else ROOT / "experiments/results")
        / split
        / config_path.stem
    )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(config_text)
    summary = []
    for number, run in enumerate(combinations, start=1):
        params = config.get("base_params", {}) | run["params"]
        if model_class is SeasonalMovingAverage:
            model = model_class(**params)
            effective_params = {
                "season_length": model.season_length,
                "n_seasons": model.n_seasons,
            }
        else:
            model = model_class(features, **params)
            effective_params = model.estimator.get_params()
        predictions, metrics = run_experiment(
            dataset,
            model,
            split=split,
            target=target,
            target_offset_hours=config["target_offset_hours"],
        )
        metrics.update(
            {
                "run": number,
                "model": config["model"],
                "dataset": str(dataset_path.resolve()),
                "target": target,
                "target_offset_hours": config["target_offset_hours"],
                "feature_columns": list(model.feature_columns),
            }
        )
        run_dir = output / f"{number:03d}"
        run_dir.mkdir()
        predictions.to_csv(run_dir / "predictions.csv")
        (run_dir / "params.json").write_text(
            json.dumps(effective_params, indent=2, allow_nan=False) + "\n"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        )
        summary.append(
            metrics
            | {"model_params": json.dumps(effective_params, allow_nan=False)}
        )
        pd.DataFrame(summary).to_csv(output / "metrics.csv", index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(run_config(args.config))


if __name__ == "__main__":
    main()
