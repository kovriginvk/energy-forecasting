"""Check collection across experiments and replacement of stale summary rows."""

import json

import pandas as pd
import pytest

from experiments.collect_metrics import collect_metrics


def test_collects_all_runs_and_rebuilds_without_duplicates(tmp_path):
    for name, split, runs in (("sma", "valid", [1, 2]), ("sma", "test", [1])):
        folder = tmp_path / split / name
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "run": runs,
                "split": split,
                "mae_g_co2_per_kwh": 10.0,
                "rmse_g_co2_per_kwh": 12.0,
                "r2": 0.8,
                "model_params": json.dumps({"fit_intercept": True}),
            }
        ).to_csv(folder / "metrics.csv", index=False)

    output = collect_metrics(tmp_path)
    combined = pd.read_csv(output)
    assert output == tmp_path / "metrics.csv"
    assert list(zip(combined["split"], combined["experiment"], combined["run"])) == [
        ("test", "sma", 1), ("valid", "sma", 1), ("valid", "sma", 2)
    ]
    assert combined["split"].tolist() == ["test", "valid", "valid"]
    assert combined["r2"].tolist() == [0.8, 0.8, 0.8]
    assert json.loads(combined.loc[0, "model_params"]) == {"fit_intercept": True}

    sma_path = tmp_path / "valid/sma/metrics.csv"
    updated = pd.read_csv(sma_path).iloc[:1].copy()
    updated["mae_g_co2_per_kwh"] = 5.0
    updated.to_csv(sma_path, index=False)
    assert collect_metrics(tmp_path) == output
    rebuilt = pd.read_csv(output)
    assert len(rebuilt) == 2
    assert rebuilt.loc[rebuilt["split"].eq("valid"), "mae_g_co2_per_kwh"].item() == 5.0


def test_no_saved_experiments(tmp_path):
    with pytest.raises(ValueError, match="No experiment metrics found"):
        collect_metrics(tmp_path)


def test_metrics_split_must_match_directory(tmp_path):
    folder = tmp_path / "valid/example"
    folder.mkdir(parents=True)
    pd.DataFrame({"split": ["test"]}).to_csv(folder / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="split does not match"):
        collect_metrics(tmp_path)
