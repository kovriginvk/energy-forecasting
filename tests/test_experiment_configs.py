"""Check configurable targets, metrics, and numbered experiment artifacts."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from evaluator import evaluate
from experiment import run_experiment
from experiments.run import run_config
from models import BaseForecaster, RegressionModel


def test_evaluator_units_and_r2():
    predictions = pd.DataFrame({"y": [0.1, 0.2, 0.3], "y_pred": [0.11, 0.18, 0.3]})
    metrics = evaluate(predictions)
    assert metrics["mae_g_co2_per_kwh"] == pytest.approx(10)
    assert metrics["rmse_g_co2_per_kwh"] == pytest.approx(np.sqrt(500 / 3))
    assert metrics["r2"] == pytest.approx(0.975)


@pytest.mark.parametrize("offset", [0, 168])
def test_custom_target_alignment_and_training_cutoff(offset):
    index = pd.date_range("2026-01-01", periods=500, freq="h")
    dataset = pd.DataFrame(
        {"generation": np.arange(500), "carbon": np.arange(500) / 1000},
        index=index,
    )
    dataset["split"] = "train"
    dataset.loc[index[300]:, "split"] = "valid"

    class RecordingModel(BaseForecaster):
        feature_columns = ("generation",)

        def fit(self, X, y):
            self.train_index = X.index
            return self

        def predict(self, X):
            return X["generation"] / 1000

    model = RecordingModel()
    predictions, metrics = run_experiment(
        dataset, model, target="carbon", target_offset_hours=offset
    )
    assert model.train_index.max() + pd.Timedelta(hours=offset) <= index[300]
    assert (
        predictions["target_time (UTC)"]
        == predictions.index + pd.Timedelta(hours=offset)
    ).all()
    assert metrics["r2"] == 1
    with pytest.raises(ValueError, match="target and split"):
        run_experiment(dataset, RegressionModel(["carbon"]), target="carbon")


def test_explicit_combinations_and_replacement_of_previous_results(tmp_path):
    index = pd.date_range("2026-01-01", periods=30, freq="h")
    dataset = pd.DataFrame({"x": np.arange(30), "carbon": np.arange(30) / 1000}, index=index)
    dataset["split"] = ["train"] * 20 + ["valid"] * 5 + ["test"] * 5
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path)
    config = {
        "dataset": str(dataset_path),
        "model": "regression",
        "target": "carbon",
        "target_offset_hours": 0,
        "split": "valid",
        "features": ["x"],
        "base_params": {"fit_intercept": True},
        "runs": [{"params": {}}, {"params": {"fit_intercept": False, "positive": True}}],
    }
    config_path = tmp_path / "valid/example.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config))
    first = run_config(config_path, tmp_path / "results")
    assert first == tmp_path / "results/valid/example"
    assert yaml.safe_load((first / "config.yaml").read_text()) == config
    assert sorted(p.name for p in first.iterdir() if p.is_dir()) == ["001", "002"]
    summary = pd.read_csv(first / "metrics.csv")
    assert summary["run"].tolist() == [1, 2]
    assert summary["r2"].tolist() == pytest.approx([1, 1])
    assert json.loads((first / "002/params.json").read_text())["fit_intercept"] is False
    for number in ("001", "002"):
        predictions = pd.read_csv(first / number / "predictions.csv")
        metrics = json.loads((first / number / "metrics.json").read_text())
        assert len(predictions) == metrics["n_predictions"] == 5
        assert metrics["target"] == "carbon"
        assert {"mae_g_co2_per_kwh", "rmse_g_co2_per_kwh", "r2"} <= metrics.keys()

    config["split"] = "test"
    test_config = tmp_path / "test/example.yaml"
    test_config.parent.mkdir()
    test_config.write_text(yaml.safe_dump(config))
    other_results = run_config(test_config, tmp_path / "results")
    assert other_results == tmp_path / "results/test/example"
    marker = other_results / "metrics.csv"
    previous_test_metrics = marker.read_text()
    config["split"] = "valid"
    config["runs"] = [{"params": {"fit_intercept": False}}]
    config_path.write_text(yaml.safe_dump(config))
    second = run_config(config_path, tmp_path / "results")
    assert first == second
    assert not (second / "002").exists()
    assert pd.read_csv(second / "metrics.csv")["run"].tolist() == [1]
    assert yaml.safe_load((second / "config.yaml").read_text()) == config
    assert json.loads((second / "001/params.json").read_text())["fit_intercept"] is False
    assert marker.read_text() == previous_test_metrics


def test_config_split_must_match_directory(tmp_path):
    config_path = tmp_path / "valid/example.yaml"
    config_path.parent.mkdir()
    config_path.write_text("split: test\n")
    with pytest.raises(ValueError, match="matching its split"):
        run_config(config_path, tmp_path / "results")
