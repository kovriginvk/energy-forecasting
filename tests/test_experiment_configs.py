"""Check configurable targets, metrics, and numbered experiment artifacts."""

import json
from pathlib import Path

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
    index = pd.date_range("2026-01-01", periods=900, freq="h")
    dataset = pd.DataFrame(
        {"generation": np.arange(900), "carbon": np.arange(900) / 1000},
        index=index,
    )
    dataset["split"] = "train"
    dataset.loc[index[400]:, "split"] = "valid"

    class RecordingModel(BaseForecaster):
        feature_columns = ("generation",)

        def __init__(self):
            self.fits = []

        def fit(self, X, y):
            self.fits.append(X.index)
            return self

        def predict(self, X):
            return X["generation"] / 1000

    model = RecordingModel()
    predictions, metrics = run_experiment(
        dataset, model, target="carbon", target_offset_hours=offset
    )
    origins = predictions["forecast_origin (UTC)"].unique()
    assert len(model.fits) == len(origins) == 2
    assert [len(rows) for rows in model.fits] == [400, 568]
    for rows, origin in zip(model.fits, origins):
        assert rows.max() + pd.Timedelta(hours=offset) == origin
    if offset:
        assert (predictions.index <= predictions["forecast_origin (UTC)"]).all()
    assert (
        predictions["target_time (UTC)"]
        == predictions.index + pd.Timedelta(hours=offset)
    ).all()
    assert predictions["horizon_hours"].tolist() == list(range(1, 169)) * 2
    assert (
        predictions["target_time (UTC)"] - predictions["forecast_origin (UTC)"]
        == pd.to_timedelta(predictions["horizon_hours"], unit="h")
    ).all()
    assert metrics["n_train"] == 400
    assert metrics["n_train_last"] == 568
    assert metrics["n_excluded_rows"] == 164
    assert metrics["evaluation_protocol"] == "weekly_expanding"
    assert metrics["r2"] == 1
    with pytest.raises(ValueError, match="target and split"):
        run_experiment(dataset, RegressionModel(["carbon"]), target="carbon")


def test_weekly_refitting_does_not_use_future_labels():
    index = pd.date_range("2026-01-01", periods=1000, freq="h")
    dataset = pd.DataFrame({"x": 1.0, "y": np.arange(1000) / 1000}, index=index)
    dataset["split"] = ["train"] * 400 + ["valid"] * 600

    class MeanModel(BaseForecaster):
        feature_columns = ("x",)

        def fit(self, X, y):
            self.mean = y.mean()
            return self

        def predict(self, X):
            return pd.Series(self.mean, index=X.index)

    expected, _ = run_experiment(dataset, MeanModel())
    changed = dataset.copy()
    # These labels are not observed at the first origin, but are at the second.
    changed.loc[index[400]:index[567], "y"] = 999
    actual, _ = run_experiment(changed, MeanModel())
    pd.testing.assert_series_equal(actual["y_pred"].iloc[:168], expected["y_pred"].iloc[:168])
    assert (actual["y_pred"].iloc[168:336] != expected["y_pred"].iloc[168:336]).all()


def test_incomplete_weeks_are_skipped_without_shifting_origins():
    index = pd.date_range("2026-01-01", periods=800, freq="h")
    dataset = pd.DataFrame({"x": np.arange(800), "y": np.arange(800) / 1000}, index=index)
    dataset["split"] = ["train"] * 200 + ["valid"] * 600
    dataset.loc[index[250], "y"] = np.nan
    dataset.loc[index[600], "x"] = np.nan
    predictions, metrics = run_experiment(dataset, RegressionModel(["x"]))
    assert metrics["n_origins"] == 1
    assert metrics["n_excluded_rows"] == 432
    assert predictions.index[0] == index[368]
    assert predictions["forecast_origin (UTC)"].iloc[0] == index[535]
    assert metrics["n_train"] == 367


def test_short_split_and_missing_hour_are_rejected():
    index = pd.date_range("2026-01-01", periods=300, freq="h")
    dataset = pd.DataFrame({"x": 1, "y": 0.1}, index=index)
    dataset["split"] = ["train"] * 200 + ["valid"] * 100
    with pytest.raises(ValueError, match="complete 168-hour window"):
        run_experiment(dataset, RegressionModel(["x"]))
    with pytest.raises(ValueError, match="hourly grid"):
        run_experiment(dataset.drop(index[20]), RegressionModel(["x"]))


def test_explicit_combinations_and_replacement_of_previous_results(tmp_path):
    index = pd.date_range("2026-01-01", periods=600, freq="h")
    dataset = pd.DataFrame({"x": np.arange(600), "carbon": np.arange(600) / 1000}, index=index)
    dataset["split"] = ["train"] * 200 + ["valid"] * 200 + ["test"] * 200
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
        assert len(predictions) == metrics["n_predictions"] == 168
        weekly = pd.read_csv(first / number / "weekly_metrics.csv")
        assert weekly["n_predictions"].tolist() == [168]
        assert weekly["n_train"].tolist() == [200]
        assert weekly["r2"].tolist() == pytest.approx([1])
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


def test_xgbm_default_missing_value_is_saved_as_valid_json(tmp_path):
    index = pd.date_range("2026-01-01", periods=400, freq="h")
    dataset = pd.DataFrame({"x": np.arange(400), "y": np.arange(400) / 1000}, index=index)
    dataset["split"] = ["train"] * 200 + ["valid"] * 200
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path)
    config = {
        "dataset": str(dataset_path),
        "model": "xgbm",
        "target": "y",
        "target_offset_hours": 0,
        "split": "valid",
        "features": ["x"],
        "runs": [{"params": {"n_estimators": 2}}],
    }
    config_path = tmp_path / "valid/xgbm.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config))
    output = run_config(config_path, tmp_path / "results")
    params = json.loads((output / "001/params.json").read_text())
    assert params["missing"] == "NaN"
    summary = pd.read_csv(output / "metrics.csv")
    assert json.loads(summary.loc[0, "model_params"])["missing"] == "NaN"
    assert summary.loc[0, "n_predictions"] == 168


@pytest.mark.parametrize("name", ["rf", "catboost"])
def test_new_model_configs_refit_weekly_and_save_results(tmp_path, name):
    config = yaml.safe_load(
        Path(f"experiments/configs/valid/target_only_v1_{name}.yaml").read_text()
    )
    rng = np.random.default_rng(42)
    index = pd.date_range("2026-01-01", periods=800, freq="h")
    dataset = pd.DataFrame(
        rng.uniform(size=(800, len(config["features"]))),
        columns=config["features"], index=index,
    )
    dataset["y"] = 0.1 + 0.2 * dataset.iloc[:, 0]
    dataset["split"] = ["train"] * 400 + ["valid"] * 400
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path)
    config["dataset"] = str(dataset_path)
    # Exercise MAE training and refitting without running the full search.
    params = {"n_estimators": 2, "criterion": "absolute_error"} if name == "rf" else {
        "iterations": 2,
    }
    config["runs"] = [{"params": params}]
    config_path = tmp_path / f"valid/{name}.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config))
    output = run_config(config_path, tmp_path / "results")
    saved = json.loads((output / "001/params.json").read_text())
    for key, value in (config["base_params"] | params).items():
        assert saved[key] == value
    metrics = json.loads((output / "001/metrics.json").read_text())
    assert metrics["n_origins"] == 2
    assert metrics["n_predictions"] == 336
    weekly = pd.read_csv(output / "001/weekly_metrics.csv")
    assert weekly["n_train"].tolist() == [400, 568]
    assert np.isfinite(weekly["mae_g_co2_per_kwh"]).all()
