"""Check selectable features, row alignment, and shared regression variants."""

import numpy as np
import pandas as pd
import pytest

from models import (
    CatBoostRegressionModel,
    LGBMRegressionModel,
    RandomForestRegressionModel,
    RegressionModel,
    XGBMRegressionModel,
)


@pytest.fixture
def samples():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "emissionfactor (kg CO2/kWh)": rng.uniform(0, 1, 400),
            "is_weekend": rng.choice([True, False], 400),
            "unused": rng.normal(size=400),
        },
        index=pd.date_range("2026-01-01", periods=400, freq="h"),
    )
    y = 2 * X.iloc[:, 0] + 0.3 * X["is_weekend"] + 1
    return X, y


@pytest.mark.parametrize(
    "model_class",
    [
        RegressionModel,
        LGBMRegressionModel,
        XGBMRegressionModel,
        RandomForestRegressionModel,
        CatBoostRegressionModel,
    ],
)
def test_selected_features_and_order_are_used_in_fit_and_predict(samples, model_class):
    X, y = samples
    features = ["is_weekend", "emissionfactor (kg CO2/kWh)"]
    params = {} if model_class is RegressionModel else {"n_estimators": 20}
    if model_class is CatBoostRegressionModel:
        params = {"iterations": 20, "depth": 2}
    model = model_class(features, **params).fit(X.iloc[:300], y.iloc[:300])
    # An excluded column must not affect training, including unknown labels.
    changed = X.assign(unused=np.nan, y=-999, split="test")
    other = model_class(features, **params).fit(changed.iloc[:300], y.iloc[:300])
    expected = model.predict(X.iloc[300:])
    actual = other.predict(changed.iloc[300:, ::-1])
    pd.testing.assert_series_equal(actual, expected)
    assert actual.index.equals(X.index[300:])
    assert actual.name == "y_pred"
    assert np.isfinite(actual).all()
    assert actual.nunique() > 1
    if model_class is RegressionModel:
        np.testing.assert_allclose(actual, y.iloc[300:], atol=1e-12)
    elif model_class is CatBoostRegressionModel:
        assert model.estimator.get_params()["iterations"] == 20
    else:
        assert model.estimator.get_params()["n_estimators"] == 20


def test_single_feature_does_not_require_other_dataset_columns(samples):
    X, _ = samples
    X = X[["emissionfactor (kg CO2/kWh)"]]
    y = 3 * X.iloc[:, 0] - 1
    model = RegressionModel(X.columns).fit(X.iloc[:300], y.iloc[:300])
    np.testing.assert_allclose(model.predict(X.iloc[300:]), y.iloc[300:], atol=1e-12)


@pytest.mark.parametrize("features", [[], "target", ["y"], ["split"], ["a", "a"]])
def test_rejects_invalid_or_leaking_feature_selection(features):
    with pytest.raises(ValueError):
        RegressionModel(features)
