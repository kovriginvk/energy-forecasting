"""Check row-wise seasonal averaging and causal target availability."""

import numpy as np
import pandas as pd
import pytest

from models import SeasonalMovingAverage


def history(values):
    return pd.DataFrame(
        {SeasonalMovingAverage.feature_columns[0]: values},
        index=pd.date_range("2026-01-01", periods=len(values), freq="h"),
    )


def test_one_season_predicts_future_label_with_current_target():
    X = history(np.arange(400, dtype=float))
    result = SeasonalMovingAverage(n_seasons=1).predict(X)
    np.testing.assert_array_equal(result, X.iloc[:, 0])
    assert result.index.equals(X.index)


def test_two_weeks_include_current_and_previous_week():
    X = history(np.arange(400, dtype=float))
    result = SeasonalMovingAverage(n_seasons=2).predict(X)
    assert result.iloc[:168].isna().all()
    np.testing.assert_array_equal(result.iloc[168:], np.arange(168, 400) - 84)


def test_four_weeks_compute_history_instead_of_using_prepared_lags():
    X = history(np.arange(800, dtype=float))
    X["target_lag_168"] = -999
    result = SeasonalMovingAverage().predict(X)
    assert result.iloc[:504].isna().all()
    assert result.iloc[504] == np.mean([504, 336, 168, 0])


def test_spacing_is_independent_of_forecast_lead():
    X = history([1, 2, 3, 5, 6, 7])
    result = SeasonalMovingAverage(season_length=3, n_seasons=2).predict(X)
    np.testing.assert_array_equal(result.iloc[3:], [3, 4, 5])


def test_predictions_do_not_use_later_rows_or_training_labels():
    X = history(np.arange(800, dtype=float))
    model = SeasonalMovingAverage().fit(
        X.iloc[:600], pd.Series(-999, index=X.index[:600])
    )
    expected = model.predict(X)
    changed = X.copy()
    changed.iloc[601:, 0] = 99999
    pd.testing.assert_series_equal(
        model.predict(changed).iloc[:601], expected.iloc[:601]
    )
    pd.testing.assert_series_equal(model.predict(X.iloc[:601]), expected.iloc[:601])


def test_missing_hour_is_not_treated_as_one_hour():
    X = history(np.arange(800)).drop(pd.Timestamp("2026-01-02"))
    with pytest.raises(ValueError, match="hourly grid"):
        SeasonalMovingAverage().predict(X)


@pytest.mark.parametrize("parameter", ["season_length", "n_seasons"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_invalid_parameters(parameter, value):
    with pytest.raises(ValueError, match="positive integer"):
        SeasonalMovingAverage(**{parameter: value})
