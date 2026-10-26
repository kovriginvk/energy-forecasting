"""Check seasonal alignment and the forecast-origin boundary."""

import numpy as np
import pytest

from models import SeasonalMovingAverage


def test_one_week_repeats_last_observed_week():
    history = np.arange(3 * 168, dtype=float)
    model = SeasonalMovingAverage(n_seasons=1).fit(history)
    np.testing.assert_array_equal(model.predict(), history[-168:])


def test_four_weeks_average_matching_hours():
    profile = np.arange(168, dtype=float)
    weeks = [profile + offset for offset in (0, 10, 20, 30)]
    history = np.concatenate([np.full(17, -999), *weeks])
    model = SeasonalMovingAverage().fit(history)
    np.testing.assert_array_equal(model.predict(), profile + 15)


def test_short_season_repeats_fixed_profile():
    model = SeasonalMovingAverage(season_length=3, n_seasons=2)
    model.fit([1, 2, 3, 5, 6, 7])
    np.testing.assert_array_equal(model.predict(8), [3, 4, 5, 3, 4, 5, 3, 4])


def test_fit_copies_and_replaces_history():
    history = np.array([1.0, 2.0])
    model = SeasonalMovingAverage(season_length=2, n_seasons=1).fit(history)
    history[:] = 99
    np.testing.assert_array_equal(model.predict(2), [1, 2])
    model.fit([3, 4])
    np.testing.assert_array_equal(model.predict(2), [3, 4])


def test_missing_or_insufficient_history_is_rejected():
    model = SeasonalMovingAverage()
    with pytest.raises(RuntimeError):
        model.predict()
    model.fit(np.ones(671))
    with pytest.raises(ValueError):
        model.predict()
    with pytest.raises(ValueError):
        model.fit([1, np.nan])
