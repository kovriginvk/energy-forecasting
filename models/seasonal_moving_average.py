"""Baseline that averages matching positions in recent seasonal cycles."""

import numpy as np
from numpy.typing import NDArray

from .base import BaseForecaster


class SeasonalMovingAverage(BaseForecaster):
    """Average the last n_seasons cycles and repeat their mean profile.

    season_length is measured in hours. With 168 and 4, the forecast averages
    matching hours from the last four weeks. Longer horizons repeat this fixed
    profile without updating it with predictions or future observations.
    """

    def __init__(self, season_length: int = 168, n_seasons: int = 4):
        for name, value in (
            ("season_length", season_length),
            ("n_seasons", n_seasons),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.season_length = season_length
        self.n_seasons = n_seasons

    def predict(self, horizon: int = 168) -> NDArray[np.float64]:
        """Predict using complete cycles immediately before the origin."""
        if not hasattr(self, "history_"):
            raise RuntimeError("Call fit before predict.")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        required = self.season_length * self.n_seasons
        if self.history_.size < required:
            raise ValueError(f"At least {required} hourly observations are required.")
        cycles = self.history_[-required:].reshape(
            self.n_seasons, self.season_length
        )
        profile = cycles.mean(axis=0)
        return np.resize(profile, horizon)
