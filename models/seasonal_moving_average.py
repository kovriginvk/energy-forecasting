"""Row-wise baseline using current and seasonally lagged target values."""

import numpy as np
import pandas as pd

from .base import BaseForecaster


class SeasonalMovingAverage(BaseForecaster):
    """Average target(t - k * season_length) for k in range(n_seasons).

    The current target is included. With season_length=168 and n_seasons=2,
    predict y(t+168) as the mean of target(t) and target(t-168). The dataset
    defines the forecast lead; season_length controls historical spacing only.
    """

    feature_columns = ("emissionfactor (kg CO2/kWh)",)

    def __init__(self, season_length: int = 168, n_seasons: int = 4):
        for name, value in (
            ("season_length", season_length),
            ("n_seasons", n_seasons),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.season_length = season_length
        self.n_seasons = n_seasons

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalMovingAverage:
        """No parameters are learned; target history is supplied in predict."""
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Use only current and preceding rows on a complete hourly grid."""
        if not isinstance(X.index, pd.DatetimeIndex) or not (
            X.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
        ):
            raise ValueError("X must be sorted on a unique, complete hourly grid.")
        target = X[self.feature_columns[0]]
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError("Target history must contain only finite values.")
        seasons = pd.concat(
            [target.shift(k * self.season_length) for k in range(self.n_seasons)],
            axis=1,
        )
        return seasons.mean(axis=1, skipna=False).rename("y_pred")
