"""Common interface for forecasting models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaseForecaster(ABC):
    """Forecast from target history on a regular hourly grid."""

    def fit(self, y: ArrayLike) -> BaseForecaster:
        """Store chronological target values ending at the forecast origin.

        Values must be finite and one-dimensional. Callers must preserve the
        hourly grid and supply only observations available at the origin.
        """
        history = np.asarray(y, dtype=float)
        if history.ndim != 1 or history.size == 0:
            raise ValueError("y must be a non-empty one-dimensional sequence.")
        if not np.isfinite(history).all():
            raise ValueError("y must contain only finite values.")
        self.history_ = history.copy()
        return self

    @abstractmethod
    def predict(self, horizon: int = 168) -> NDArray[np.float64]:
        """Return the next horizon hourly predictions in chronological order."""
