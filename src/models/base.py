"""Common interface for row-wise forecasting models."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Predict each row's future label from its available features.

    feature_columns declares the dataset columns consumed by the model.
    X is indexed by feature time; y belongs to the same rows, 168 hours ahead
    in the target-only dataset. Never include y or split in feature_columns.
    """

    feature_columns: tuple[str, ...]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseForecaster:
        """Fit on complete training rows with known labels."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions indexed like X, without using later rows.

        History-based models receive preceding rows in X and return NaN where
        history is insufficient. Callers select evaluation rows afterwards.
        """
