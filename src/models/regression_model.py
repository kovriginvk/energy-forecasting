"""Linear regression with an explicit selection of row-wise input features."""

from collections.abc import Sequence
from typing import ClassVar

import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import BaseForecaster


class RegressionModel(BaseForecaster):
    """Fit a regressor to the same-row future y using selected features."""

    default_params: ClassVar[dict] = {}

    def __init__(self, feature_columns: Sequence[str], **model_params):
        if isinstance(feature_columns, str) or len(feature_columns) == 0:
            raise ValueError("feature_columns must be a non-empty sequence of names.")
        self.feature_columns = tuple(feature_columns)
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates.")
        if {"y", "split"}.intersection(self.feature_columns):
            raise ValueError("y and split must not be model inputs.")
        self.model_params = self.default_params | model_params
        self.estimator = self._create_estimator(**self.model_params)

    def _create_estimator(self, **model_params):
        return LinearRegression(**model_params)

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Preserve input order and use names accepted by all three libraries.
        return X.loc[:, list(self.feature_columns)].set_axis(
            [f"feature_{i}" for i in range(len(self.feature_columns))], axis=1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RegressionModel:
        """Learn only from the supplied training rows and their future labels."""
        self.estimator.fit(self._features(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict one future label per row, retaining the feature-time index."""
        return pd.Series(
            self.estimator.predict(self._features(X)), index=X.index, name="y_pred"
        )
