"""XGBoost variant of the row-wise regression model."""

from typing import ClassVar

from .regression_model import RegressionModel


class XGBMRegressionModel(RegressionModel):
    """Use XGBoost with the shared feature selection and prediction interface."""

    default_params: ClassVar[dict] = {"random_state": 42, "n_jobs": 1}

    def _create_estimator(self, **model_params):
        from xgboost import XGBRegressor

        return XGBRegressor(**model_params)
