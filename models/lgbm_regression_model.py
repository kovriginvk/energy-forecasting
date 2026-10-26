"""LightGBM variant of the row-wise regression model."""

from typing import ClassVar

from .regression_model import RegressionModel


class LGBMRegressionModel(RegressionModel):
    """Use LightGBM with the shared feature selection and prediction interface."""

    default_params: ClassVar[dict] = {"random_state": 42, "n_jobs": 1, "verbosity": -1}

    def _create_estimator(self, **model_params):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**model_params)
