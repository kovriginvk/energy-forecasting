"""CatBoost variant of the row-wise regression model."""

from typing import ClassVar

from .regression_model import RegressionModel


class CatBoostRegressionModel(RegressionModel):
    """Use CatBoost with the shared feature selection and prediction interface."""

    default_params: ClassVar[dict] = {
        "random_seed": 42,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }

    def _create_estimator(self, **model_params):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**model_params)
