"""Forecasting models."""

from .base import BaseForecaster
from .seasonal_moving_average import SeasonalMovingAverage

__all__ = ["BaseForecaster", "SeasonalMovingAverage"]
