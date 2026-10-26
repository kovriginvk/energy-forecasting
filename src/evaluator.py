"""Score aligned carbon-intensity predictions in g CO2/kWh."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def evaluate(predictions: pd.DataFrame) -> dict[str, float]:
    """Expect actuals and predictions in kg CO2/kWh, on identical rows."""
    values = predictions[["y", "y_pred"]].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Evaluation needs at least two finite prediction pairs.")
    errors = (values[:, 1] - values[:, 0]) * 1000
    return {
        "mae_g_co2_per_kwh": float(np.abs(errors).mean()),
        "rmse_g_co2_per_kwh": float(np.sqrt((errors**2).mean())),
        "r2": float(r2_score(values[:, 0], values[:, 1])),
    }
