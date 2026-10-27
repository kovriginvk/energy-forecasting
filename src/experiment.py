"""Evaluate complete weekly forecasts with expanding-window refitting."""

import pandas as pd

from evaluator import evaluate
from models import BaseForecaster


def run_experiment(
    dataset: pd.DataFrame,
    model: BaseForecaster,
    split: str = "valid",
    target: str = "y",
    target_offset_hours: int = 168,
) -> tuple[pd.DataFrame, dict]:
    """Refit at weekly origins using all labels available at each origin.

    The offset describes label time relative to the row index, not the number
    of forecast horizons. Future generation inputs assume perfect forecasts.
    Only complete 168-hour windows are scored; incomplete windows are skipped.
    """
    if split not in {"valid", "test"}:
        raise ValueError("Evaluation split must be valid or test.")
    if (
        isinstance(target_offset_hours, bool)
        or not isinstance(target_offset_hours, int)
        or target_offset_hours < 0
    ):
        raise ValueError("target_offset_hours must be a non-negative integer.")
    if {target, "split"}.intersection(model.feature_columns):
        raise ValueError("The target and split must not be model inputs.")
    if not isinstance(dataset.index, pd.DatetimeIndex) or not (
        dataset.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Dataset must have a sorted, unique, complete hourly grid.")
    if target_offset_hours not in {0, 168}:
        raise ValueError("Weekly evaluation supports target offsets of 0 or 168 hours.")
    offset = pd.Timedelta(hours=target_offset_hours)
    hour = pd.Timedelta(hours=1)
    week = pd.Timedelta(hours=168)
    label_time = dataset.index + offset
    X = dataset.loc[:, list(model.feature_columns)]
    complete_features = X.notna().all(axis=1)
    complete = dataset[target].notna() & complete_features
    selected = dataset["split"].eq(split)
    if not selected.any():
        raise ValueError("The experiment needs rows in the evaluation split.")
    forecasts = []
    first_target = label_time[selected][0]
    last_target = label_time[selected][-1]
    for start in pd.date_range(first_target, last_target, freq=week):
        end = start + week - hour
        window = selected & (label_time >= start) & (label_time <= end)
        if window.sum() != 168 or not complete.loc[window].all():
            continue
        origin = start - hour
        # Include earlier validation/test labels once they are observed.
        train = complete & (label_time <= origin)
        if not train.any():
            raise ValueError("The experiment needs complete training rows at each origin.")
        model.fit(X.loc[train], dataset.loc[train, target])
        context = X.loc[: dataset.index[window][-1]].dropna()
        predicted = model.predict(context).reindex(dataset.index[window])
        if predicted.isna().any():
            raise ValueError("Insufficient history for some evaluation rows.")
        forecast = dataset.loc[window, [target]].rename(columns={target: "y"})
        forecast["y_pred"] = predicted
        forecast.insert(0, "target_time (UTC)", label_time[window])
        forecast["forecast_origin (UTC)"] = origin
        forecast["horizon_hours"] = range(1, 169)
        forecast["n_train"] = int(train.sum())
        forecasts.append(forecast)
    if not forecasts:
        raise ValueError("The experiment needs at least one complete 168-hour window.")
    predictions = pd.concat(forecasts)
    predictions.index.name = "feature_time (UTC)"
    metrics = {
        "split": split,
        "evaluation_protocol": "weekly_expanding",
        "n_train": int(predictions["n_train"].iloc[0]),
        "n_train_last": int(predictions["n_train"].iloc[-1]),
        "n_origins": len(forecasts),
        "n_predictions": len(predictions),
        "n_excluded_rows": int(selected.sum()) - len(predictions),
        **evaluate(predictions),
    }
    return predictions, metrics
