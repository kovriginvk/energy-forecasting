"""Fit once and predict a chronological dataset split."""

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
    """Score each row's target, retaining preceding prediction context.

    The offset describes label time relative to the row index, not the number
    of forecast horizons. Future generation inputs assume perfect forecasts.
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
        dataset.index.is_monotonic_increasing and dataset.index.is_unique
    ):
        raise ValueError("Dataset must have a sorted, unique datetime index.")
    offset = pd.Timedelta(hours=target_offset_hours)
    X = dataset.loc[:, list(model.feature_columns)]
    complete_features = X.notna().all(axis=1)
    train = dataset["split"].eq("train") & dataset[target].notna() & complete_features
    selected = dataset["split"].eq(split) & dataset[target].notna() & complete_features
    if selected.any():
        # Training labels must already exist at the first evaluation row time.
        train &= dataset.index + offset <= dataset.index[selected][0]
    if not train.any() or not selected.any():
        raise ValueError("The experiment needs complete training and evaluation rows.")
    model.fit(X.loc[train], dataset.loc[train, target])
    context = X.loc[: dataset.index[selected][-1]].dropna()
    predicted = model.predict(context).reindex(dataset.index)
    if predicted.loc[selected].isna().any():
        raise ValueError("Insufficient history for some evaluation rows.")
    predictions = dataset.loc[selected, [target]].rename(columns={target: "y"})
    predictions["y_pred"] = predicted.loc[selected]
    predictions.index.name = "feature_time (UTC)"
    predictions.insert(0, "target_time (UTC)", predictions.index + offset)
    metrics = {
        "split": split,
        "n_train": int(train.sum()),
        "n_predictions": len(predictions),
        **evaluate(predictions),
    }
    return predictions, metrics
