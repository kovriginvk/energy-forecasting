"""Check target-only alignment and availability for weekly forecasts."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "1_prepare_dataset_target_only_v1.py"
    )
)
prepare_dataset = SCRIPT["prepare_dataset"]
TARGET = SCRIPT["TARGET"]


@pytest.fixture
def raw():
    return pd.DataFrame(
        {TARGET: np.arange(1200, dtype=float)},
        index=pd.date_range(
            "2026-06-01", periods=1200, freq="h", name="validfrom (UTC)"
        ),
    )


def test_shifted_target_lags_and_missing_edges(raw):
    original = raw.copy()
    result = prepare_dataset(raw)

    assert result.index.equals(raw.index)
    pd.testing.assert_series_equal(result[TARGET], raw[TARGET])
    np.testing.assert_array_equal(result["y"].iloc[:-168], raw[TARGET].iloc[168:])
    assert result["y"].iloc[-168:].isna().all()
    for lag in SCRIPT["LAGS"]:
        assert result[f"target_lag_{lag}"].iloc[:lag].isna().all()
        np.testing.assert_array_equal(
            result[f"target_lag_{lag}"].iloc[lag:], raw[TARGET].iloc[:-lag]
        )
    pd.testing.assert_frame_equal(raw, original)


def test_weekly_block_uses_only_history_available_at_origin(raw):
    result = prepare_dataset(raw)
    origin = 900
    block = result.iloc[origin - 167 : origin + 1]
    np.testing.assert_array_equal(
        block["y"], raw[TARGET].iloc[origin + 1 : origin + 169]
    )

    changed = raw.copy()
    changed.iloc[origin + 1 :, 0] = -999
    changed_block = prepare_dataset(changed).iloc[origin - 167 : origin + 1]
    features = [TARGET, *(f"target_lag_{lag}" for lag in SCRIPT["LAGS"])]
    pd.testing.assert_frame_equal(block[features], changed_block[features])


@pytest.mark.parametrize(
    ("boundary", "before", "after"),
    [("2026-07-01", "train", "valid"), ("2026-09-01", "valid", "test")],
)
def test_split_uses_future_label_time(raw, boundary, before, after):
    raw.index = pd.date_range(
        start=pd.Timestamp(boundary) - pd.Timedelta(hours=169),
        periods=len(raw),
        freq="h",
    )
    result = prepare_dataset(raw)
    assert result["split"].iloc[0] == before
    assert result["split"].iloc[1:].eq(after).all()


def test_rejects_missing_hours(raw):
    with pytest.raises(ValueError, match="hourly grid"):
        prepare_dataset(raw.drop(raw.index[100]))
