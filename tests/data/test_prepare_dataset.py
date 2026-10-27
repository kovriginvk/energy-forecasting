"""Check historical imputation and same-hour target alignment."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "1_prepare_dataset_v1.py")
)
prepare_dataset = SCRIPT["prepare_dataset"]


@pytest.fixture
def raw():
    index = pd.date_range("2026-01-01", periods=800, freq="h")
    return pd.DataFrame(
        {
            column: np.arange(800, dtype=float)
            for column in [*SCRIPT["FEATURES"], SCRIPT["TARGET"]]
        },
        index=index.rename("validfrom (UTC)"),
    )


def test_target_alignment_and_selected_features(raw):
    result = prepare_dataset(raw)

    assert result.index.equals(raw.index)
    assert SCRIPT["TARGET"] not in result.columns
    pd.testing.assert_frame_equal(result[SCRIPT["FEATURES"]], raw[SCRIPT["FEATURES"]])
    assert not result.filter(like="is_imputed").to_numpy().any()
    pd.testing.assert_series_equal(result["y"], raw[SCRIPT["TARGET"]].rename("y"))


def test_imputation_uses_history_and_preserves_observations(raw):
    start = 400
    wind_column = "wind_sea_volume (kWh)"
    cogeneration_column = "cogeneration_volume (kWh)"
    damaged = raw.copy()
    damaged.loc[raw.index[start : start + 25], wind_column] = np.nan
    damaged.loc[raw.index[start : start + 26], cogeneration_column] = np.nan
    original = damaged.copy()

    result = prepare_dataset(damaged)

    np.testing.assert_array_equal(
        result[wind_column].iloc[start : start + 25],
        np.resize(raw[wind_column].iloc[start - 24 : start].to_numpy(), 25),
    )
    np.testing.assert_array_equal(
        result[cogeneration_column].iloc[start : start + 26],
        (
            raw[cogeneration_column].iloc[start - 168 : start - 168 + 26].to_numpy()
            + raw[cogeneration_column].iloc[start - 336 : start - 336 + 26].to_numpy()
        )
        / 2,
    )
    assert result["wind_sea_is_imputed"].sum() == 25
    assert result["cogeneration_is_imputed"].sum() == 26
    assert result["is_imputed"].sum() == 26
    pd.testing.assert_series_equal(
        result["is_imputed"],
        result[
            ["wind_land_is_imputed", "wind_sea_is_imputed", "cogeneration_is_imputed"]
        ]
        .any(axis=1)
        .rename("is_imputed"),
    )
    for column in SCRIPT["FEATURES"]:
        observed = damaged[column].notna()
        pd.testing.assert_series_equal(
            result.loc[observed, column],
            damaged.loc[result.index[observed], column],
        )
    pd.testing.assert_frame_equal(damaged, original)

    damaged.loc[raw.index[start + 26 :], [wind_column, cogeneration_column]] = 999999
    changed_future = prepare_dataset(damaged)
    pd.testing.assert_frame_equal(
        result.iloc[start : start + 26][SCRIPT["FEATURES"]],
        changed_future.iloc[start : start + 26][SCRIPT["FEATURES"]],
    )


def test_repairs_only_known_wind_boundary(raw):
    suspect_hour = pd.Timestamp("2026-10-20 21:00:00")
    raw.index = pd.date_range(
        end=suspect_hour + pd.Timedelta(hours=1),
        periods=len(raw),
        freq="h",
        name=raw.index.name,
    )
    original = raw.copy()

    result = prepare_dataset(raw)

    expected = raw[SCRIPT["FEATURES"]].copy()
    for source in ("wind_land", "wind_sea"):
        column = f"{source}_volume (kWh)"
        expected.loc[suspect_hour, column] = raw.loc[
            suspect_hour - pd.Timedelta(hours=1), column
        ]
        assert result[f"{source}_is_imputed"].sum() == 1
        assert result.loc[suspect_hour, f"{source}_is_imputed"]
    pd.testing.assert_frame_equal(result[SCRIPT["FEATURES"]], expected)
    assert result["is_imputed"].sum() == 1
    assert result.loc[suspect_hour, "is_imputed"]
    assert not result["cogeneration_is_imputed"].any()
    pd.testing.assert_series_equal(result["y"], raw[SCRIPT["TARGET"]].rename("y"))
    pd.testing.assert_frame_equal(raw, original)


def test_rejects_missing_hours_before_positional_imputation(raw):
    with pytest.raises(ValueError, match="hourly grid"):
        prepare_dataset(raw.drop(raw.index[100]))


@pytest.mark.parametrize(
    ("boundary", "before", "after"),
    [
        ("2026-07-01", "train", "valid"),
        ("2026-09-01", "valid", "test"),
    ],
)
def test_chronological_split_boundaries(raw, boundary, before, after):
    raw.index = pd.date_range(
        start=pd.Timestamp(boundary) - pd.Timedelta(hours=1),
        periods=len(raw),
        freq="h",
        name=raw.index.name,
    )

    result = prepare_dataset(raw)

    assert result["split"].iloc[0] == before
    assert result["split"].iloc[1:].eq(after).all()
    assert result.index.equals(raw.index)
