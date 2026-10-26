"""Check local calendar dates and label alignment in both datasets."""

import runpy
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(
    params=["1_prepare_dataset_v1.py", "1_prepare_dataset_target_only_v1.py"]
)
def script(request):
    return runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts" / request.param)
    )


@pytest.mark.parametrize(
    ("label_time", "is_holiday", "is_weekend"),
    [
        ("2024-05-05 10:00", True, True),  # Liberation Day in a non-fifth year.
        ("2025-04-25 21:00", False, False),
        ("2025-04-25 22:00", True, True),  # King's Day moved to Saturday.
        ("2025-04-27 10:00", False, True),
        ("2026-04-03 10:00", True, False),  # Good Friday.
        ("2026-12-24 22:00", False, False),
        ("2026-12-24 23:00", True, False),  # Christmas starts at local midnight.
        ("2026-07-03 21:00", False, False),
        ("2026-07-03 22:00", False, True),  # Summer UTC offset.
        ("2026-07-05 22:00", False, False),
        ("2026-01-02 22:00", False, False),
        ("2026-01-02 23:00", False, True),  # Winter UTC offset.
    ],
)
def test_flags_use_local_label_date(script, label_time, is_holiday, is_weekend):
    index = pd.DatetimeIndex([pd.Timestamp(label_time)]) - pd.Timedelta(
        hours=script.get("HORIZON", 0)
    )
    raw = pd.DataFrame(
        {column: [1.0] for column in [*script.get("FEATURES", []), script["TARGET"]]},
        index=index,
    )

    result = script["prepare_dataset"](raw)

    assert bool(result["is_holiday"].iloc[0]) == is_holiday
    assert bool(result["is_weekend"].iloc[0]) == is_weekend
    assert result.index.equals(raw.index)
    if "HORIZON" in script:
        assert result["y"].isna().all()
