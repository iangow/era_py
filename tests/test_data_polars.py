import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from era_pl import available_data, get_idd_periods, load_data
from era_pl.data import _restore_types


def test_available_data_smoke_polars():
    names = available_data()
    assert "camp_attendance" in names


def test_load_data_smoke_polars():
    df = load_data("camp_attendance")
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0


def test_iliev_2010_integer_columns_polars():
    df = load_data("iliev_2010")
    assert df.schema["pfyear"] == pl.Int32
    assert df.schema["cik"] == pl.Int32


def test_cmsw_2018_logical_columns_polars():
    df = load_data("cmsw_2018")
    assert df.schema["selfdealflag"] == pl.Boolean
    assert df.schema["wbflag"] == pl.Boolean
    assert df.schema["tousesox"] == pl.Boolean


def test_restore_types_casts_logical_columns_polars():
    df = pl.DataFrame({"selfdealflag": [0, 1, None]}, schema={"selfdealflag": pl.Int8})

    restored = _restore_types(df, "cmsw_2018")

    assert restored.schema["selfdealflag"] == pl.Boolean


def test_get_idd_periods_default_state_universe_matches_farr():
    periods = get_idd_periods("1994-01-01", "2010-12-31")

    assert periods.height == 65
    assert periods.select("state").unique().height == 51
    assert periods.group_by("period_type").len().sort("period_type").to_dicts() == [
        {"period_type": "Post-adoption", "len": 21},
        {"period_type": "Post-rejection", "len": 3},
        {"period_type": "Pre-adoption", "len": 41},
    ]


def test_restore_date_strings_and_typed_dates(monkeypatch, tmp_path):
    import datetime as dt
    import json
    import era_pl.data as data_mod

    folder = tmp_path / "_data"
    folder.mkdir()
    (folder / "dates.meta.json").write_text(json.dumps({
        "original_classes": {"day": ["Date"]},
    }))
    monkeypatch.setattr(data_mod, "files", lambda package: tmp_path)
    strings = pl.DataFrame({"day": ["2020-01-02", None, "invalid"]})
    restored = _restore_types(strings, "dates")
    assert restored["day"].to_list() == [dt.date(2020, 1, 2), None, None]
    assert _restore_types(restored, "dates").equals(restored)


def test_ff_daily_date_bounds_accept_strings_and_dates(monkeypatch):
    import datetime as dt
    import era_pl.data as data_mod

    monkeypatch.setattr(data_mod, "_zip_url_to_lines", lambda *a, **k: [
        ",Mkt-RF,SMB,HML,RF", "20200102,1,2,3,0", "20200103,4,5,6,0",
    ])
    strings = data_mod.get_ff_daily_factors(start="2020-01-03", end="2020-01-03")
    dates = data_mod.get_ff_daily_factors(start=dt.date(2020, 1, 3), end=dt.date(2020, 1, 3))
    assert strings.equals(dates)
    assert strings["date"].to_list() == [dt.date(2020, 1, 3)]
    for bound in ["2020-01-03", dt.date(2020, 1, 3), dt.datetime(2020, 1, 3, 15)]:
        expressions = data_mod.get_ff_daily_factors(start=pl.lit(bound), end=pl.lit(bound))
        assert_frame_equal(expressions, strings)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("1994-01-01", "2010-12-31"),
        (dt.date(1994, 1, 1), dt.date(2010, 12, 31)),
        (dt.datetime(1994, 1, 1, 15), dt.datetime(2010, 12, 31, 15)),
        (
            dt.datetime(1994, 1, 1, 15, tzinfo=dt.timezone.utc),
            dt.datetime(2010, 12, 31, 15, tzinfo=dt.timezone.utc),
        ),
    ],
    ids=["string", "date", "datetime", "timezone-aware-datetime"],
)
def test_idd_date_expression_bounds_match_literal_bounds(start, end):
    expected = get_idd_periods("1994-01-01", "2010-12-31")
    actual = get_idd_periods(pl.lit(start), pl.lit(end))

    assert_frame_equal(actual, expected)
