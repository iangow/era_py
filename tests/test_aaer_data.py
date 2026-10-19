import datetime as dt

import polars as pl
from polars.testing import assert_frame_equal

from era_pl import data_info, load_data
from era_py import load_data as pandas_load_data


def test_aaer_dates_preserve_verified_dates_and_distinct_releases():
    df = load_data("aaer_dates")
    assert df.shape == (3386, 4)
    assert df.schema == {
        "aaer_num": pl.String, "aaer_date": pl.Date,
        "aaer_desc": pl.String, "year": pl.Int32,
    }
    assert df.is_duplicated().sum() == 0
    expected = {
        "AAER-1209": dt.date(1999, 11, 16),
        "AAER-1440": dt.date(2001, 9, 6),
        "AAER-1470": dt.date(2001, 10, 23),
        "AAER-1813": dt.date(2003, 7, 14),
        "AAER-1839": dt.date(2003, 8, 13),
        "AAER-2216": dt.date(2005, 3, 21),
        "AAER-2218": dt.date(2005, 3, 24),
    }
    for number, date in expected.items():
        assert df.filter(pl.col("aaer_num") == number)["aaer_date"].to_list() == [date]
    for number in ["AAER-1670", "AAER-3448"]:
        assert df.filter(pl.col("aaer_num") == number).height == 1
    assert df.filter(pl.col("aaer_num") == "AAER-2215").height == 2
    assert "AAER-1271" in df["aaer_num"]
    assert "AAER-x2218" not in df["aaer_num"]
    assert set(df.filter(pl.col("aaer_date").is_null())["aaer_num"]) == {
        "AAER-2493", "AAER-2571", "AAER-2579",
    }
    details = data_info("aaer_dates")["metadata"]["documentation"]["details"]
    assert "AAER-1470" in details
    assert "November 16, 1999" in details


def test_aaer_dates_match_in_both_python_namespaces():
    expected = load_data("aaer_dates")
    actual = pl.from_pandas(pandas_load_data("aaer_dates")).with_columns(
        pl.col("aaer_date").cast(pl.Date)
    )
    assert_frame_equal(actual, expected)
