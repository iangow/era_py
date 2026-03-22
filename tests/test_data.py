import pandas as pd
import polars as pl

from era_py import (
    available_data,
    get_ff_daily_factors,
    get_ff_ind,
    get_me_breakpoints,
    get_size_rets_monthly,
    load_data,
)


def test_available_data_smoke():
    names = available_data()
    assert "camp_attendance" in names


def test_load_data_smoke():
    df = load_data("camp_attendance")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_iliev_2010_integer_columns():
    df = load_data("iliev_2010")
    assert str(df["pfyear"].dtype) == "Int32"
    assert str(df["cik"].dtype) == "Int32"


def test_ff_ind_wrapper_returns_pandas(monkeypatch):
    import era_pl

    expected = pl.DataFrame({"ff_ind": [5], "sic_min": [100], "sic_max": [199]})
    monkeypatch.setattr(era_pl, "get_ff_ind", lambda ind, timeout=30.0: expected)

    df = get_ff_ind(5)

    assert isinstance(df, pd.DataFrame)
    assert df.to_dict("list") == expected.to_pandas().to_dict("list")


def test_ff_daily_wrapper_returns_pandas(monkeypatch):
    import era_pl

    expected = pl.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "Mkt-RF": [1.0],
            "SMB": [0.5],
            "HML": [0.2],
            "RF": [0.01],
        }
    )
    monkeypatch.setattr(
        era_pl,
        "get_ff_daily_factors",
        lambda dataset="F-F_Research_Data_Factors_daily", start=None, end=None, timeout=30.0: expected,
    )

    df = get_ff_daily_factors(start="2024-01-01", end="2024-01-31")

    assert isinstance(df, pd.DataFrame)
    assert df.to_dict("records") == expected.to_pandas().to_dict("records")


def test_size_rets_wrapper_returns_pandas(monkeypatch):
    import era_pl

    expected = pl.DataFrame({"month": [pd.Timestamp("2024-01-01")], "decile": [1], "ew_ret": [0.01], "vw_ret": [0.02]})
    monkeypatch.setattr(era_pl, "get_size_rets_monthly", lambda timeout=30.0: expected.lazy())

    df = get_size_rets_monthly()

    assert isinstance(df, pd.DataFrame)
    assert df.to_dict("records") == expected.to_pandas().to_dict("records")


def test_me_breakpoints_wrapper_returns_pandas(monkeypatch):
    import era_pl

    expected = pl.DataFrame({"month": [pd.Timestamp("2024-01-01")], "decile": [1], "me_min": [0.0], "me_max": [10.0]})
    monkeypatch.setattr(era_pl, "get_me_breakpoints", lambda timeout=30.0: expected.lazy())

    df = get_me_breakpoints()

    assert isinstance(df, pd.DataFrame)
    assert df.to_dict("records") == expected.to_pandas().to_dict("records")
