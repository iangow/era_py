import pandas as pd

from era_py import available_data, load_data


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
