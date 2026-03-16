import polars as pl

from era_pl import available_data, load_data


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
