import polars as pl

from era_pl import available_data, load_data


def test_available_data_smoke_polars():
    names = available_data()
    assert "camp_attendance" in names


def test_load_data_smoke_polars():
    df = load_data("camp_attendance")
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
