import polars as pl

from era_pl import available_data, load_data
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
