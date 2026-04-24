import datetime as dt

import polars as pl

from era_pl import get_annc_dates, get_event_dates, get_trading_dates
import era_pl.events as events_mod


def test_trading_and_annc_dates_polars():
    dsi = pl.DataFrame({
        "date": [dt.date(2020, 1, 2), dt.date(2020, 1, 3), dt.date(2020, 1, 6)],
    }).lazy()

    trading = get_trading_dates(dsi).collect()
    annc = get_annc_dates(get_trading_dates(dsi)).collect()

    assert trading["td"].to_list() == [1, 2, 3]
    assert trading["td"].dtype == pl.Int32
    assert annc.height == 5


def test_get_trading_dates_loads_dsi_when_omitted(monkeypatch):
    source = pl.DataFrame({
        "date": [dt.date(2020, 1, 3), dt.date(2020, 1, 2)],
    }).lazy()

    def fake_load_parquet(table, schema):
        assert table == "dsi"
        assert schema == "crsp"
        return source

    monkeypatch.setattr(events_mod, "load_parquet", fake_load_parquet)

    out = get_trading_dates().collect()
    assert out["date"].to_list() == [dt.date(2020, 1, 2), dt.date(2020, 1, 3)]
    assert out["td"].to_list() == [1, 2]
    assert out["td"].dtype == pl.Int32


def test_get_annc_dates_loads_trading_dates_when_omitted(monkeypatch):
    trading_dates = pl.DataFrame({
        "date": [dt.date(2020, 1, 2), dt.date(2020, 1, 6)],
        "td": [1, 2],
    }).lazy()

    monkeypatch.setattr(events_mod, "get_trading_dates", lambda: trading_dates)

    out = get_annc_dates().collect()
    assert out["annc_date"].to_list() == [
        dt.date(2020, 1, 2),
        dt.date(2020, 1, 3),
        dt.date(2020, 1, 4),
        dt.date(2020, 1, 5),
        dt.date(2020, 1, 6),
    ]
    assert out["td"].to_list() == [1, 2, 2, 2, 2]


def test_get_event_dates_polars_defaults_end_date():
    dsi = pl.DataFrame({
        "date": [dt.date(2020, 1, 2), dt.date(2020, 1, 3), dt.date(2020, 1, 6)],
    }).lazy()
    trading = get_trading_dates(dsi)

    events = pl.DataFrame({
        "permno": [10001],
        "event_date": [dt.date(2020, 1, 4)],
    }).lazy()

    out = get_event_dates(events, trading).collect()
    assert out.height == 1
    assert out["end_event_date"][0] == dt.date(2020, 1, 4)
