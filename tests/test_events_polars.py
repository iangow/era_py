import datetime as dt

import polars as pl

from era_pl import get_annc_dates, get_event_dates, get_trading_dates


def test_trading_and_annc_dates_polars():
    dsi = pl.DataFrame({
        "date": [dt.date(2020, 1, 2), dt.date(2020, 1, 3), dt.date(2020, 1, 6)],
    }).lazy()

    trading = get_trading_dates(dsi).collect()
    annc = get_annc_dates(get_trading_dates(dsi)).collect()

    assert trading["td"].to_list() == [1, 2, 3]
    assert annc.height == 5


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
