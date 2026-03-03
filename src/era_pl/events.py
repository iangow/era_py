from __future__ import annotations

import os

import polars as pl
import polars.selectors as cs


def cast_decimals(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return frame.with_columns(cs.decimal().cast(pl.Float64))


def load_parquet(table, schema, data_dir=None):
    if not data_dir:
        data_dir = os.path.expanduser(os.environ["DATA_DIR"])
    path = os.path.join(data_dir, schema, f"{table}.parquet")
    return pl.scan_parquet(path).with_columns(cs.decimal().cast(pl.Float64))


def get_trading_dates(dsi):
    if isinstance(dsi, pl.DataFrame):
        dsi = dsi.lazy()
    return (
        dsi
        .select("date")
        .sort("date")
        .with_row_index("td", offset=1)
        .with_columns(td=pl.col("td").cast(pl.Int64))
    )


def get_annc_dates(trading_dates):
    trading_dates_df = trading_dates.collect()
    min_date = trading_dates_df["date"].min()
    max_date = trading_dates_df["date"].max()

    return (
        pl.DataFrame({
            "annc_date": pl.date_range(min_date, max_date, interval="1d", eager=True)
        })
        .join(trading_dates_df, left_on="annc_date", right_on="date", how="left")
        .sort("annc_date")
        .with_columns(td=pl.col("td").backward_fill())
        .with_columns(td=pl.col("td").cast(pl.Int64))
        .lazy()
    )


def get_event_dates(events, trading_dates, win_start=-1, win_end=1):
    annc_dates = get_annc_dates(trading_dates)

    if isinstance(events, pl.DataFrame):
        events = events.lazy()

    events = events.with_columns(
        end_event_date=pl.coalesce([pl.col("end_event_date"), pl.col("event_date")])
        if "end_event_date" in events.collect_schema().names()
        else pl.col("event_date")
    )

    return (
        events
        .join(
            annc_dates.select("annc_date", pl.col("td").alias("start_event_td")),
            left_on="event_date",
            right_on="annc_date",
            how="left",
        )
        .join(
            annc_dates.select("annc_date", pl.col("td").alias("end_event_td")),
            left_on="end_event_date",
            right_on="annc_date",
            how="left",
        )
        .with_columns(
            start_td=pl.col("start_event_td") + win_start,
            end_td=pl.col("end_event_td") + win_end,
        )
        .join(
            trading_dates.select(pl.col("td").alias("start_td"), pl.col("date").alias("start_date")),
            on="start_td",
            how="left",
        )
        .join(
            trading_dates.select(pl.col("td").alias("end_td"), pl.col("date").alias("end_date")),
            on="end_td",
            how="left",
        )
        .select("permno", "event_date", "end_event_date", "start_td", "end_td", "start_date", "end_date")
    )


def copy_inline(df, con):
    raise NotImplementedError("copy_inline is not implemented in era_pl.")


def get_event_rets(*args, **kwargs):
    raise NotImplementedError("get_event_rets is not implemented in era_pl yet.")
