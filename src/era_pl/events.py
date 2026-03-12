from __future__ import annotations

import os

from dotenv import load_dotenv

import polars as pl
import polars.selectors as cs


def cast_decimals(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return frame.with_columns(cs.decimal().cast(pl.Float64))


def load_parquet(table, schema, data_dir=None):
    if not data_dir:
        load_dotenv()
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


def get_event_dates(
    data,
    trading_dates=None,
    permno="permno",
    event_date="event_date",
    win_start=0,
    win_end=0,
    end_event_date=None,
    data_dir=None,
):
    if isinstance(trading_dates, str):
        permno = trading_dates
        trading_dates = None

    if trading_dates is None:
        trading_dates = get_trading_dates(load_parquet("dsi", schema="crsp", data_dir=data_dir))
    elif isinstance(trading_dates, pl.DataFrame):
        trading_dates = trading_dates.lazy()

    annc_dates = get_annc_dates(trading_dates)

    if isinstance(data, pl.DataFrame):
        data = data.lazy()

    if end_event_date is None:
        events = data.select(
            pl.col(permno).alias("permno"),
            pl.col(event_date).alias("event_date"),
        ).with_columns(
            pl.col("event_date").cast(pl.Date),
            pl.col("event_date").cast(pl.Date).alias("end_event_date"),
        )
    else:
        events = data.select(
            pl.col(permno).alias("permno"),
            pl.col(event_date).alias("event_date"),
            pl.col(end_event_date).alias("end_event_date"),
        ).with_columns(
            pl.col("event_date").cast(pl.Date),
            pl.coalesce([pl.col("end_event_date"), pl.col("event_date")])
            .cast(pl.Date)
            .alias("end_event_date"),
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
