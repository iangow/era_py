from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd

import ibis
import ibis.backends.postgres
import ibis.selectors as s
from ibis import _
import os

@ibis.udf.scalar.builtin
def generate_series(a, b, c) -> ibis.dtype("date"):
    ...


@ibis.udf.scalar.builtin
def unnest(a) -> ibis.dtype("date"):
    ...


def load_parquet(con, table, schema, data_dir=None):
    if not data_dir:
        data_dir = os.path.expanduser(os.environ["DATA_DIR"])
    df = con.read_parquet(os.path.join(data_dir, schema, f"{table}.parquet"))
    return cast_decimals(df)


def cast_decimals(t):
    cols = {col: "float64" for col, dtype in t.schema().items()
            if dtype.is_decimal()}
    return t.cast(cols) if cols else t


def get_trading_dates(conn):
    if isinstance(conn, ibis.backends.postgres.Backend):
        dsi = conn.table("dsi", database="crsp")
    else:
        dsi = load_parquet(conn, "dsi", schema="crsp")

    trading_dates = dsi.select("date").mutate(td=1 + ibis.row_number().over(order_by="date"))
    return trading_dates


def get_annc_dates(db):
    win = ibis.window(order_by=ibis.desc(_.annc_date), following=0)

    trading_dates = get_trading_dates(db)

    if isinstance(db, ibis.backends.postgres.Backend):
        annc_dates = (
            trading_dates
            .aggregate(min_date=_.date.min(), max_date=_.date.max() + ibis.interval(days=1))
            .mutate(
                annc_date=generate_series(
                    _.min_date,
                    _.max_date,
                    ibis.interval(days=1),
                )
            )
            .select("annc_date")
            .left_join(trading_dates, _.annc_date == trading_dates.date)
            .mutate(
                td=_.td.min().over(win),
                date=_.date.min().over(win),
            )
            .order_by(_.annc_date)
        )
    else:
        annc_dates = (
            trading_dates
            .aggregate(min_date=_.date.min(), max_date=_.date.max() + ibis.interval(days=1))
            .mutate(
                annc_date=generate_series(
                    _.min_date,
                    _.max_date,
                    ibis.interval(days=1),
                )
            )
            .select("annc_date")
            .mutate(annc_date=unnest(_.annc_date))
            .left_join(trading_dates, _.annc_date == trading_dates.date)
            .mutate(
                td=_.td.min().over(win),
                date=_.date.min().over(win),
            )
            .order_by(_.annc_date)
        )
    return annc_dates


def compute(df, backend=None):
    if backend is None:
        backend = df._find_backend()

    return backend.create_table(f"_{uuid.uuid4().hex[:8]}", df, overwrite=True)


def copy_inline(df, con):
    def format_val(val, series):
        if pd.isna(val):
            return "NULL"
        dtype = series.dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return f"'{val}'::date"
        if pd.api.types.is_bool_dtype(dtype):
            return "TRUE" if val else "FALSE"
        if pd.api.types.is_string_dtype(dtype) or dtype == object:
            return f"'{str(val).replace(chr(39), chr(39) * 2)}'"
        return str(val)

    def pandas_dtype_to_ibis(dtype):
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "date"
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_integer_dtype(dtype):
            return "int32"
        if pd.api.types.is_float_dtype(dtype):
            return "float64"
        return "string"

    rows = ", ".join(
        "(" + ", ".join(format_val(row[col], df[col]) for col in df.columns) + ")"
        for _, row in df.iterrows()
    )
    cols = ", ".join(f'"{col}"' for col in df.columns)
    schema = {col: pandas_dtype_to_ibis(df[col].dtype) for col in df.columns}
    return con.sql(f"SELECT * FROM (VALUES {rows}) AS t({cols})", schema=schema)


def get_event_dates(
    data,
    conn,
    permno="permno",
    event_date="event_date",
    win_start=0,
    win_end=0,
    end_event_date=None,
):
    trading_dates = get_trading_dates(conn)
    annc_dates = get_annc_dates(conn)

    if not isinstance(data, ibis.expr.types.Table):
        data = ibis.memtable(data)

    if end_event_date is None:
        end_event_date = event_date
        data_local = data.select(permno, event_date).rename(
            permno="permno", event_date="event_date"
        )
        data_local = data_local.mutate(
            event_date=_.event_date.cast("date"),
            end_event_date=_.event_date.cast("date"),
        )
    else:
        data_local = data.select(permno, event_date, end_event_date).rename(
            permno="permno", event_date="event_date", end_event_date="end_event_date"
        )
        data_local = data_local.mutate(
            event_date=_.event_date.cast(ibis.dtype("date")),
            end_event_date=_.end_event_date.cast(ibis.dtype("date")),
        )

    annc_start = annc_dates.select(
        annc_date=_.annc_date,
        td_start=_.td + win_start,
    )

    event_tds = (
        data_local.join(annc_start, data_local.event_date == annc_start.annc_date).drop(
            "annc_date"
        )
    )

    annc_end = annc_dates.select(
        annc_date=_.annc_date,
        td_end=_.td + win_end,
    )

    event_tds = event_tds.join(
        annc_end, event_tds.end_event_date == annc_end.annc_date
    ).drop("annc_date")

    td_start_map = trading_dates.select(td_start=_.td, start_date=_.date)
    td_end_map = trading_dates.select(td_end=_.td, end_date=_.date)

    event_dates = event_tds.join(td_start_map, "td_start").join(td_end_map, "td_end").drop(
        "td_start", "td_end"
    )
    return event_dates


def get_event_rets(
    data,
    conn,
    permno="permno",
    event_date="event_date",
    win_start=0,
    win_end=0,
    end_event_date=None,
    suffix="",
    data_dir=None,
):
    if isinstance(conn, ibis.backends.postgres.Backend):
        dsedelist = conn.table("dsedelist", database="crsp")
        erdport1 = conn.table("erdport1", database="crsp")
        dsf = conn.table("dsf", database="crsp")
        dsi = conn.table("dsi", database="crsp")
    else:
        dsedelist = load_parquet(conn, "dsedelist", schema="crsp", data_dir=data_dir)
        erdport1 = load_parquet(conn, "erdport1", schema="crsp", data_dir=data_dir)
        dsf = load_parquet(conn, "dsf", schema="crsp", data_dir=data_dir)
        dsi = load_parquet(conn, "dsi", schema="crsp", data_dir=data_dir)

    event_dates = get_event_dates(
        data,
        conn=conn,
        permno=permno,
        event_date=event_date,
        win_start=win_start,
        win_end=win_end,
        end_event_date=end_event_date,
    )

    dsedelist = (
        dsedelist.rename(date="dlstdt")
        .select("permno", "date", "dlret")
        .filter(_.dlret.notnull())
    )

    dsf_plus = (
        dsf.join(dsedelist, ["permno", "date"], how="outer")
        .filter(_.ret.notnull() | _.dlret.notnull())
        .mutate(ret=(1 + ibis.coalesce(_.ret, 0)) * (1 + ibis.coalesce(_.dlret, 0)) - 1)
        .select("permno", "date", "ret")
    )

    erdport = erdport1.select("permno", "date", "decret")

    dsf_w_erdport = dsf_plus.join(erdport, ["permno", "date"], how="left").drop(
        "permno_right",
        "date_right",
    )

    rets = (
        dsf_w_erdport.join(dsi, "date", how="left")
        .select("permno", "date", "ret", "decret", "vwretd")
        .pipe(cast_decimals)
    )

    permnos = (
        event_dates.select("permno").distinct().execute()["permno"].tolist()
    )

    event_rets = (
        rets.filter(_.permno.isin(permnos))
        .join(
            event_dates,
            [
                "permno",
                _.date >= event_dates.start_date,
                _.date <= event_dates.end_date,
            ],
        )
        .group_by(["permno", "event_date", "end_event_date"])
        .aggregate(
            ret_raw=(1 + _.ret).log().sum().exp() - 1,
            ret_mkt=(1 + _.ret).log().sum().exp() - (1 + _.vwretd).log().sum().exp(),
            ret_sz=(1 + _.ret).log().sum().exp() - (1 + _.decret).log().sum().exp(),
        )
        .rename({f"{c}{suffix}": c for c in ["ret_raw", "ret_mkt", "ret_sz"]})
        .execute()
    )

    return event_rets
