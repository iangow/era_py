from __future__ import annotations

import io
import json
import tempfile
import zipfile
from contextlib import contextmanager
from importlib.resources import files
from typing import Any, Iterator

import pandas as pd
import polars as pl
import pyreadr
import requests


def available_data() -> list[str]:
    data_dir = files("era_pl").joinpath("_data")
    return sorted(
        item.name.removesuffix(".parquet")
        for item in data_dir.iterdir()
        if item.name.endswith(".parquet")
    )


def _restore_types(df: pl.DataFrame, name: str) -> pl.DataFrame:
    meta_file = files("era_pl").joinpath("_data", f"{name}.meta.json")
    if not meta_file.is_file():
        return df

    with meta_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    factor_levels = metadata.get("factor_levels", {})
    if isinstance(factor_levels, dict):
        for col in factor_levels:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Categorical))

    original_classes = metadata.get("original_classes", {})
    if isinstance(original_classes, dict):
        for col, r_class in original_classes.items():
            if col not in df.columns:
                continue
            class_names = r_class if isinstance(r_class, list) else [r_class]
            if "Date" in class_names:
                df = df.with_columns(
                    pl.col(col).cast(pl.Date, strict=False)
                )
            if "POSIXct" in class_names:
                df = df.with_columns(
                    pl.col(col).str.to_datetime(time_zone="UTC", strict=False)
                )

    return df


def load_data(name: str, *, restore_categories: bool = True) -> pl.DataFrame:
    data_file = files("era_pl").joinpath("_data", f"{name}.parquet")
    if not data_file.is_file():
        available = ", ".join(available_data())
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")

    with data_file.open("rb") as f:
        try:
            df = pl.read_parquet(f)
        except pl.exceptions.ComputeError as err:
            if "invalid UTF-8" not in str(err):
                raise
            f.seek(0)
            df = pl.read_parquet(f, use_pyarrow=True)

    if restore_categories:
        df = _restore_types(df, name)

    return df


def load_farr_rda(name: str, *, timeout: float = 30.0) -> Any:
    urls = [
        f"https://raw.githubusercontent.com/iangow/farr/main/data/{name}.rda",
        f"https://raw.githubusercontent.com/iangow/farr/main/data/{name}.RData",
        f"https://github.com/iangow/farr/raw/refs/heads/main/data/{name}.rda",
        f"https://github.com/iangow/farr/raw/refs/heads/main/data/{name}.RData",
    ]

    resp = None
    last_error = None
    for url in urls:
        try:
            candidate = requests.get(url, timeout=timeout)
            if candidate.status_code == 200:
                resp = candidate
                break
        except requests.RequestException as err:
            last_error = err

    if resp is None:
        if last_error is not None:
            raise last_error
        raise requests.HTTPError(
            f"Could not load '{name}' from farr data. Tried: {', '.join(urls)}"
        )

    suffix = ".RData" if resp.url.endswith(".RData") else ".rda"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(resp.content)
        f.flush()
        data = pyreadr.read_r(f.name)

    if name not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(
            f"Object '{name}' not found inside '{name}.rda'. "
            f"Available objects: {available}"
        )

    obj = data[name]
    if isinstance(obj, pd.DataFrame):
        if "datadate" in obj.columns:
            obj["datadate"] = pd.to_datetime(obj["datadate"])
        return pl.from_pandas(obj)

    return obj


@contextmanager
def _zip_url_to_file(url: str, *, timeout: float = 30.0) -> Iterator[io.StringIO]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError(f"ZIP archive at '{url}' does not contain files")

        with zf.open(names[0]) as raw:
            text = raw.read().decode("latin-1")

    yield io.StringIO(text)


def _zip_url_to_lines(url: str, *, timeout: float = 30.0) -> list[str]:
    with _zip_url_to_file(url, timeout=timeout) as f:
        return f.read().splitlines()


def _lines_to_df(lines: list[str], *, has_header: bool = True) -> pl.DataFrame:
    block = "\n".join(lines)
    df = pl.read_csv(io.StringIO(block), null_values="-99.99", has_header=has_header)
    return df.rename({df.columns[0]: "date"})


def _read_size_data(lines: list[str]) -> pl.LazyFrame:
    return (
        _lines_to_df(lines)
        .lazy()
        .with_columns(
            month=(pl.col("date").cast(pl.String) + "01")
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
        )
        .drop("date")
        .unpivot(index="month", variable_name="quantile", value_name="ret")
        .with_columns(
            ret=pl.col("ret").cast(pl.Float64, strict=False) / 100.0,
            decile=(
                pl.when(pl.col("quantile") == "Hi 10").then(pl.lit(10))
                .when(pl.col("quantile") == "Lo 10").then(pl.lit(1))
                .when(pl.col("quantile").str.contains("-Dec"))
                .then(
                    pl.col("quantile")
                    .str.replace("-Dec", "")
                    .cast(pl.Int32, strict=False)
                )
                .otherwise(None)
            ),
        )
        .filter(pl.col("decile").is_not_null())
        .select("month", "ret", "decile")
        .sort(["month", "decile"])
    )


def get_size_rets_monthly(*, timeout: float = 30.0) -> pl.LazyFrame:
    url = (
        "https://mba.tuck.dartmouth.edu/pages/"
        "faculty/ken.french/ftp/Portfolios_Formed_on_ME_CSV.zip"
    )
    text = _zip_url_to_lines(url, timeout=timeout)

    vw_start = next(i for i, line in enumerate(text) if "Value Weight" in line and "Monthly" in line) + 1
    vw_end = next(i for i, line in enumerate(text) if "Equal Weight" in line and "Monthly" in line)
    ew_start = vw_end + 1
    ew_end = next(i for i, line in enumerate(text) if line.startswith("  Value Weight") and "Annual" in line)

    vw_rets = _read_size_data(text[vw_start:vw_end])
    ew_rets = _read_size_data(text[ew_start:ew_end])

    return (
        ew_rets
        .select("month", "decile", pl.col("ret").alias("ew_ret"))
        .join(
            vw_rets.select("month", "decile", pl.col("ret").alias("vw_ret")),
            on=["month", "decile"],
            how="inner",
        )
        .sort(["month", "decile"])
    )


def get_me_breakpoints(*, timeout: float = 30.0) -> pl.LazyFrame:
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/ME_Breakpoints_CSV.zip"
    text = _zip_url_to_lines(url, timeout=timeout)
    last_line = next(i for i, line in enumerate(text) if line.startswith("Copyright"))

    df_raw = _lines_to_df(text[1:last_line], has_header=False)
    df_raw.columns = ["month", "n"] + [f"p{i}" for i in range(5, 101, 5)]

    breakpoints_raw = (
        df_raw
        .lazy()
        .with_columns(
            month=(pl.col("month").cast(pl.String) + "01")
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
        )
        .select(["month"] + [f"p{i}" for i in range(10, 101, 10)])
        .unpivot(index="month", variable_name="decile", value_name="cutoff")
        .with_columns(
            decile=(pl.col("decile").str.slice(1).cast(pl.Int32, strict=False) / 10).cast(pl.Int32),
            cutoff=pl.col("cutoff").cast(pl.Float64, strict=False),
        )
        .select("month", "decile", "cutoff")
        .sort(["month", "decile"])
    )

    return (
        breakpoints_raw
        .with_columns(
            me_min=pl.col("cutoff").shift(1).over("month").fill_null(0),
            me_max=pl.when(pl.col("decile") == 10).then(float("inf")).otherwise(pl.col("cutoff")),
        )
        .drop("cutoff")
        .sort(["month", "decile"])
    )


def get_ff_ind(ind: str | int, *, timeout: float = 30.0) -> pl.DataFrame:
    ind_str = str(ind)
    url = (
        "http://mba.tuck.dartmouth.edu"
        f"/pages/faculty/ken.french/ftp/Siccodes{ind_str}.zip"
    )

    with _zip_url_to_file(url, timeout=timeout) as f:
        df = (
            pl.from_pandas(
                pd.read_fwf(
                    f,
                    widths=[3, 7, 1000],
                    names=["ff_ind", "ff_ind_short_desc", "temp"],
                )
            )
            .with_columns(
                ff_ind_desc=pl.when(pl.col("ff_ind").is_not_null())
                .then(pl.col("temp"))
                .otherwise(None),
                sic_range=pl.when(pl.col("ff_ind").is_null())
                .then(pl.col("temp"))
                .otherwise(None),
            )
            .with_columns(
                pl.col("ff_ind").forward_fill(),
                pl.col("ff_ind_short_desc").forward_fill(),
                pl.col("ff_ind_desc").forward_fill(),
            )
            .filter(pl.col("sic_range").is_not_null())
            .with_columns(
                pl.col("sic_range")
                .str.extract_groups(
                    r"^(?P<sic_min>[0-9]+)-"
                    r"(?P<sic_max>[0-9]+)\s*"
                    r"(?P<sic_desc>.*)$"
                )
                .alias("sic_parts")
            )
            .unnest("sic_parts")
            .with_columns(
                pl.col("ff_ind").cast(pl.Int32),
                pl.col("sic_min").cast(pl.Int32),
                pl.col("sic_max").cast(pl.Int32),
            )
            .drop("sic_range", "temp")
        )
    return df
