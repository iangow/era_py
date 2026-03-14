from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import date
from importlib.resources import files
from typing import Any, Iterator

import numpy as np
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


def get_test_scores(
    effect_size: float = 15,
    n_students: int = 1000,
    n_grades: int = 4,
    include_unobservables: bool = False,
    random_assignment: bool = False,
    seed: int | None = None,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    treatment_grade = 7
    sd_e = 5
    sd_talent = 5
    mean_talent = 15
    mean_score = 400

    grades = np.arange(
        treatment_grade - n_grades // 2,
        treatment_grade - n_grades // 2 + n_grades,
    )
    grade_effect_map = {
        1: 50,
        2: 52,
        3: 58,
        4: 76,
        5: 80,
        6: 98,
        7: 103,
        8: 119,
        9: 123,
        10: 131,
        11: 138,
        12: 150,
    }

    ids = np.arange(1, n_students + 1, dtype=np.int32)
    talents = pl.DataFrame(
        {
            "id": ids,
            "talent": rng.normal(loc=mean_talent, scale=sd_talent, size=n_students),
        }
    )

    base = (
        pl.DataFrame(
            {
                "grade": np.repeat(grades, n_students),
                "id": np.tile(ids, len(grades)),
            }
        )
        .join(talents, on="id", how="inner")
        .with_columns(
            grade_effect=pl.col("grade")
            .replace_strict(grade_effect_map, default=None)
            .cast(pl.Float64)
        )
    )

    scores = rng.normal(loc=mean_score, scale=sd_e, size=base.height)
    test_scores_pre = base.with_columns(
        score=pl.Series(scores) + pl.col("talent") + pl.col("grade_effect")
    )

    treatment = (
        test_scores_pre.filter(pl.col("grade") == treatment_grade - 1)
        .select("id", "score")
    )

    x = treatment["score"].to_numpy()
    if random_assignment:
        temp = np.ones_like(x)
    else:
        temp = 1 - (x - x.min()) / (x.max() - x.min())
    treat_score = rng.uniform(size=len(x)) * temp
    treat = treat_score > np.median(treat_score)

    treatment = treatment.with_columns(
        treat=pl.Series(treat.astype(np.int8))
    ).select("id", "treat")

    test_scores = (
        test_scores_pre.join(treatment, on="id", how="inner")
        .with_columns(
            post=(pl.col("grade") >= treatment_grade).cast(pl.Int8),
        )
        .with_columns(
            score=pl.when((pl.col("treat") == 1) & (pl.col("post") == 1))
            .then(pl.col("score") + effect_size)
            .otherwise(pl.col("score"))
        )
        .select("id", "grade", "post", "treat", "score", "talent", "grade_effect")
    )

    if include_unobservables:
        return test_scores
    return test_scores.select("id", "grade", "post", "treat", "score")


def get_idd_periods(
    min_date: date,
    max_date: date,
    all_states: pl.DataFrame,
) -> pl.DataFrame:
    idd_dates = load_data("idd_dates")

    df_pre = (
        idd_dates
        .filter((pl.col("idd_type") == "Adopt") & (pl.col("idd_date") > min_date))
        .with_columns(
            period_type=pl.lit("Pre-adoption"),
            start_date=pl.lit(min_date),
            end_date=pl.col("idd_date"),
        )
        .select("state", "period_type", "start_date", "end_date")
    )

    df_never = (
        all_states
        .select("state")
        .unique()
        .join(idd_dates.select("state").unique(), on="state", how="anti")
        .with_columns(
            period_type=pl.lit("Pre-adoption"),
            start_date=pl.lit(min_date),
            end_date=pl.lit(max_date),
        )
    )

    df_post_adopt = (
        idd_dates
        .sort(["state", "idd_date"])
        .with_columns(
            start_date=pl.max_horizontal("idd_date", pl.lit(min_date)),
            end_date=pl.col("idd_date").shift(-1).over("state").fill_null(max_date),
        )
        .filter(pl.col("idd_type") == "Adopt")
        .with_columns(period_type=pl.lit("Post-adoption"))
        .select("state", "period_type", "start_date", "end_date")
    )

    df_post_reject = (
        idd_dates
        .filter(pl.col("idd_type") == "Reject")
        .with_columns(
            period_type=pl.lit("Post-rejection"),
            start_date=pl.max_horizontal("idd_date", pl.lit(min_date)),
            end_date=pl.lit(max_date),
        )
        .select("state", "period_type", "start_date", "end_date")
    )

    return pl.concat(
        [df_never, df_pre, df_post_adopt, df_post_reject],
        how="vertical",
    ).sort(["state", "start_date"])


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
    df_raw = pl.read_csv(io.StringIO(block), has_header=has_header)

    value_cols = df_raw.columns[1:]
    df_raw = df_raw.with_columns(
        pl.col(value_cols)
        .cast(pl.String)
        .str.strip_chars()
        .replace("-99.99", None)
        .cast(pl.Float64, strict=False)
    )

    return df_raw.rename({df_raw.columns[0]: "date"})


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
            ret=pl.col("ret") / 100.0,
            decile=(
                pl.when(pl.col("quantile") == "Hi 10").then(pl.lit(10))
                .when(pl.col("quantile") == "Lo 10").then(pl.lit(1))
                .when(pl.col("quantile").str.contains(r"(^Dec\s+\d+$)|(\d+-Dec$)"))
                .then(
                    pl.col("quantile")
                    .str.extract(r"(\d+)")
                    .cast(pl.Int32, strict=False)
                )
                .otherwise(None)
            ),
        )
        .filter(pl.col("month").is_not_null(), pl.col("decile").is_not_null())
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
        .filter(pl.col("month").is_not_null())
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
        header_re = re.compile(r"^\s*(\d+)\s+(\S+)\s+(.*\S)\s*$")
        sic_re = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+(.*\S)\s*$")

        rows: list[dict[str, object]] = []
        current_header: tuple[int, str, str] | None = None

        for raw_line in f:
            line = raw_line.rstrip()
            if not line.strip():
                continue

            header_match = header_re.match(line)
            if header_match:
                current_header = (
                    int(header_match.group(1)),
                    header_match.group(2),
                    header_match.group(3),
                )
                continue

            sic_match = sic_re.match(line)
            if sic_match and current_header is not None:
                ff_ind, short_desc, desc = current_header
                rows.append(
                    {
                        "ff_ind": ff_ind,
                        "ff_ind_short_desc": short_desc,
                        "ff_ind_desc": desc,
                        "sic_min": int(sic_match.group(1)),
                        "sic_max": int(sic_match.group(2)),
                        "sic_desc": sic_match.group(3),
                    }
                )

    return pl.DataFrame(
        rows,
        schema={
            "ff_ind": pl.Int32,
            "ff_ind_short_desc": pl.String,
            "ff_ind_desc": pl.String,
            "sic_min": pl.Int32,
            "sic_max": pl.Int32,
            "sic_desc": pl.String,
        },
    )
