from __future__ import annotations

import json
import tempfile
from importlib.resources import files
from typing import Any

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
        df = pl.read_parquet(f)

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
