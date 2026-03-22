from __future__ import annotations

import json
import tempfile
from typing import Any
from importlib.resources import files

import requests
import pandas as pd

_DATA_PACKAGE = "era_data"


def available_data() -> list[str]:
    """Return packaged dataset names available via load_data()."""
    data_dir = files(_DATA_PACKAGE).joinpath("_data")
    return sorted(
        item.name.removesuffix(".parquet")
        for item in data_dir.iterdir()
        if item.name.endswith(".parquet")
    )


def load_data(name: str, *, restore_categories: bool = True) -> pd.DataFrame:
    """
    Load a packaged dataset by name from the shared packaged data directory.

    Parameters
    ----------
    name:
        Dataset name, e.g. "zhang_2007_windows".
    restore_categories:
        Restore factor columns as pandas Categorical using sidecar metadata.
    """
    data_file = files(_DATA_PACKAGE).joinpath("_data", f"{name}.parquet")
    if not data_file.is_file():
        available = ", ".join(available_data())
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")

    with data_file.open("rb") as f:
        df = pd.read_parquet(f)

    if restore_categories:
        meta_file = files(_DATA_PACKAGE).joinpath("_data", f"{name}.meta.json")
        if meta_file.is_file():
            with meta_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            factor_levels = metadata.get("factor_levels", {})
            if isinstance(factor_levels, dict):
                for col, levels in factor_levels.items():
                    if col in df.columns:
                        df[col] = pd.Categorical(df[col], categories=levels)
            original_classes = metadata.get("original_classes", {})
            if isinstance(original_classes, dict):
                for col, r_class in original_classes.items():
                    if col not in df.columns:
                        continue
                    class_names = r_class if isinstance(r_class, list) else [r_class]
                    if "integer" in class_names:
                        df[col] = df[col].astype("Int32")
                    if "Date" in class_names:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    if "POSIXct" in class_names:
                        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return df


def load_farr_rda(name: str, *, timeout: float = 30.0) -> Any:
    """
    Load an .rda dataset named `name` from the farr GitHub repo and return the object.

    Parameters
    ----------
    name:
        Dataset/object name (and file stem), e.g. "camp_scores".
    timeout:
        Requests timeout in seconds.

    Returns
    -------
    The R object stored in the .rda under key `name`.
    """
    import pyreadr

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

    df = data[name]

    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"])

    return df


def get_ff_ind(ind: str | int, *, timeout: float = 30.0) -> pd.DataFrame:
    """Download Fama-French SIC industry definitions as a pandas DataFrame."""
    from era_pl import get_ff_ind as _get_ff_ind

    return _get_ff_ind(ind, timeout=timeout).to_pandas()


def get_ff_daily_factors(
    dataset: str = "F-F_Research_Data_Factors_daily",
    *,
    start: str | None = None,
    end: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Download Ken French daily factor data as a pandas DataFrame."""
    from era_pl import get_ff_daily_factors as _get_ff_daily_factors

    return _get_ff_daily_factors(
        dataset=dataset,
        start=start,
        end=end,
        timeout=timeout,
    ).to_pandas()


def get_size_rets_monthly(*, timeout: float = 30.0) -> pd.DataFrame:
    """Download Ken French monthly size-portfolio returns as a pandas DataFrame."""
    from era_pl import get_size_rets_monthly as _get_size_rets_monthly

    return _get_size_rets_monthly(timeout=timeout).collect().to_pandas()


def get_me_breakpoints(*, timeout: float = 30.0) -> pd.DataFrame:
    """Download Ken French market-equity breakpoints as a pandas DataFrame."""
    from era_pl import get_me_breakpoints as _get_me_breakpoints

    return _get_me_breakpoints(timeout=timeout).collect().to_pandas()
