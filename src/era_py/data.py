from __future__ import annotations

import tempfile
from typing import Any

import requests
import pyreadr
import pyarrow as pa
import pandas as pd

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
    url = f"https://github.com/iangow/farr/raw/refs/heads/main/data/{name}.rda"

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".rda") as f:
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
    
    # df = df.convert_dtypes(dtype_backend='numpy_nullable')

    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"])

    return df
