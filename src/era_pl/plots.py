from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from patsy import dmatrix


def _to_pandas_frame(data):
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def spline_smooth(data, *, x: str, y: str, df: int = 6, n: int = 200) -> pl.DataFrame:
    d = _to_pandas_frame(data)[[x, y]].dropna()

    X = dmatrix(f"cr({x}, df={df})", data=d, return_type="dataframe")
    fit = sm.OLS(d[y].to_numpy(), X.to_numpy()).fit()

    grid = pd.DataFrame({x: np.linspace(d[x].min(), d[x].max(), n)})
    Xg = dmatrix(f"cr({x}, df={df})", data=grid, return_type="dataframe")
    grid[f"{y}_smooth"] = fit.predict(Xg.to_numpy())

    return pl.from_pandas(grid)
