from __future__ import annotations

import pandas as pd
import polars as pl

from era_py.plots import (
    plotnine_star,
    rdplot as _rdplot,
    spline_smooth as _spline_smooth,
)


def _to_pandas_frame(data):
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def spline_smooth(
    data: pl.DataFrame | pd.DataFrame,
    *,
    x: str,
    y: str,
    df: int = 6,
    n: int = 200,
) -> pd.DataFrame:
    return _spline_smooth(_to_pandas_frame(data), x=x, y=y, df=df, n=n)


def rdplot(
    data: pl.DataFrame | pd.DataFrame,
    *,
    y: str,
    x: str,
    cutoff: float,
    y_label: str,
    x_label: str,
    degree: int = 4,
    title: str = "RD Plot",
    bins: int | tuple[int, int] | None = None,
    binselect: str = "esmv",
    masspoints: str = "adjust",
    use_rdrobust: bool = True,
):
    return _rdplot(
        _to_pandas_frame(data),
        y=y,
        x=x,
        cutoff=cutoff,
        y_label=y_label,
        x_label=x_label,
        degree=degree,
        title=title,
        bins=bins,
        binselect=binselect,
        masspoints=masspoints,
        use_rdrobust=use_rdrobust,
    )
