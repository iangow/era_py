from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from patsy import dmatrix

from era_py.plots import plotnine_star


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


def _binned_means(
    df: pl.DataFrame,
    *,
    x: str,
    y: str,
    cutoff: float,
    bins: int = 20,
) -> pl.DataFrame:
    def summarize_side(side_df: pl.DataFrame, side: str) -> pl.DataFrame:
        if side_df.height == 0:
            return pl.DataFrame(
                {"x": [], "y": [], "side": []},
                schema={"x": pl.Float64, "y": pl.Float64, "side": pl.String},
            )

        ordered = side_df.sort(x)
        x_vals = ordered[x].to_numpy()
        y_vals = ordered[y].to_numpy()
        groups = np.array_split(np.arange(len(x_vals)), min(bins, len(x_vals)))
        return pl.DataFrame(
            {
                "x": [float(x_vals[g].mean()) for g in groups if len(g) > 0],
                "y": [float(y_vals[g].mean()) for g in groups if len(g) > 0],
                "side": [side for g in groups if len(g) > 0],
            }
        )

    clean = df.select(x, y).drop_nulls()
    left = clean.filter(pl.col(x) < cutoff)
    right = clean.filter(pl.col(x) >= cutoff)
    return pl.concat(
        [
            summarize_side(left, "Left"),
            summarize_side(right, "Right"),
        ],
        how="vertical",
    )


def _side_polyfit(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    degree: int,
    cutoff: float,
) -> np.ndarray:
    centered = x_vals - cutoff
    max_degree = min(degree, max(len(x_vals) - 1, 0))
    return np.polyfit(centered, y_vals, deg=max_degree)


def _side_polypredict(
    coefs: np.ndarray,
    grid: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    return np.polyval(coefs, grid - cutoff)


def rdplot(
    df: pl.DataFrame,
    *,
    y: str,
    x: str,
    cutoff: float,
    y_label: str,
    x_label: str,
    degree: int = 4,
    title: str = "RD Plot",
):
    plotnine = importlib.import_module("plotnine")

    clean = df.select(x, y).drop_nulls()
    bins_df = _binned_means(clean, x=x, y=y, cutoff=cutoff)

    left = clean.filter(pl.col(x) < cutoff)
    right = clean.filter(pl.col(x) >= cutoff)

    x_left = left[x].to_numpy()
    y_left = left[y].to_numpy()
    x_right = right[x].to_numpy()
    y_right = right[y].to_numpy()

    grid_left = np.linspace(float(x_left.min()), cutoff, 200)
    grid_right = np.linspace(cutoff, float(x_right.max()), 200)

    coef_left = _side_polyfit(x_left, y_left, degree=degree, cutoff=cutoff)
    coef_right = _side_polyfit(x_right, y_right, degree=degree, cutoff=cutoff)

    line_df = pl.concat(
        [
            pl.DataFrame(
                {
                    "x": grid_left,
                    "y": _side_polypredict(coef_left, grid_left, cutoff),
                    "side": ["Left"] * len(grid_left),
                }
            ),
            pl.DataFrame(
                {
                    "x": grid_right,
                    "y": _side_polypredict(coef_right, grid_right, cutoff),
                    "side": ["Right"] * len(grid_right),
                }
            ),
        ],
        how="vertical",
    )

    return (
        plotnine.ggplot()
        + plotnine.geom_point(
            bins_df.to_pandas(), plotnine.aes(x="x", y="y"), color="darkblue"
        )
        + plotnine.geom_line(
            line_df.to_pandas(),
            plotnine.aes(x="x", y="y", group="side"),
            color="red",
        )
        + plotnine.geom_vline(xintercept=cutoff)
        + plotnine.theme_bw()
        + plotnine.labs(title=title, x=x_label, y=y_label)
    )
