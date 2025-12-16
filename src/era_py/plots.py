from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


def spline_smooth(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    df: int = 6,
    n: int = 200,
) -> pd.DataFrame:
    """
    Fit a cubic regression spline y ~ s(x) and return a prediction grid for plotting.

    Parameters
    ----------
    data : DataFrame
        Input data.
    x, y : str
        Column names for x and y.
    df : int
        Degrees of freedom for the spline basis (patsy cr()).
    n : int
        Number of grid points for the smooth curve.

    Returns
    -------
    DataFrame with columns [x, f"{y}_smooth"] suitable for plotnine geom_line().
    """
    d = data[[x, y]].dropna()

    X = dmatrix(f"cr({x}, df={df})", data=d, return_type="dataframe")
    fit = sm.OLS(d[y].to_numpy(), X.to_numpy()).fit()

    grid = pd.DataFrame({x: np.linspace(d[x].min(), d[x].max(), n)})
    Xg = dmatrix(f"cr({x}, df={df})", data=grid, return_type="dataframe")
    grid[f"{y}_smooth"] = fit.predict(Xg.to_numpy())

    return grid