from __future__ import annotations

import importlib
import inspect

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


class plotnine_star:
    """
    Temporarily expose public plotnine names in the caller's global namespace.

    Intended for module scope or notebook-cell use, not for use inside functions.
    """

    def __init__(self, module=None):
        self.module = module
        self._namespaces = []

    def __enter__(self):
        module = self.module
        if module is None:
            module = importlib.import_module("plotnine")

        frame = inspect.currentframe().f_back
        exports = {
            name: getattr(module, name)
            for name in dir(module)
            if not name.startswith("_")
        }

        namespaces = [frame.f_globals]
        if frame.f_locals is not frame.f_globals:
            namespaces.append(frame.f_locals)

        self._namespaces = []

        for namespace in namespaces:
            saved = {}
            added = set()

            for name, value in exports.items():
                if name in namespace:
                    saved[name] = namespace[name]
                else:
                    added.add(name)
                namespace[name] = value

            self._namespaces.append((namespace, saved, added))

        return self

    def __exit__(self, exc_type, exc, tb):
        for namespace, saved, added in reversed(self._namespaces):
            for name in added:
                namespace.pop(name, None)

            for name, value in saved.items():
                namespace[name] = value

        self._namespaces = []
        return False


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
