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


def _binned_means(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    cutoff: float,
    bins: int = 20,
) -> pd.DataFrame:
    def summarize_side(side_df: pd.DataFrame, side: str) -> pd.DataFrame:
        if side_df.empty:
            return pd.DataFrame(columns=["x", "y", "side"])

        ordered = side_df.sort_values(x)
        x_vals = ordered[x].to_numpy()
        y_vals = ordered[y].to_numpy()
        groups = np.array_split(np.arange(len(x_vals)), min(bins, len(x_vals)))
        return pd.DataFrame(
            {
                "x": [float(x_vals[g].mean()) for g in groups if len(g) > 0],
                "y": [float(y_vals[g].mean()) for g in groups if len(g) > 0],
                "side": [side for g in groups if len(g) > 0],
            }
        )

    clean = data[[x, y]].dropna()
    left = clean.loc[clean[x] < cutoff]
    right = clean.loc[clean[x] >= cutoff]
    return pd.concat(
        [
            summarize_side(left, "Left"),
            summarize_side(right, "Right"),
        ],
        ignore_index=True,
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
    data: pd.DataFrame,
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

    clean = data[[x, y]].dropna()
    bins_df = _binned_means(clean, x=x, y=y, cutoff=cutoff)

    left = clean.loc[clean[x] < cutoff]
    right = clean.loc[clean[x] >= cutoff]

    x_left = left[x].to_numpy()
    y_left = left[y].to_numpy()
    x_right = right[x].to_numpy()
    y_right = right[y].to_numpy()

    grid_left = np.linspace(float(x_left.min()), cutoff, 200)
    grid_right = np.linspace(cutoff, float(x_right.max()), 200)

    coef_left = _side_polyfit(x_left, y_left, degree=degree, cutoff=cutoff)
    coef_right = _side_polyfit(x_right, y_right, degree=degree, cutoff=cutoff)

    line_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "x": grid_left,
                    "y": _side_polypredict(coef_left, grid_left, cutoff),
                    "side": ["Left"] * len(grid_left),
                }
            ),
            pd.DataFrame(
                {
                    "x": grid_right,
                    "y": _side_polypredict(coef_right, grid_right, cutoff),
                    "side": ["Right"] * len(grid_right),
                }
            ),
        ],
        ignore_index=True,
    )

    return (
        plotnine.ggplot()
        + plotnine.geom_point(
            bins_df, plotnine.aes(x="x", y="y"), color="darkblue"
        )
        + plotnine.geom_line(
            line_df,
            plotnine.aes(x="x", y="y", group="side"),
            color="red",
        )
        + plotnine.geom_vline(xintercept=cutoff)
        + plotnine.theme_bw()
        + plotnine.labs(title=title, x=x_label, y=y_label)
    )


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
