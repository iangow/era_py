from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import importlib
import inspect
import io

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
    bins: int | tuple[int, int] | None = None,
    binselect: str = "esmv",
) -> pd.DataFrame:
    def summarize_side(side_df: pd.DataFrame, side: str, n_bins: int) -> pd.DataFrame:
        if side_df.empty:
            return pd.DataFrame(columns=["x", "y", "side"])

        ordered = side_df.sort_values(x)
        x_vals = ordered[x].to_numpy()
        y_vals = ordered[y].to_numpy()
        if side == "Left":
            edges = np.linspace(float(x_vals.min()), cutoff, n_bins + 1)
            bin_id = np.searchsorted(edges, x_vals, side="right") - 1
        else:
            edges = np.linspace(cutoff, float(x_vals.max()), n_bins + 1)
            bin_id = np.searchsorted(edges, x_vals, side="left") - 1
        bin_id = np.clip(bin_id, 0, n_bins - 1)
        grouped = (
            pd.DataFrame({"bin": bin_id, x: x_vals, y: y_vals})
            .groupby("bin", sort=True)
            .agg({x: "mean", y: "mean"})
        )
        return pd.DataFrame(
            {
                "x": grouped[x].astype(float).to_numpy(),
                "y": grouped[y].astype(float).to_numpy(),
                "side": side,
            }
        )

    clean = data[[x, y]].dropna()
    left = clean.loc[clean[x] < cutoff]
    right = clean.loc[clean[x] >= cutoff]
    bins_left, bins_right = _resolve_rd_bins(
        clean[x].to_numpy(dtype=float),
        clean[y].to_numpy(dtype=float),
        cutoff=cutoff,
        bins=bins,
        binselect=binselect,
    )
    return pd.concat(
        [
            summarize_side(left, "Left", bins_left),
            summarize_side(right, "Right", bins_right),
        ],
        ignore_index=True,
    )


@contextmanager
def _writable_pandas_values():
    series_values = pd.Series.values
    frame_values = pd.DataFrame.values
    pd.Series.values = property(lambda self: self.to_numpy(copy=True))
    pd.DataFrame.values = property(lambda self: self.to_numpy(copy=True))
    try:
        yield
    finally:
        pd.Series.values = series_values
        pd.DataFrame.values = frame_values


def _rdrobust_plot_data(
    clean: pd.DataFrame,
    *,
    x: str,
    y: str,
    cutoff: float,
    degree: int,
    bins: int | tuple[int, int] | None,
    binselect: str,
    masspoints: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    try:
        rdrobust = importlib.import_module("rdrobust")
    except ImportError:
        return None

    try:
        with _writable_pandas_values(), redirect_stdout(io.StringIO()):
            result = rdrobust.rdplot(
                clean[y].to_numpy(dtype=float),
                clean[x].to_numpy(dtype=float),
                c=cutoff,
                p=degree,
                nbins=bins,
                binselect=binselect,
                masspoints=masspoints,
                hide=True,
            )
    except Exception:
        return None

    bins_df = result.vars_bins.rename(
        columns={"rdplot_mean_bin": "x", "rdplot_mean_y": "y"}
    )[["x", "y"]].copy()
    bins_df["side"] = np.where(bins_df["x"] < cutoff, "Left", "Right")

    line_df = result.vars_poly.rename(
        columns={"rdplot_x": "x", "rdplot_y": "y"}
    )[["x", "y"]].copy()
    midpoint = len(line_df) // 2
    line_df["side"] = ["Left"] * midpoint + ["Right"] * (len(line_df) - midpoint)
    return bins_df, line_df


def _native_rdplot_data(
    clean: pd.DataFrame,
    *,
    x: str,
    y: str,
    cutoff: float,
    degree: int,
    bins: int | tuple[int, int] | None,
    binselect: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins_df = _binned_means(
        clean,
        x=x,
        y=y,
        cutoff=cutoff,
        bins=bins,
        binselect=binselect,
    )

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
    return bins_df, line_df


def _resolve_rd_bins(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    cutoff: float,
    bins: int | tuple[int, int] | None,
    binselect: str,
) -> tuple[int, int]:
    if bins is not None:
        if isinstance(bins, tuple):
            return max(1, int(bins[0])), max(1, int(bins[1]))
        return max(1, int(bins)), max(1, int(bins))

    if binselect != "esmv":
        raise ValueError("Only binselect='esmv' is currently supported.")

    selected = _esmv_bin_counts(x_vals, y_vals, cutoff=cutoff)
    if selected is None:
        return 20, 20
    return selected


def _esmv_bin_counts(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    cutoff: float,
) -> tuple[int, int] | None:
    left = x_vals < cutoff
    right = ~left
    x_left = x_vals[left]
    x_right = x_vals[right]
    y_left = y_vals[left]
    y_right = y_vals[right]
    n = len(x_vals)

    if n < 20 or len(x_left) < 5 or len(x_right) < 5:
        return None

    def side_count(
        x_side: np.ndarray,
        y_side: np.ndarray,
        side_range: float,
    ) -> int | None:
        if side_range <= 0 or np.var(y_side) == 0:
            return 1

        order = np.argsort(x_side)
        x_ordered = x_side[order]
        y_ordered = y_side[order]
        dx = np.diff(x_ordered)
        dy = np.diff(y_ordered)
        v_hat = (0.5 / side_range) * np.sum(dx * dy**2)
        if not np.isfinite(v_hat) or v_hat <= 0:
            return None

        j_hat = np.ceil((np.var(y_side) / v_hat) * (n / np.log(n) ** 2))
        if not np.isfinite(j_hat):
            return None
        return max(1, min(int(j_hat), len(x_side)))

    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    left_count = side_count(x_left, y_left, cutoff - x_min)
    right_count = side_count(x_right, y_right, x_max - cutoff)
    if left_count is None or right_count is None:
        return None
    return left_count, right_count


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
    bins: int | tuple[int, int] | None = None,
    binselect: str = "esmv",
    masspoints: str = "adjust",
    use_rdrobust: bool = True,
):
    plotnine = importlib.import_module("plotnine")

    clean = data[[x, y]].dropna().astype({x: float, y: float})
    plot_data = None
    if use_rdrobust:
        plot_data = _rdrobust_plot_data(
            clean,
            x=x,
            y=y,
            cutoff=cutoff,
            degree=degree,
            bins=bins,
            binselect=binselect,
            masspoints=masspoints,
        )
    if plot_data is None:
        plot_data = _native_rdplot_data(
            clean,
            x=x,
            y=y,
            cutoff=cutoff,
            degree=degree,
            bins=bins,
            binselect=binselect,
        )
    bins_df, line_df = plot_data

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
