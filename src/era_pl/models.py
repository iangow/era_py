import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import qr


def _to_pandas_frame(data):
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def ols_dropcollinear(data, formula, tol=1e-10, **fit_kwargs):
    data_pd = _to_pandas_frame(data)
    fm0 = smf.ols(formula, data=data_pd).fit(**fit_kwargs)

    y = fm0.model.endog
    X = fm0.model.exog
    names = np.array(fm0.model.exog_names)

    _, R, piv = qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return fm0

    rank = int((diag > tol * diag.max()).sum())
    keep = np.sort(piv[:rank])
    drop = np.sort(piv[rank:])

    keep_names = list(names[keep])
    drop_names = list(names[drop])

    X_keep = pd.DataFrame(X[:, keep], columns=keep_names)
    fm = sm.OLS(y, X_keep).fit(**fit_kwargs)

    fm.dropped_terms = drop_names

    full_index = list(names)
    fm.params_full = pd.Series(fm.params, index=keep_names).reindex(full_index)
    fm.bse_full = pd.Series(fm.bse, index=keep_names).reindex(full_index)
    fm.tvalues_full = pd.Series(fm.tvalues, index=keep_names).reindex(full_index)
    fm.pvalues_full = pd.Series(fm.pvalues, index=keep_names).reindex(full_index)

    if fm.dropped_terms:
        print(
            "Model matrix is rank deficient.\nParameters were not estimable:\n  "
            + ", ".join(f"'{str(t)}'" for t in fm.dropped_terms)
        )

    return fm


def get_got_data(rng, N, T, Xvol, Evol, rho_X, rho_E):
    """
    Generate panel data with cross-sectional and time-series dependence.

    Parameters
    ----------
    rng:
        Numpy random generator.
    N:
        Number of firms.
    T:
        Number of time periods.
    Xvol:
        Cross-sectional dependence in X.
    Evol:
        Cross-sectional dependence in errors.
    rho_X:
        Time-series dependence in X.
    rho_E:
        Time-series dependence in errors.
    """
    f_X = rng.normal(size=T)
    f_E = rng.normal(size=T)

    data = []
    for i in range(N):
        u_X = rng.normal(size=T)
        u_E = rng.normal(size=T)

        x = np.zeros(T)
        e = np.zeros(T)

        for t in range(T):
            if t == 0:
                x[t] = Xvol * f_X[t] + np.sqrt(1 - Xvol**2) * u_X[t]
                e[t] = Evol * f_E[t] + np.sqrt(1 - Evol**2) * u_E[t]
            else:
                x[t] = rho_X * x[t - 1] + np.sqrt(1 - rho_X**2) * (
                    Xvol * f_X[t] + np.sqrt(1 - Xvol**2) * u_X[t]
                )
                e[t] = rho_E * e[t - 1] + np.sqrt(1 - rho_E**2) * (
                    Evol * f_E[t] + np.sqrt(1 - Evol**2) * u_E[t]
                )

        y = x + e

        for t in range(T):
            data.append({"firm": i, "year": t, "x": x[t], "y": y[t]})

    return pl.DataFrame(data)
