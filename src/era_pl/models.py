import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import qr
from scipy.stats import t as t_dist


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



def fit_test_score_panel(data: pl.DataFrame, vcov="iid") -> pl.DataFrame:
    """
    Fast chapter-specific estimator for the test-score panel model.

    This helper reproduces the estimate and inference returned by
    `pyfixest.feols("score ~ I(treat * post) | grade + id", ...)`
    for the vcov choices used in the panel-data chapter.
    """
    if vcov == "iid":
        vcov_type = "iid"
    elif vcov == {"CRV1": "grade"} or vcov == "grade":
        vcov_type = "CRV1-grade"
    elif vcov == {"CRV1": "id"} or vcov == "id":
        vcov_type = "CRV1-id"
    elif vcov == {"CRV1": "grade + id"} or vcov in (
        ["grade", "id"],
        ("grade", "id"),
    ):
        vcov_type = "CRV1-grade-id"
    else:
        raise ValueError(
            "vcov must be 'iid', {'CRV1': 'grade'}, {'CRV1': 'id'}, "
            "{'CRV1': 'grade + id'}, 'grade', 'id', or ['grade', 'id']."
        )

    id_idx = data["id"].to_numpy().astype(np.int64) - 1
    _, grade_idx = np.unique(data["grade"].to_numpy(), return_inverse=True)

    y = data["score"].to_numpy().astype(np.float64)
    x = (data["treat"] * data["post"]).to_numpy().astype(np.float64)

    id_counts = np.bincount(id_idx)
    grade_counts = np.bincount(grade_idx)

    id_means_y = np.bincount(id_idx, weights=y) / id_counts
    grade_means_y = np.bincount(grade_idx, weights=y) / grade_counts
    y_tilde = y - id_means_y[id_idx] - grade_means_y[grade_idx] + y.mean()

    id_means_x = np.bincount(id_idx, weights=x) / id_counts
    grade_means_x = np.bincount(grade_idx, weights=x) / grade_counts
    x_tilde = x - id_means_x[id_idx] - grade_means_x[grade_idx] + x.mean()

    xx = float(x_tilde @ x_tilde)
    estimate = float((x_tilde @ y_tilde) / xx)
    resid = y_tilde - estimate * x_tilde

    nobs = len(y)
    n_id = int(id_idx.max()) + 1
    n_grade = int(grade_idx.max()) + 1
    df_resid = max(nobs - n_id - n_grade, 1)

    if vcov_type == "iid":
        sigma2 = float(resid @ resid) / df_resid
        se = float(np.sqrt(sigma2 / xx))
        t = estimate / se
        p_value = float(2 * t_dist.sf(abs(t), df=df_resid))
    elif vcov_type in ("CRV1-grade", "CRV1-id"):
        score = x_tilde * resid

        if vcov_type == "CRV1-grade":
            cluster_idx = grade_idx
            n_clusters = n_grade
            df_k = n_id + 1
        else:
            cluster_idx = id_idx
            n_clusters = n_id
            df_k = n_grade + 1

        meat = float(
            np.bincount(cluster_idx, weights=score)
            @ np.bincount(cluster_idx, weights=score)
        )
        adj = ((nobs - 1) / max(nobs - df_k, 1)) * (
            n_clusters / max(n_clusters - 1, 1)
        )
        var = max((adj * meat) / (xx * xx), 0.0)
        se = float(np.sqrt(var))
        t = estimate / se if se > 0 else np.inf
        p_value = float(2 * t_dist.sf(abs(t), df=max(n_clusters - 1, 1)))
    else:
        score = x_tilde * resid
        meat_id = float(
            np.bincount(id_idx, weights=score)
            @ np.bincount(id_idx, weights=score)
        )
        meat_grade = float(
            np.bincount(grade_idx, weights=score)
            @ np.bincount(grade_idx, weights=score)
        )
        meat_cell = float(score @ score)

        scale = (nobs - 1) / max(nobs - 2, 1)
        g_min = min(n_id, n_grade)
        g_adj = g_min / max(g_min - 1, 1)

        var = max(
            (g_adj * scale * (meat_id + meat_grade - meat_cell))
            / (xx * xx),
            0.0,
        )
        se = float(np.sqrt(var))
        t = estimate / se if se > 0 else np.inf
        p_value = float(2 * t_dist.sf(abs(t), df=max(g_min - 1, 1)))

    return pl.DataFrame({
        "estimate": [estimate],
        "se": [se],
        "t": [float(t)],
        "p_value": [p_value],
    })
