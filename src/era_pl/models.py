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


def _to_numpy_matrix(data) -> np.ndarray:
    if isinstance(data, pl.DataFrame):
        return data.to_numpy()
    if isinstance(data, pl.Series):
        return data.to_numpy()
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    if isinstance(data, pd.Series):
        return data.to_numpy()
    return np.asarray(data)


def predict_proba_series(
    model,
    X,
    class_index: int = 1,
    name: str = "score",
) -> pl.Series:
    probs = np.asarray(model.predict_proba(X))[:, class_index]
    return pl.Series(name, probs)


def rus_sample(y_train: np.ndarray, ir: float = 1.0, rng=None) -> np.ndarray:
    rng = np.random.default_rng(2021) if rng is None else rng
    y_train = np.asarray(y_train, dtype=np.int8)
    classes, counts = np.unique(y_train, return_counts=True)
    maj_class = classes[np.argmax(counts)]
    rows_minor = np.flatnonzero(y_train != maj_class)
    rows_major = np.flatnonzero(y_train == maj_class)
    sampled_minor = rng.choice(rows_minor, size=len(rows_minor), replace=True)
    n_major = min(int(np.ceil(len(sampled_minor) * ir)), len(rows_major))
    sampled_major = rng.choice(rows_major, size=n_major, replace=False)
    return np.concatenate([sampled_minor, sampled_major])


def w_update(
    prediction: np.ndarray,
    response: np.ndarray,
    w: np.ndarray,
    learn_rate: float,
) -> tuple[np.ndarray, float, float]:
    misclass = (prediction != response).astype(np.float64)
    err = float(np.sum(w * misclass) / np.sum(w))
    if 0 < err < 0.5:
        alpha = float(learn_rate * np.log((1 - err) / err))
    else:
        alpha = 0.0
    w = w * np.exp(alpha * misclass)
    w = w / w.sum()
    return w, alpha, err


class BoostModel:
    def __init__(self, weak_learners, alpha):
        self.weak_learners = weak_learners
        self.alpha = np.asarray(alpha, dtype=np.float64)

    def _signed_votes(self, X) -> np.ndarray:
        X = np.asarray(_to_numpy_matrix(X), dtype=np.float32)
        if not self.weak_learners:
            return np.zeros((X.shape[0], 0), dtype=np.float64)
        preds = np.column_stack([
            np.where(model.predict(X) == 1, 1.0, -1.0)
            for model in self.weak_learners
        ])
        return preds * self.alpha

    def predict(self, X) -> np.ndarray:
        signed_sum = self._signed_votes(X).sum(axis=1)
        return (signed_sum > 0).astype(np.int8)

    def predict_proba(self, X) -> np.ndarray:
        votes = self._signed_votes(X)
        if votes.shape[1] == 0 or self.alpha.sum() == 0:
            probs = np.repeat(0.5, len(_to_numpy_matrix(X)))
        else:
            probs = ((votes > 0) * self.alpha).sum(axis=1) / self.alpha.sum()
        return np.column_stack([1 - probs, probs])


class ConstantProbModel:
    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict(self, X) -> np.ndarray:
        return np.repeat(int(self.prob >= 0.5), len(_to_numpy_matrix(X))).astype(np.int8)

    def predict_proba(self, X) -> np.ndarray:
        probs = np.repeat(self.prob, len(_to_numpy_matrix(X))).astype(np.float64)
        return np.column_stack([1 - probs, probs])


def rusboost(
    data,
    features: list[str],
    target: str,
    *,
    size: int = 30,
    rus: bool = True,
    learn_rate: float = 1.0,
    maxdepth: int | None = None,
    minbucket: int | None = None,
    ir: float = 1.0,
    random_state: int = 2021,
):
    from sklearn.tree import DecisionTreeClassifier

    X = _to_numpy_matrix(data[features]).astype(np.float32)
    y = _to_numpy_matrix(data[target]).astype(np.int8)
    if len(np.unique(y)) < 2:
        return ConstantProbModel(prob=float(y.mean()) if len(y) else 0.5)

    y_signed = np.where(y == 1, 1, -1).astype(np.int8)
    w = np.repeat(1 / len(y), len(y)).astype(np.float64)
    weak_learners = []
    alpha = []
    rng = np.random.default_rng(random_state)

    for _ in range(size):
        if rus:
            rows = rus_sample(y_signed, ir=ir, rng=rng)
            X_train = X[rows]
            y_train = y[rows]
            w_train = w[rows]
        else:
            X_train = X
            y_train = y
            w_train = w

        model = DecisionTreeClassifier(
            max_depth=maxdepth,
            min_samples_leaf=(minbucket if minbucket is not None else 1),
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        pred = np.where(model.predict(X) == 1, 1, -1).astype(np.int8)
        w, alpha_m, _ = w_update(pred, y_signed, w, learn_rate)
        if alpha_m == 0 and not rus:
            break
        weak_learners.append(model)
        alpha.append(alpha_m)

    return BoostModel(weak_learners=weak_learners, alpha=alpha)
