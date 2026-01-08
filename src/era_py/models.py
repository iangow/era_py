import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import qr  # pivoted QR

def ols_dropcollinear(data, formula, tol=1e-10, **fit_kwargs):
    """
    Drop-in replacement for ols(data, formula, ...), but with R-like
    rank-deficiency handling:
      - refits using a full-rank subset of columns
      - returns a normal OLSResults (so fm.model.exog works)
      - stores dropped terms and 'full' (NaN-filled) coef vectors for reporting
    """
    # 1) Fit once with the formula to get Patsy design + names
    fm0 = smf.ols(formula, data=data).fit(**fit_kwargs)

    y = fm0.model.endog
    X = fm0.model.exog
    names = np.array(fm0.model.exog_names)

    # 2) Pivoted QR to determine independent columns
    Q, R, piv = qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        # pathological case
        return fm0

    rank = int((diag > tol * diag.max()).sum())
    keep = np.sort(piv[:rank])
    drop = np.sort(piv[rank:])

    keep_names = list(names[keep])
    drop_names = list(names[drop])

    # 3) Refit on the kept columns (as a DataFrame to preserve names)
    X_keep = pd.DataFrame(X[:, keep], columns=keep_names)
    fm = sm.OLS(y, X_keep).fit(**fit_kwargs)

    # 4) Attach R-like metadata for printing/reporting
    fm.dropped_terms = drop_names

    full_index = list(names)
    fm.params_full  = pd.Series(fm.params, index=keep_names).reindex(full_index)
    fm.bse_full     = pd.Series(fm.bse, index=keep_names).reindex(full_index)
    fm.tvalues_full = pd.Series(fm.tvalues, index=keep_names).reindex(full_index)
    fm.pvalues_full = pd.Series(fm.pvalues, index=keep_names).reindex(full_index)

    if fm.dropped_terms:
        print(
            "Model matrix is rank deficient.\nParameters were not estimable:\n  "
            + ", ".join(f"'{str(t)}'" for t in fm.dropped_terms)
        )     

    return fm
