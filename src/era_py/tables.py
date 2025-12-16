import pandas as pd

def _stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

import re
import pandas as pd

def modelsummary(models, *, coef_omit=None, stars=True, gof_map=("nobs",), output="dataframe"):
    # allow single model
    if not isinstance(models, (list, tuple)):
        models = [models]

    def star(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    cols = []
    for j, m in enumerate(models, start=1):
        params = m.params
        pvals = m.pvalues
        s = params.copy()
        if stars:
            s = s.map(lambda v: f"{v:.3f}") + pvals.map(star)
        else:
            s = s.map(lambda v: f"{v:.3f}")
        s.name = f"Model {j}"
        cols.append(s)

    tbl = pd.concat(cols, axis=1).reset_index(names="term")

    if coef_omit is not None:
        tbl = tbl.loc[~tbl["term"].str.contains(coef_omit, regex=True)]

    # Add GOF rows (just nobs for now)
    gof_rows = []
    if gof_map:
        if "nobs" in gof_map or ("nobs" in set(gof_map)):
            row = {"term": "nobs"}
            for j, m in enumerate(models, start=1):
                row[f"Model {j}"] = f"{int(m.nobs)}"
            gof_rows.append(row)

    if gof_rows:
        tbl = pd.concat([tbl, pd.DataFrame(gof_rows)], ignore_index=True)

    if output == "dataframe":
        return tbl

    if output == "gt":
        from great_tables import GT
        return GT(tbl)

    if output == "styler":
        return tbl.style.hide(axis="index")

    raise ValueError(f"Unknown output={output!r}")