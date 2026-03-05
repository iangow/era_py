import math
import pandas as pd


def modelsummary(
    models,
    *,
    coef_omit=None,
    stars=True,
    estimate=None,
    statistic=True,
    statistic_fmt="({std_error:.3f})",
    gof_map=("nobs",),
    output="dataframe",
):
    # allow single model
    if not isinstance(models, (list, tuple)):
        models = [models]

    def _extract_params_pvals_ses(model):
        # statsmodels-like objects
        if hasattr(model, "params"):
            params = model.params
            pvals = getattr(model, "pvalues", None)
            if pvals is None:
                pvals = pd.Series(index=params.index, dtype=float)
            ses = getattr(model, "bse", pd.Series(index=params.index, dtype=float))
            return params, pvals, ses

        # pyfixest Feols-like objects
        if hasattr(model, "coef"):
            params = model.coef()
            pvals = model.pvalue() if hasattr(model, "pvalue") else pd.Series(index=params.index, dtype=float)
            ses = model.se() if hasattr(model, "se") else pd.Series(index=params.index, dtype=float)
            return params, pvals, ses

        raise TypeError(
            "Unsupported model type: expected statsmodels-like (.params/.pvalues/.bse) "
            "or pyfixest-like (.coef()/.pvalue()/.se())."
        )

    def _extract_nobs(model):
        if hasattr(model, "nobs"):
            return model.nobs
        if hasattr(model, "_N"):
            return model._N
        return None

    # star rules
    if stars is True:
        star_rules = {"***": 0.01, "**": 0.05, "*": 0.10}
    elif stars is False:
        star_rules = {}
    elif isinstance(stars, dict):
        star_rules = dict(stars)
    else:
        raise TypeError("stars must be bool or dict like {'*':0.1,'**':0.05,'***':0.01}")

    # smallest p cutoff first (e.g. 0.01 before 0.05)
    sorted_rules = sorted(star_rules.items(), key=lambda kv: kv[1])

    def star_for_p(p):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return ""
        for sym, cutoff in sorted_rules:
            if p < cutoff:
                return sym
        return ""

    def format_cell(v, p):
        est_txt = f"{v:.3f}"
        star_txt = star_for_p(p)

        # preserve old behavior when estimate is not provided
        if estimate is None:
            return est_txt + star_txt if stars else est_txt

        # new behavior: template formatting
        return estimate.format(
            estimate=est_txt,
            stars=star_txt,
            p_value=f"{p:.3f}" if p is not None and not (isinstance(p, float) and math.isnan(p)) else "",
            pvalue=f"{p:.3f}" if p is not None and not (isinstance(p, float) and math.isnan(p)) else "",
        )

    extracted = []
    ordered_terms = []
    seen = set()
    for m in models:
        params, pvals, ses = _extract_params_pvals_ses(m)
        extracted.append((params, pvals, ses, m))
        for term in params.index:
            if term not in seen:
                seen.add(term)
                ordered_terms.append(term)

    if coef_omit is not None:
        ordered_terms = [t for t in ordered_terms if not pd.Series([t]).str.contains(coef_omit, regex=True).iloc[0]]

    rows = []
    for term in ordered_terms:
        coef_row = {"term": term}
        for j, (params, pvals, ses, m) in enumerate(extracted, start=1):
            if term in params.index:
                coef_row[f"Model {j}"] = format_cell(params[term], pvals.get(term, float("nan")))
            else:
                coef_row[f"Model {j}"] = ""
        rows.append(coef_row)

        if statistic in (True, "std_error", "std.error", "se"):
            stat_row = {"term": ""}
            for j, (params, pvals, ses, m) in enumerate(extracted, start=1):
                if term in ses.index:
                    stat_row[f"Model {j}"] = statistic_fmt.format(std_error=float(ses[term]))
                else:
                    stat_row[f"Model {j}"] = ""
            rows.append(stat_row)

    tbl = pd.DataFrame(rows)

    # Add GOF rows (just nobs for now)
    gof_rows = []
    if gof_map:
        if "nobs" in gof_map or ("nobs" in set(gof_map)):
            row = {"term": "nobs"}
            for j, (params, pvals, ses, m) in enumerate(extracted, start=1):
                nobs = _extract_nobs(m)
                row[f"Model {j}"] = f"{int(nobs)}" if nobs is not None else ""
            gof_rows.append(row)

    if gof_rows:
        tbl = pd.concat([tbl, pd.DataFrame(gof_rows)], ignore_index=True)

    if output == "dataframe":
        return tbl

    if output == "gt":
        from great_tables import GT

        return GT(tbl).sub_missing(missing_text="")

    if output == "styler":
        return tbl.style.hide(axis="index")

    raise ValueError(f"Unknown output={output!r}")
