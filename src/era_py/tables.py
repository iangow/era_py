import pandas as pd

def _stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def modelsummary(
    model,
    *,
    coef_omit=None,
    stars=True,
    statistic="std.error",
    output="dataframe",   # "dataframe" | "gt" | "styler"
    digits=3,
):
    # Prefer full vectors if present (from ols_dropcollinear)
    params = getattr(model, "params_full", model.params)
    bse = getattr(model, "bse_full", model.bse)
    tvals = getattr(model, "tvalues_full", model.tvalues)
    pvals = getattr(model, "pvalues_full", model.pvalues)

    df = (
        pd.DataFrame(
            {
                "term": params.index,
                "estimate": params.values,
                "std.error": bse.values,
                "t": tvals.values,
                "p.value": pvals.values,
            }
        )
        .dropna(subset=["estimate"])
    )

    if coef_omit is not None:
        df = df.loc[~df["term"].str.contains(coef_omit, regex=True)]

    # stars + formatting
    if stars:
        df["stars"] = df["p.value"].apply(_stars)
    else:
        df["stars"] = ""

    est_fmt = f"{{:.{digits}f}}"
    stat_fmt = f"{{:.{digits}f}}"

    df["estimate"] = df["estimate"].map(est_fmt.format) + df["stars"]

    stat_col = {"std.error": "std.error", "t": "t"}[statistic]
    df[stat_col] = df[stat_col].map(stat_fmt.format)

    out_df = df[["term", "estimate", stat_col]]

    if output == "dataframe":
        return out_df

    if output == "styler":
        return out_df.style.hide(axis="index")

    if output == "gt":
        try:
            # most common "GT" in Python Quarto land
            from great_tables import GT
        except ImportError as e:
            raise ImportError(
                "output='gt' requires the 'great-tables' package. "
                "Install with: pip install great-tables"
            ) from e

        return GT(out_df)

    raise ValueError(f"Unknown output={output!r}")