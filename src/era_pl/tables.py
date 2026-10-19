import pandas as pd
import polars as pl

from era_py.tables import modelsummary as _modelsummary_py


def modelsummary(models, *, output="dataframe", **kwargs):
    """Polars wrapper around era_py.modelsummary.

    Behavior is delegated to era_py.modelsummary; only difference is that
    dataframe output is converted from pandas.DataFrame to polars.DataFrame.
    """
    out = _modelsummary_py(models, output=output, **kwargs)

    if output == "dataframe" and isinstance(out, pd.DataFrame):
        return pl.from_pandas(out)

    return out
