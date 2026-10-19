import polars as pl

from era_pl import ols_dropcollinear


def test_ols_dropcollinear_runs_polars():
    df = pl.DataFrame({"y": [1, 2, 3, 4], "x": [1, 2, 3, 4], "g": [1, 1, 2, 2]})
    res = ols_dropcollinear(df, "y ~ x + C(g)")
    assert hasattr(res, "params")
    assert hasattr(res, "dropped_terms")
