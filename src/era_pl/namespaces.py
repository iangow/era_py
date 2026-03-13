from __future__ import annotations

from collections.abc import Callable

import polars as pl


@pl.api.register_expr_namespace("era")
class EraExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def ntile(self, n: int) -> pl.Expr:
        return (
            ((self._expr.rank("ordinal") - 1) * n / pl.len())
            .floor()
            .cast(pl.Int32)
            + 1
        )


def _with_group_keys(
    fun: Callable[[pl.DataFrame], pl.DataFrame],
    by: list[str],
):
    def wrapped(g: pl.DataFrame) -> pl.DataFrame:
        keys = g.select(by).row(0, named=True)
        res = fun(g)
        if not isinstance(res, pl.DataFrame):
            raise TypeError("fun(g) must return a Polars DataFrame")
        if res.height != 1:
            raise ValueError(
                "fun(g) must return exactly one row per group"
            )
        return pl.DataFrame({k: [keys[k]] for k in by}).hstack(res)

    return wrapped


@pl.api.register_dataframe_namespace("era")
class EraDataFrame:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def map_by(self, by: list[str], fun):
        return self._df.group_by(*by).map_groups(_with_group_keys(fun, by))
