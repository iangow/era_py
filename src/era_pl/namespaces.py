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
        ).name.keep()

    def winsorize(self, prob: float = 0.01) -> pl.Expr:
        return self._expr.clip(
            self._expr.quantile(prob),
            self._expr.quantile(1 - prob),
        ).name.keep()

    def truncate(self, prob: float = 0.01) -> pl.Expr:
        lower = self._expr.quantile(prob)
        upper = self._expr.quantile(1 - prob)
        return (
            pl.when(self._expr.is_between(lower, upper, closed="both"))
            .then(self._expr)
            .otherwise(None)
        ).name.keep()

    def div_if_pos(self, other: str | pl.Expr) -> pl.Expr:
        denom = pl.col(other) if isinstance(other, str) else other
        return (
            pl.when(denom > 0)
            .then(self._expr / denom)
            .otherwise(None)
        ).name.keep()


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
