from __future__ import annotations

from collections.abc import Callable

import polars as pl


def _quantile_type2(expr: pl.Expr, prob: float) -> pl.Expr:
    """Return an expression matching R's ``quantile(type = 2)`` rule.

    This is the quantile definition used by ``farr::winsorize()``:
    use the next order statistic when ``n * p`` is non-integer and
    average adjacent order statistics when ``n * p`` is exactly integer.
    """

    sorted_expr = expr.drop_nulls().sort()
    n = expr.count()
    n_f = n.cast(pl.Float64)
    max_idx = (n - 1).clip(lower_bound=0)
    np = n_f * pl.lit(prob)
    floor_np = np.floor()
    is_integer = (np - floor_np).abs() < 1e-12
    use_midpoint = (pl.lit(0.0) < np) & (np < n_f) & is_integer

    # When n * p is not integer, type = 2 uses the next order statistic.
    upper_idx = (
        pl.when(np <= 1)
        .then(pl.lit(0))
        .when(np >= n_f)
        .then(max_idx)
        .otherwise(np.ceil().cast(pl.Int64) - 1)
        .cast(pl.UInt32)
    )
    lower_mid_idx = (
        (floor_np.cast(pl.Int64) - 1)
        .clip(lower_bound=0, upper_bound=max_idx)
        .cast(pl.UInt32)
    )
    upper_mid_idx = (
        floor_np.cast(pl.Int64)
        .clip(lower_bound=0, upper_bound=max_idx)
        .cast(pl.UInt32)
    )

    # At discontinuities (integer n * p), type = 2 averages adjacent values.
    midpoint = (sorted_expr.get(lower_mid_idx) + sorted_expr.get(upper_mid_idx)) / 2
    return pl.when(use_midpoint).then(midpoint).otherwise(sorted_expr.get(upper_idx))


def _resolve_tail_probs(
    prob: float | None,
    p_low: float | None,
    p_high: float | None,
) -> tuple[float, float]:
    """Resolve tail probabilities using the same defaults as ``farr::winsorize()``.

    ``prob`` supplies the symmetric default cutoffs. Explicit ``p_low`` and
    ``p_high`` override those defaults when provided. This argument handling is
    pure Python and negligible relative to the quantile expressions themselves.
    """

    if p_low is None and prob is not None:
        p_low = prob
    if p_high is None and prob is not None:
        p_high = 1 - prob

    if p_low is None and p_high is None:
        raise ValueError(
            "At least one of `prob`, `p_low`, or `p_high` must be supplied."
        )

    return p_low, p_high


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

    def winsorize(
        self,
        prob: float | None = 0.01,
        p_low: float | None = None,
        p_high: float | None = None,
    ) -> pl.Expr:
        """Clip values to type-2 quantile cutoffs, following ``farr::winsorize()``.

        Argument handling mirrors the R helper:

        - ``winsorize(prob=0.01)`` is equivalent to
          ``winsorize(p_low=0.01, p_high=0.99)``.
        - If ``p_low`` and/or ``p_high`` are supplied, those explicit cutoffs are
          used in preference to the defaults implied by ``prob``.
        - If ``prob=None``, only the explicitly supplied side(s) are applied, so
          one-sided winsorization is possible.
        """

        p_low, p_high = _resolve_tail_probs(prob, p_low, p_high)
        out = self._expr
        if p_low is not None:
            out = out.clip(lower_bound=_quantile_type2(self._expr, p_low))
        if p_high is not None:
            out = out.clip(upper_bound=_quantile_type2(self._expr, p_high))
        return out.name.keep()

    def truncate(
        self,
        prob: float | None = 0.01,
        p_low: float | None = None,
        p_high: float | None = None,
    ) -> pl.Expr:
        """Set values outside type-2 quantile cutoffs to null.

        Argument resolution matches :meth:`winsorize`.
        """

        p_low, p_high = _resolve_tail_probs(prob, p_low, p_high)
        keep = pl.lit(True)
        if p_low is not None:
            keep = keep & (self._expr >= _quantile_type2(self._expr, p_low))
        if p_high is not None:
            keep = keep & (self._expr <= _quantile_type2(self._expr, p_high))
        return pl.when(keep).then(self._expr).otherwise(None).name.keep()

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
