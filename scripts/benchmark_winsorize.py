from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import polars as pl

from era_pl import load_data


@dataclass
class BenchmarkCase:
    name: str
    description: str
    builder: Callable[[int, str], pl.DataFrame | pl.LazyFrame]


def _winsorize_quantile(expr: pl.Expr, prob: float = 0.01) -> pl.Expr:
    return expr.clip(expr.quantile(prob), expr.quantile(1 - prob))


def _repeat_df(df: pl.DataFrame, scale: int) -> pl.DataFrame:
    if scale <= 1:
        return df

    copies: list[pl.DataFrame] = []
    for i in range(scale):
        copies.append(
            df.with_columns(
                pl.lit(i).alias("__rep"),
            )
        )
    return pl.concat(copies, how="vertical_relaxed")


def build_cmsw_case(scale: int, method: str) -> pl.DataFrame:
    cmsw_2018 = load_data("cmsw_2018").filter(pl.col("tousesox") == 1)
    cmsw_2018 = _repeat_df(cmsw_2018, scale)
    cols = ["blckownpct", "initabret", "pctinddir", "mkt2bk", "lev"]
    win = pl.col if method == "quantile" else None
    return cmsw_2018.with_columns(
        [
            (
                _winsorize_quantile(pl.col(col))
                if method == "quantile"
                else pl.col(col).era.winsorize()
            ).alias(col)
            for col in cols
        ]
    )


def build_comp_grouped_case(scale: int, method: str) -> pl.DataFrame:
    comp = load_data("comp").select(
        "gvkey", "fyear", "roa", "lev", "mtb", "inv_at", "d_sale", "d_ar", "ppe"
    )
    # fyear=1995 is entirely null for these columns in the packaged sample.
    comp = comp.filter(pl.col("fyear") != 1995)
    comp = _repeat_df(comp, scale)
    cols = ["roa", "lev", "mtb", "inv_at", "d_sale", "d_ar", "ppe"]
    return comp.with_columns(
        [
            (
                _winsorize_quantile(pl.col(col), 0.01)
                if method == "quantile"
                else pl.col(col).era.winsorize(0.01)
            ).over("fyear").alias(col)
            for col in cols
        ]
    )


def build_cmsw_lazy_case(scale: int, method: str) -> pl.DataFrame:
    cmsw_2018 = _repeat_df(load_data("cmsw_2018"), scale)
    cols = ["blckownpct", "initabret", "pctinddir", "mkt2bk", "lev"]
    return (
        cmsw_2018
        .lazy()
        .filter(pl.col("tousesox") == 1)
        .with_columns(
            [
                (
                    _winsorize_quantile(pl.col(col))
                    if method == "quantile"
                    else pl.col(col).era.winsorize()
                ).alias(col)
                for col in cols
            ]
        )
        .collect()
    )


CASES = [
    BenchmarkCase(
        name="cmsw_eager",
        description="Book-style ungrouped winsorization from extreme-vals.qmd",
        builder=build_cmsw_case,
    ),
    BenchmarkCase(
        name="comp_grouped",
        description="Grouped by fyear, matching the natural-revisited.qmd pattern",
        builder=build_comp_grouped_case,
    ),
    BenchmarkCase(
        name="cmsw_lazy",
        description="Lazy pipeline variant of the extreme-vals.qmd winsorization",
        builder=build_cmsw_lazy_case,
    ),
]


def run_case(
    case: BenchmarkCase,
    repeats: int,
    warmups: int,
    scale: int,
    method: str,
) -> None:
    for _ in range(warmups):
        case.builder(scale, method)

    times_ms: list[float] = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = case.builder(scale, method)
        times_ms.append((time.perf_counter() - t0) * 1000)

    rows = out.height if isinstance(out, pl.DataFrame) else out.collect().height
    cols = out.width if isinstance(out, pl.DataFrame) else out.collect().width
    print(f"{case.name} [{method}]: {case.description}")
    print(f"  rows={rows:,} cols={cols} scale={scale}")
    print(
        "  "
        + " ".join(
            [
                f"min={min(times_ms):.1f}ms",
                f"median={statistics.median(times_ms):.1f}ms",
                f"mean={statistics.fmean(times_ms):.1f}ms",
                f"max={max(times_ms):.1f}ms",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--scale", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for case in CASES:
        for method in ["type2", "quantile"]:
            run_case(
                case,
                repeats=args.repeats,
                warmups=args.warmups,
                scale=args.scale,
                method=method,
            )


if __name__ == "__main__":
    main()
