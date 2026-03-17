"""Base pandas/Ibis-oriented API for users installing ``era-py``."""

from __future__ import annotations

from importlib import import_module

__version__ = "0.0.20"

__all__ = [
    "ols_dropcollinear",
    "available_data",
    "load_data",
    "load_farr_rda",
    "modelsummary",
    "spline_smooth",
    "rdplot",
    "plotnine_star",
    "__version__",
    "NumberedLines",
    "ptime",
    "drive_download",
    "ensure_lm_10x_summary_parquet",
    "get_trading_dates",
    "get_annc_dates",
    "copy_inline",
    "get_event_dates",
    "get_event_rets",
    "load_parquet",
    "wrds_connect", 
    "wrds_table",
]

_LAZY_IMPORTS = {
    "ols_dropcollinear": (".models", "ols_dropcollinear"),
    "available_data": (".data", "available_data"),
    "load_data": (".data", "load_data"),
    "load_farr_rda": (".data", "load_farr_rda"),
    "modelsummary": (".tables", "modelsummary"),
    "spline_smooth": (".plots", "spline_smooth"),
    "rdplot": (".plots", "rdplot"),
    "plotnine_star": (".plots", "plotnine_star"),
    "NumberedLines": (".text", "NumberedLines"),
    "ptime": (".text", "ptime"),
    "drive_download": (".downloads", "drive_download"),
    "ensure_lm_10x_summary_parquet": (".downloads", "ensure_lm_10x_summary_parquet"),
    "get_trading_dates": (".events", "get_trading_dates"),
    "get_annc_dates": (".events", "get_annc_dates"),
    "copy_inline": (".events", "copy_inline"),
    "get_event_dates": (".events", "get_event_dates"),
    "get_event_rets": (".events", "get_event_rets"),
    "load_parquet": (".events", "load_parquet"),
    "cast_decimals": (".events", "cast_decimals"),
    "wrds_connect": (".wrds", "wrds_connect"),
    "wrds_table": (".wrds", "wrds_table"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
