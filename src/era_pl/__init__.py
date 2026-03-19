"""Polars-oriented API for users installing ``era-py[polars]``."""

from __future__ import annotations

from importlib import import_module

__version__ = "0.0.22"

from . import namespaces as _namespaces

__all__ = [
    "ols_dropcollinear",
    "fit_test_score_panel",
    "get_got_data",
    "available_data",
    "load_data",
    "load_farr_rda",
    "get_ff_ind",
    "get_size_rets_monthly",
    "get_me_breakpoints",
    "get_test_scores",
    "get_idd_periods",
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
    "cast_decimals",
    "sec_resolve_user_agent",
    "sec_parse_company_bytes",
    "sec_get_index",
    "sec_index_available_update",
    "sec_index_targets",
    "sec_index_update",
    "sec_index_update_all",
]

_LAZY_IMPORTS = {
    "ols_dropcollinear": (".models", "ols_dropcollinear"),
    "fit_test_score_panel": (".models", "fit_test_score_panel"),
    "get_got_data": (".models", "get_got_data"),
    "available_data": (".data", "available_data"),
    "load_data": (".data", "load_data"),
    "load_farr_rda": (".data", "load_farr_rda"),
    "get_ff_ind": (".data", "get_ff_ind"),
    "get_size_rets_monthly": (".data", "get_size_rets_monthly"),
    "get_me_breakpoints": (".data", "get_me_breakpoints"),
    "get_test_scores": (".data", "get_test_scores"),
    "get_idd_periods": (".data", "get_idd_periods"),
    "modelsummary": (".tables", "modelsummary"),
    "spline_smooth": (".plots", "spline_smooth"),
    "rdplot": (".plots", "rdplot"),
    "plotnine_star": (".plots", "plotnine_star"),
    "NumberedLines": (".text", "NumberedLines"),
    "ptime": (".text", "ptime"),
    "drive_download": ("era_py.downloads", "drive_download"),
    "ensure_lm_10x_summary_parquet": ("era_py.downloads", "ensure_lm_10x_summary_parquet"),
    "get_trading_dates": (".events", "get_trading_dates"),
    "get_annc_dates": (".events", "get_annc_dates"),
    "copy_inline": (".events", "copy_inline"),
    "get_event_dates": (".events", "get_event_dates"),
    "get_event_rets": (".events", "get_event_rets"),
    "load_parquet": (".events", "load_parquet"),
    "cast_decimals": (".events", "cast_decimals"),
    "sec_resolve_user_agent": (".sec", "sec_resolve_user_agent"),
    "sec_parse_company_bytes": (".sec", "sec_parse_company_bytes"),
    "sec_get_index": (".sec", "sec_get_index"),
    "sec_index_available_update": (".sec", "sec_index_available_update"),
    "sec_index_targets": (".sec", "sec_index_targets"),
    "sec_index_update": (".sec", "sec_index_update"),
    "sec_index_update_all": (".sec", "sec_index_update_all"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__) if module_name.startswith(".") else import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
