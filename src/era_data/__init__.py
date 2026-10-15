"""Shared packaged datasets used by both the pandas and Polars APIs."""

from .catalog import available_data, data_catalog, data_info

__all__ = ["available_data", "data_catalog", "data_info"]
