from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

import pyarrow.parquet as pq

_DATA_PACKAGE = "era_data"


def _data_dir():
    return files(_DATA_PACKAGE).joinpath("_data")


def available_data() -> list[str]:
    """Return the names of packaged example datasets."""

    return sorted(
        item.name.removesuffix(".parquet")
        for item in _data_dir().iterdir()
        if item.name.endswith(".parquet")
    )


def _dataset_files(name: str):
    data_file = _data_dir().joinpath(f"{name}.parquet")
    if not data_file.is_file():
        available = ", ".join(available_data())
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return data_file, _data_dir().joinpath(f"{name}.meta.json")


def data_info(name: str) -> dict[str, Any]:
    """Return documentation, schema, and stored metadata for one dataset."""

    data_file, meta_file = _dataset_files(name)
    metadata: dict[str, Any] = {}
    if meta_file.is_file():
        with meta_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    documentation = metadata.get("documentation", {})
    column_descriptions = documentation.get("columns", {})
    with data_file.open("rb") as file:
        parquet = pq.ParquetFile(file)
        schema = parquet.schema_arrow
        num_rows = parquet.metadata.num_rows

    columns = [
        {
            "name": field.name,
            "dtype": str(field.type),
            "description": column_descriptions.get(field.name),
        }
        for field in schema
    ]
    return {
        "name": name,
        "title": documentation.get("title"),
        "description": documentation.get("description"),
        "source": documentation.get("source"),
        "documentation_url": documentation.get("documentation_url"),
        "num_rows": num_rows,
        "num_columns": len(columns),
        "columns": columns,
        "metadata": metadata,
    }


def data_catalog() -> list[dict[str, Any]]:
    """Return :func:`data_info` records for all packaged datasets."""

    return [data_info(name) for name in available_data()]
