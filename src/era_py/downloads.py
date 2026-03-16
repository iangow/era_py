from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


LM_10X_ID = "1puReWu4AMuV0jfWTrrf8IbzNNEU6kfpo"


def _load_project_dotenv() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)


def _resolve_required_path(
    value: str | Path | None,
    env_var: str,
) -> Path:
    if value is None:
        _load_project_dotenv()
        value = os.getenv(env_var)
    if value is None:
        raise ValueError(f"Set `{env_var}` or pass `{env_var.lower()}` explicitly.")
    return Path(value).expanduser()


def _resolve_optional_path(
    value: str | Path | None,
    env_var: str,
) -> Path | None:
    if value is None:
        _load_project_dotenv()
        value = os.getenv(env_var)
    if value is None:
        return None
    return Path(value).expanduser()


def _looks_like_html(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as fh:
        prefix = fh.read(512).lower()
    return b"<html" in prefix or b"<!doctype html" in prefix


def _write_lm_10x_summary_parquet(csv_path: Path, parquet_path: Path) -> None:
    import polars as pl

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    header = pl.read_csv(csv_path, n_rows=0).columns
    (
        pl.scan_csv(csv_path)
        .rename({col: col.lower() for col in header})
        .with_columns(
            pl.col("cik").cast(pl.Int32, strict=False),
            pl.col("filing_date")
            .cast(pl.String)
            .str.zfill(8)
            .str.strptime(pl.Date, "%Y%m%d", strict=False),
            pl.when(pl.col("cpr").cast(pl.String) == "-99")
            .then(None)
            .otherwise(pl.col("cpr").cast(pl.String))
            .str.zfill(8)
            .str.strptime(pl.Date, "%Y%m%d", strict=False)
            .alias("cpr"),
        )
        .sink_parquet(parquet_path)
    )


def drive_download(file_id: str, destination: str | Path) -> None:
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "drive_download() requires the optional dependency `gdown`. "
            "Install `era_py` with its standard dependencies or install `gdown`."
        ) from exc

    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(
        url=url,
        output=str(destination),
        quiet=False,
        fuzzy=True,
    )


def ensure_lm_10x_summary_parquet(
    *,
    data_dir: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    data_path = _resolve_required_path(data_dir, "DATA_DIR")
    parquet_path = data_path / "glms" / "lm_10x_summary.parquet"
    if parquet_path.exists() and not overwrite:
        return parquet_path

    raw_path = _resolve_optional_path(raw_data_dir, "RAW_DATA_DIR")
    if raw_path is not None:
        csv_path = raw_path / "glms" / "lm_10x_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not csv_path.exists() or _looks_like_html(csv_path):
            drive_download(LM_10X_ID, csv_path)
        _write_lm_10x_summary_parquet(csv_path, parquet_path)
        return parquet_path

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "lm_10x_summary.csv"
        drive_download(LM_10X_ID, csv_path)
        _write_lm_10x_summary_parquet(csv_path, parquet_path)

    return parquet_path


__all__ = ["drive_download", "ensure_lm_10x_summary_parquet"]
