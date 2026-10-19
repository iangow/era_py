from pathlib import Path

import polars as pl

from era_pl import ensure_lm_10x_summary_parquet
import era_py.downloads as downloads


CSV_TEXT = """CIK,FILING_DATE,CPR,FORM_TYPE,ACC_NUM,GROSSFILESIZE
1001,20010131,20001231,10-K,0001,1234
"""


def test_ensure_lm_10x_summary_parquet_uses_existing_raw_csv(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    csv_path = raw_dir / "glms" / "lm_10x_summary.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(CSV_TEXT, encoding="utf-8")

    def fail_download(*args, **kwargs):
        raise AssertionError("download should be skipped when raw CSV exists")

    monkeypatch.setattr(downloads, "drive_download", fail_download)

    parquet_path = ensure_lm_10x_summary_parquet(
        data_dir=data_dir,
        raw_data_dir=raw_dir,
    )

    assert parquet_path == data_dir / "glms" / "lm_10x_summary.parquet"
    assert parquet_path.exists()

    out = pl.read_parquet(parquet_path)
    assert out.shape == (1, 6)
    assert out.schema["filing_date"] == pl.Date
    assert out.schema["cpr"] == pl.Date


def test_ensure_lm_10x_summary_parquet_uses_temp_csv_without_raw_data_dir(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    seen = {}
    monkeypatch.delenv("RAW_DATA_DIR", raising=False)

    def fake_download(file_id: str, destination: str | Path) -> None:
        destination = Path(destination)
        seen["destination"] = destination
        destination.write_text(CSV_TEXT, encoding="utf-8")

    monkeypatch.setattr(downloads, "drive_download", fake_download)

    parquet_path = ensure_lm_10x_summary_parquet(data_dir=data_dir)

    assert parquet_path.exists()
    assert "destination" in seen
    assert seen["destination"].name == "lm_10x_summary.csv"
    assert data_dir.joinpath("glms", "lm_10x_summary.csv").exists() is False
