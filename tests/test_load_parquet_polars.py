import polars as pl

from era_pl import load_parquet


def test_load_parquet_polars_signature(tmp_path):
    base = tmp_path
    schema_dir = base / "crsp"
    schema_dir.mkdir(parents=True)

    source = pl.DataFrame({"permno": [1, 2], "ret": [0.1, 0.2]})
    source.write_parquet(schema_dir / "dsf.parquet")

    out = load_parquet("dsf", "crsp", data_dir=str(base))
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().shape == (2, 2)
