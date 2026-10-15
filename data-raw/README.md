## Data Conversion
Batch-convert `.rda`/`.RData` files (for example from `farr/data`) to packaged Parquet:

```bash
Rscript data-raw/convert_farr_data_to_parquet.R ../farr/data src/era_data/_data
```

This writes:
- `src/era_data/_data/<dataset>.parquet`
- `src/era_data/_data/<dataset>.meta.json` (factor levels + original R classes)

Both the pandas (`era_py`) and Polars (`era_pl`) loaders use this shared data
directory. Existing documentation and provenance in the metadata are preserved
when data values are refreshed.

To refresh selected datasets and their documentation from a local `farr`
checkout, supply dataset names to the converter and `--dataset` to the
metadata script:

```bash
Rscript data-raw/convert_farr_data_to_parquet.R ../farr/data src/era_data/_data iliev_2010
uv run --no-sync python scripts/sync_dataset_metadata.py ../farr --dataset iliev_2010
```

### Public-float date corrections

The `iliev_2010` data include the five public-float date corrections documented
in [farr issue #28](https://github.com/iangow/farr/issues/28). Four fractional
years were replaced, and four months were corrected from June to July, using
the companies' Form 10-K cover pages. Both `pfyear` and `pfdate` match the
corrected R dataset; the other values and all 7,214 observations are retained.

The canonical correction table and original source CSV are maintained in
`farr/data-raw`. Regenerate `farr/data/iliev_2010.rda` there before running the
commands above; do not apply a separate rounding rule in Python. Other source
date anomalies remain outside these corrections.

The packaged metadata include the correction notes from the R help page,
available through `data_info("iliev_2010")["metadata"]["documentation"]["details"]`
in either Python namespace.
