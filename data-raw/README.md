## Data Conversion
Batch-convert `.rda`/`.RData` files (for example from `farr/data`) to packaged Parquet:

```bash
Rscript data-raw/convert_farr_data_to_parquet.R ../farr/data src/era_py/_data
```

This writes:
- `src/era_py/_data/<dataset>.parquet`
- `src/era_py/_data/<dataset>.meta.json` (factor levels + original R classes)
