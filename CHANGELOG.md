# Changelog

Release notes cover version 0.0.32 onward. Dates are the dates of publication on
[PyPI](https://pypi.org/project/era-py/#history). For earlier changes, see the
[commit history](https://github.com/iangow/era_py/commits/main/).

## Version 0.0.35 (2026-10-15)

### Data and metadata

- Added `data_info(name)` and `data_catalog()` to both `era_py` and `era_pl`.
  These provide dataset descriptions, sources, dimensions, schemas, and variable
  definitions for all 27 packaged datasets without loading their observations.
- Refreshed `aaer_dates` from the canonical `farr` data, retaining all 3,386
  reconciled rows, historical entries, and source-backed corrections.
- Synchronized five public-float date corrections in `iliev_2010` with `farr`,
  including four fractional years. All 7,214 observations are retained.

### Functions and dependencies

- Made Ibis an optional dependency. Install `era-py[ibis]` for the Ibis/Postgres
  event-study and WRDS helpers; the base API no longer requires Ibis.
- Corrected `era_pl`'s `ntile()` expression to preserve nulls, exclude them from
  bucket counts, and assign larger buckets first, matching R/SQL behavior.
- Updated Polars AdaBoost and RUSBoost to retain a perfect learner with weight
  1, handle unusable learners, and report an error when none can be fitted.
- Added validation of boosting probability weights: weights must be finite,
  nonnegative, and have a positive finite total. Predictions continue to use
  normalized weighted vote shares.
- Fixed daily and monthly Ibis cumulative event returns for complete losses
  (`ret = -1`) in stocks and market or size benchmarks. Missing-value handling
  is unchanged; returns below -1 remain errors.

### Validation

- Added regression tests for the data, metadata, optional dependencies, Polars
  helpers, boosting, and Ibis event returns. CI now installs the Ibis and DuckDB
  dependencies needed for the event-return tests.

## Version 0.0.34 (2026-10-12)

- Aligned `camp_attendance`, `iliev_2010`, and `test_scores` with the teaching
  data in the R package `farr`.
- Updated dataset metadata and tests for both the pandas and Polars loaders.

## Version 0.0.33 (2026-10-07)

- Added support for Polars 2 and required `polars>=2.0.0` in the Polars extra.
- Updated date-expression handling in the Polars data helpers and added
  compatibility tests for data loading and expression methods.

## Version 0.0.32 (2026-10-04)

- Added explicit parsing of string dates when restoring packaged Polars data
  and constructing event dates.
- Normalized date inputs used to filter Fama-French daily factors.
