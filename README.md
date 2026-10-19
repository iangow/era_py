# era_py

Helpers for the Python port of *Empirical Research in Accounting: Tools and Methods*.

The `era_pl` Polars API is used by `era_pl_book` and its companion materials.
Install it with `pip install "era-py[polars]"` and import `era_pl`.

[Website](https://iangow.github.io/era_py/) |
[Changelog](https://iangow.github.io/era_py/news/index.html)

## Install

```
pip install era-py
pip install "era-py[ibis]"
pip install "era-py[polars]"
pip install "era-py[tables]"
```

For repository development, prefer the checked-in virtual environment:

```bash
.venv/bin/python -m pytest
```

`era-py` provides two import paths:

- `import era_py` is the base API built around pandas workflows. Install the
  `ibis` extra for the Ibis/Postgres event-study and WRDS helpers.
- `import era_pl` is the Polars API and requires `pip install "era-py[polars]"`.

## Usage
```python
from era_py import data_info, load_data, ols_dropcollinear

camp = load_data("camp_attendance")
camp_info = data_info("camp_attendance")
```

Use `data_info(name)` to inspect a dataset's description, provenance, shape,
schema, and variable definitions without loading its observations. Use
`data_catalog()` to return the same information for every packaged dataset.

## Testing Optional Dependencies

For quick checks of optional extras without changing your local environment, use
`uv run --with ...`:

```bash
PYTHONPATH=src uv run --with polars python -c "import era_pl; print(era_pl.__version__)"
```

This is useful for verifying the `era_pl` import path and other Polars-specific
behavior without permanently installing `polars` into your project environment.

## Website

The changelog is maintained in `CHANGELOG.md` and included in the website, so
release notes need to be edited in only one place. The site uses Quarto and
quartodoc, following the `db2pq` documentation setup. Build and publish with:

```bash
./scripts/build-docs.sh
./scripts/publish-docs.sh
```

See [the documentation build instructions](docs/README.md) for setup details.
The build reads source docstrings and packaged dataset metadata; it does not
run external-data examples or require access to research databases.
