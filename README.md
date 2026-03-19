# era_py

Helpers for the Python port of *Empirical Research in Accounting: Tools and Methods*.

## Install

```
pip install era-py
pip install "era-py[polars]"
pip install "era-py[tables]"
```

For repository development, prefer the checked-in virtual environment:

```bash
.venv/bin/python -m pytest
```

`era-py` provides two import paths:

- `import era_py` is the base API built around pandas/Ibis-style workflows.
- `import era_pl` is the Polars API and requires `pip install "era-py[polars]"`.

## Usage
```python
from era_py import load_data, ols_dropcollinear

camp = load_data("camp_attendance")
```

## Testing Optional Dependencies

For quick checks of optional extras without changing your local environment, use
`uv run --with ...`:

```bash
PYTHONPATH=src uv run --with polars python -c "import era_pl; print(era_pl.__version__)"
```

This is useful for verifying the `era_pl` import path and other Polars-specific
behavior without permanently installing `polars` into your project environment.
