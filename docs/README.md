# Documentation

The site uses Quarto and quartodoc, with the Cosmo theme used by `db2pq` and
`farr`. Its content follows the package-site structure: overview and
installation, function reference, dataset catalog, and changelog.

## Setup

Install Quarto and prepare the Python environment with the optional APIs needed
for reference generation:

```bash
uv sync --extra docs --extra polars --extra ibis --extra tables --extra plots
```

The build script uses `uv run --no-sync` and adds quartodoc in a temporary
overlay. It does not resynchronize an existing project environment. Reference
generation imports the local checkout using `PYTHONPATH=src`; it does not
execute examples or access research databases.

## Build and publish

From the repository root:

```bash
./scripts/build-docs.sh
./scripts/publish-docs.sh
```

The publish script rebuilds the site, then publishes `docs/_site/` to the
`gh-pages` branch. Pass `--no-prompt` for an unattended publish after reviewing
the changes.

## Sources

- `CHANGELOG.md`: canonical release notes, included by `docs/news/index.qmd`.
- `docs/_quarto.yml`: site navigation and the quartodoc reference inventory.
- `docs/filters/page-title.lua`: turns leading Markdown headings into page titles,
  retaining reference anchors and making section headings searchable.
- Python docstrings: source for the generated function reference.
- `era_data.data_catalog()`: source for dataset descriptions, shapes, schemas,
  variable definitions, sources, and correction notes.

Generated reference pages, dataset pages, and rendered output are ignored by
Git. Update their source rather than editing generated files. Publishing the
website does not publish a new package version to PyPI.
