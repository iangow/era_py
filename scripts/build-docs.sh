#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src uv run --no-sync --with 'quartodoc>=0.11,<0.12' --with 'griffe<2' \
  quartodoc build --config docs/_quarto.yml
PYTHONPATH=src uv run --no-sync python scripts/build-dataset-docs.py
quarto render docs
