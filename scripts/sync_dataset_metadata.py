from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


def _group(text: str, start: int) -> tuple[str, int]:
    if text[start] != "{":
        raise ValueError("Expected an opening brace")
    depth = 1
    position = start + 1
    while depth:
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
        position += 1
    return text[start + 1 : position - 1], position


def _section(text: str, name: str) -> str | None:
    marker = f"\\{name}{{"
    start = text.find(marker)
    if start == -1:
        return None
    value, _ = _group(text, start + len(marker) - 1)
    return value


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\\href\{([^{}]*)\}\{([^{}]*)\}", r"\2 (\1)", text)
    substitutions = {
        "doi": lambda value: f"https://doi.org/{value}",
        "url": lambda value: value,
        "code": lambda value: f"`{value}`",
        "emph": lambda value: value,
    }
    changed = True
    while changed:
        changed = False
        for command, replacement in substitutions.items():
            pattern = re.compile(rf"\\{command}\{{([^{{}}]*)\}}")
            text, count = pattern.subn(lambda match: replacement(match.group(1)), text)
            changed = changed or count > 0
    return " ".join(text.replace(r"\&", "&").split())


def _column_docs(text: str) -> dict[str, str]:
    format_text = _section(text, "format") or ""
    columns: dict[str, str] = {}
    position = 0
    while True:
        position = format_text.find(r"\item{", position)
        if position == -1:
            return columns
        name, position = _group(format_text, position + len(r"\item"))
        description, position = _group(format_text, position)
        columns[name] = _clean(description) or ""


def _documentation(name: str, man_dir: Path) -> dict[str, object]:
    text = (man_dir / f"{name}.Rd").read_text()
    columns = _column_docs(text)
    if name == "aaer_firm_year":
        columns["gvkey"] = columns.pop("gkvey")
    if name == "aus_bank_funds":
        columns["ceq"] = "Common equity"
    if name == "undisclosed_names":
        columns["disclosed"] = "Indicator that the customer name is disclosed"
        columns.pop("disclosure")
    documentation = {
        "title": _clean(_section(text, "title")),
        "description": _clean(_section(text, "description")),
        "source": _clean(_section(text, "source")),
        "documentation_url": f"https://iangow.github.io/farr/reference/{name}.html",
        "columns": columns,
    }
    details = _clean(_section(text, "details"))
    if details:
        documentation["details"] = details
    return documentation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize packaged dataset documentation from farr manuals."
    )
    parser.add_argument("farr_root", type=Path)
    parser.add_argument("--dataset", action="append", help="Refresh only this dataset (repeatable).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/era_data/_data"),
    )
    args = parser.parse_args()
    man_dir = args.farr_root / "man"

    parquet_files = sorted(args.data_dir.glob("*.parquet"))
    if args.dataset:
        missing = set(args.dataset) - {path.stem for path in parquet_files}
        if missing:
            parser.error(f"Unknown datasets: {', '.join(sorted(missing))}")
        parquet_files = [path for path in parquet_files if path.stem in args.dataset]

    for parquet_file in parquet_files:
        name = parquet_file.stem
        metadata_file = parquet_file.with_suffix(".meta.json")
        metadata = json.loads(metadata_file.read_text())
        documentation = _documentation(name, man_dir)
        documented = set(documentation["columns"])
        actual = set(pq.read_schema(parquet_file).names)
        if documented != actual:
            raise ValueError(
                f"Column documentation mismatch for {name}: "
                f"missing={sorted(actual - documented)}, "
                f"extra={sorted(documented - actual)}"
            )
        metadata["documentation"] = documentation
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
