from __future__ import annotations

import json
from pathlib import Path

from era_data import data_catalog


def table_cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "docs" / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = data_catalog()
    index_lines = [
        "---",
        "title: Packaged datasets",
        "---",
        "",
        "Both `era_py` and `era_pl` load the same packaged data. Use",
        "`load_data(name)` to load a dataset and `data_info(name)` to inspect",
        "its documentation without loading the observations.",
        "",
        "| Dataset | Description | Rows | Columns |",
        "| --- | --- | ---: | ---: |",
    ]
    for dataset in datasets:
        name = dataset["name"]
        title = dataset["title"] or name
        index_lines.append(
            f"| [`{name}`]({name}.qmd) | {table_cell(title)} | "
            f"{dataset['num_rows']:,} | {dataset['num_columns']} |"
        )
        lines = [
            "---",
            f"title: {json.dumps(title)}",
            "---",
            "",
            dataset["description"] or "",
            "",
            "## Usage",
            "",
            "```python",
            "from era_pl import load_data",
            "",
            f'data = load_data("{name}")',
            "```",
            "",
            "Use `from era_py import load_data` for a pandas dataframe.",
            "",
            "## Format",
            "",
            f"{dataset['num_rows']:,} rows and {dataset['num_columns']} columns.",
            "Column types below describe the stored Parquet data; loaders also",
            "restore categories and date types using the packaged metadata.",
            "",
            "| Column | Stored type | Description |",
            "| --- | --- | --- |",
        ]
        for column in dataset["columns"]:
            lines.append(
                f"| `{column['name']}` | `{table_cell(column['dtype'])}` | "
                f"{table_cell(column['description'])} |"
            )
        details = dataset["metadata"].get("documentation", {}).get("details")
        if details:
            lines.extend(["", "## Details", "", details])
        if dataset["source"]:
            lines.extend(["", "## Source", "", dataset["source"]])
        if dataset["documentation_url"]:
            lines.extend([
                "",
                f"[Related R documentation]({dataset['documentation_url']}).",
            ])
        (output_dir / f"{name}.qmd").write_text("\n".join(lines) + "\n")
    (output_dir / "index.qmd").write_text("\n".join(index_lines) + "\n")
    print(f"Generated documentation for {len(datasets)} datasets.")


if __name__ == "__main__":
    main()
