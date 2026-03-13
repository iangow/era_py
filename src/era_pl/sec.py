from __future__ import annotations

import gzip
import os
import re
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import find_dotenv, load_dotenv

import polars as pl
import pyarrow.parquet as pq
import requests


SEC_INDEX_FILE_RE = re.compile(r"sec_index_(\d{4})q([1-4])\.parquet$")


def _load_project_dotenv() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)


def _resolve_data_dir(data_dir: str | None = None) -> Path:
    if not data_dir:
        _load_project_dotenv()
        data_dir = os.path.expanduser(os.environ["DATA_DIR"])
    return Path(data_dir).expanduser()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sec_request(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int = 120,
    attempts: int = 3,
    backoff: float = 1.0,
) -> requests.Response:
    last_error: requests.RequestException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as err:
            last_error = err
            if attempt == attempts:
                raise
            time.sleep(backoff * attempt)

    raise RuntimeError(f"Unreachable retry state for {url}") from last_error


def sec_resolve_user_agent(user_agent: str | None = None) -> dict[str, str]:
    if user_agent is None:
        _load_project_dotenv()
        user_agent = os.getenv("USER_AGENT")

    if not user_agent:
        raise ValueError(
            "Set `user_agent` or the `USER_AGENT` environment variable "
            "before downloading from the SEC website."
        )

    return {"User-Agent": user_agent}


def sec_parse_company_bytes(content: bytes) -> pl.DataFrame:
    with gzip.open(
        BytesIO(content),
        "rt",
        encoding="macintosh",
        errors="replace",
    ) as fh:
        lines = fh.readlines()[10:]

    rows: list[dict[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        date_text = line[91:103].strip()
        rows.append(
            {
                "company_name": line[0:62].strip(),
                "form_type": line[62:79].strip(),
                "cik": (
                    int(line[79:91].strip())
                    if line[79:91].strip() else None
                ),
                "date_filed": (
                    datetime.strptime(date_text, "%Y-%m-%d").date()
                    if "-" in date_text
                    else datetime.strptime(date_text, "%Y%m%d").date()
                ),
                "file_name": line[103:].strip(),
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("cik").cast(pl.Int32))


def sec_last_modified_to_string(last_modified: datetime) -> str:
    return last_modified.isoformat()


def sec_last_modified_from_string(last_modified: str) -> datetime:
    return datetime.fromisoformat(last_modified)


def sec_parquet_last_modified(path: str | Path) -> datetime | None:
    schema_md = pq.read_schema(Path(path).expanduser()).metadata
    if not schema_md or b"last_modified" not in schema_md:
        return None
    return sec_last_modified_from_string(schema_md[b"last_modified"].decode("utf-8"))


def _sec_write_parquet_with_last_modified(
    frame: pl.DataFrame,
    path: Path,
    last_modified: datetime,
) -> None:
    table = frame.to_arrow()
    md = dict(table.schema.metadata or {})
    md[b"last_modified"] = sec_last_modified_to_string(last_modified).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(md), path)


def sec_index_available_update(
    year: int,
    quarter: int,
    user_agent: str | None = None,
) -> datetime:
    url = (
        "https://www.sec.gov/Archives/edgar/full-index/"
        f"{year}/QTR{quarter}/"
    )
    resp = _sec_request(
        url,
        headers=sec_resolve_user_agent(user_agent),
        timeout=120,
    )
    resp.raise_for_status()

    match = re.search(
        (
            r'company\.gz</a></td>\s*<td[^>]*>[^<]+</td>\s*'
            r'<td[^>]*>([^<]+)</td>'
        ),
        resp.text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            (
                f"Could not parse last-modified timestamp "
                f"for {year} Q{quarter}."
            )
        )

    dt = datetime.strptime(match.group(1).strip(), "%m/%d/%Y %I:%M:%S %p")
    return dt.replace(tzinfo=ZoneInfo("America/New_York"))


def sec_get_index(
    year: int,
    quarter: int,
    overwrite: bool = False,
    data_dir: str | None = None,
    schema: str = "edgar",
    user_agent: str | None = None,
    last_modified: datetime | None = None,
) -> pl.LazyFrame | None:
    data_path = _resolve_data_dir(data_dir)
    edgar_dir = data_path / schema
    _ensure_dir(edgar_dir)
    pq_path = edgar_dir / f"sec_index_{year}q{quarter}.parquet"
    if pq_path.exists() and not overwrite:
        return pl.scan_parquet(pq_path)

    if last_modified is None:
        last_modified = sec_index_available_update(
            year,
            quarter,
            user_agent=user_agent,
        )

    url = (
        "https://www.sec.gov/Archives/edgar/full-index/"
        f"{year}/QTR{quarter}/company.gz"
    )
    resp = _sec_request(
        url,
        headers=sec_resolve_user_agent(user_agent),
        timeout=120,
    )
    if resp.status_code != 200:
        return None

    _sec_write_parquet_with_last_modified(
        sec_parse_company_bytes(resp.content),
        pq_path,
        last_modified,
    )
    return pl.scan_parquet(pq_path)


def sec_index_targets() -> pl.DataFrame:
    now = datetime.now() - timedelta(days=1)
    current_year = now.year
    current_qtr = (now.month - 1) // 3 + 1
    return pl.DataFrame(
        {
            "year": [
                year
                for year in range(1993, current_year + 1)
                for q in range(1, 5)
                if year < current_year or q <= current_qtr
            ],
            "quarter": [
                q
                for year in range(1993, current_year + 1)
                for q in range(1, 5)
                if year < current_year or q <= current_qtr
            ],
        },
        schema={"year": pl.Int32, "quarter": pl.Int32},
    )


def sec_index_update(
    year: int,
    quarter: int,
    data_dir: str | None = None,
    schema: str = "edgar",
    user_agent: str | None = None,
    last_modified: datetime | None = None,
    report: str = "all",
) -> bool:
    if report not in {"all", "update", "none"}:
        raise ValueError('report must be one of {"all", "update", "none"}.')

    try:
        if last_modified is None:
            last_modified = sec_index_available_update(
                year,
                quarter,
                user_agent=user_agent,
            )

        data_path = _resolve_data_dir(data_dir)
        pq_path = data_path / schema / f"sec_index_{year}q{quarter}.parquet"
        local_last_modified = (
            sec_parquet_last_modified(pq_path) if pq_path.exists() else None
        )

        if local_last_modified is not None and local_last_modified >= last_modified:
            if report == "all":
                print(f"{year} Q{quarter}: Index file is up to date.")
            return False

        if report in {"all", "update"}:
            print(f"{year} Q{quarter}: Updated index file available. Fetching update.")
        return sec_get_index(
            year,
            quarter,
            overwrite=True,
            data_dir=str(data_path),
            schema=schema,
            user_agent=user_agent,
            last_modified=last_modified,
        ) is not None
    except requests.RequestException as err:
        print(f"{year} Q{quarter}: Request failed ({err.__class__.__name__}).")
        return False


def sec_index_local(
    data_dir: str | None = None,
    schema: str = "edgar",
) -> pl.DataFrame | None:
    data_path = _resolve_data_dir(data_dir)
    schema_path = data_path / schema
    if not schema_path.exists():
        return None

    rows: list[dict[str, object]] = []
    for path in sorted(schema_path.glob("sec_index_*.parquet")):
        match = SEC_INDEX_FILE_RE.match(path.name)
        if match is None:
            continue
        last_modified = sec_parquet_last_modified(path)
        rows.append(
            {
                "year": int(match.group(1)),
                "quarter": int(match.group(2)),
                "last_modified": last_modified,
            }
        )

    if not rows:
        return None

    return pl.DataFrame(rows, schema={
        "year": pl.Int32,
        "quarter": pl.Int32,
        "last_modified": pl.Datetime("us", "America/New_York"),
    })


def sec_index_update_all(
    data_dir: str | None = None,
    schema: str = "edgar",
    user_agent: str | None = None,
    report: str = "update",
) -> pl.DataFrame:
    if report not in {"all", "update", "none"}:
        raise ValueError('report must be one of {"all", "update", "none"}.')

    data_path = _resolve_data_dir(data_dir)
    targets = sec_index_targets()
    return targets.with_columns(
        available=pl.struct("year", "quarter").map_elements(
            lambda x: sec_index_update(
                int(x["year"]),
                int(x["quarter"]),
                data_dir=str(data_path),
                schema=schema,
                user_agent=user_agent,
                report=report,
            ),
            return_dtype=pl.Boolean,
        )
    )
