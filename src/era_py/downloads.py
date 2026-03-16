from __future__ import annotations

from pathlib import Path


def drive_download(file_id: str, destination: str | Path) -> None:
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "drive_download() requires the optional dependency `gdown`. "
            "Install `era_py` with its standard dependencies or install `gdown`."
        ) from exc

    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(
        url=url,
        output=str(destination),
        quiet=False,
        fuzzy=True,
    )


__all__ = ["drive_download"]
