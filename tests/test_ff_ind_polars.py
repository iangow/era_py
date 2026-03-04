import io
import zipfile

from era_pl import get_ff_ind


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def _build_zip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Siccodes5.txt", text)
    return buf.getvalue()


def test_get_ff_ind_parses_ranges(monkeypatch):
    sample = (
        "  1  Cons  Consumer NonDurables\n"
        "     0100-0199 Agricultural production crops\n"
        "     0200-0299 Livestock\n"
        "  2  Enrgy Energy\n"
        "     1200-1299 Coal mining\n"
    )

    payload = _build_zip_bytes(sample)

    def _fake_get(url, timeout=30.0):
        assert "Siccodes5.zip" in url
        return _FakeResponse(payload)

    monkeypatch.setattr("era_pl.data.requests.get", _fake_get)

    df = get_ff_ind(5)

    assert df.height == 3
    assert df.columns == [
        "ff_ind",
        "ff_ind_short_desc",
        "ff_ind_desc",
        "sic_min",
        "sic_max",
        "sic_desc",
    ]

    first = df.row(0, named=True)
    assert first["ff_ind"] == 1
    assert first["ff_ind_short_desc"].strip() == "Cons"
    assert first["ff_ind_desc"].strip() == "Consumer NonDurables"
    assert first["sic_min"] == 100
    assert first["sic_max"] == 199
    assert first["sic_desc"].strip() == "Agricultural production crops"
