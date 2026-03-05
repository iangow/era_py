import re
import era_py.text as text_mod
from era_py import NumberedLines, ptime

def test_slice_preserves_index():
    nl = NumberedLines(["a", "b", "c", "d"])
    sub = nl[1:3]
    assert sub.index == [1, 2]
    assert list(sub) == ["b", "c"]

def test_reset_index():
    nl = NumberedLines(["a", "b", "c"])[1:]
    assert nl.reset_index().index == [0, 1]

def test_filter_out():
    blank = re.compile(r"^\s*$")
    nl = NumberedLines(["x", "   ", "y"])
    out = nl.filter_out(blank)
    assert list(out) == ["x", "y"]
    assert out.index == [0, 2]


def test_ptime_public_export():
    assert callable(ptime)


def test_ptime_formats_milliseconds(monkeypatch):
    stamps = iter([10.0, 10.1234])
    rendered = []

    monkeypatch.setattr(text_mod.time, "perf_counter", lambda: next(stamps))
    monkeypatch.setattr(text_mod, "Markdown", lambda s: s)
    monkeypatch.setattr(text_mod, "display", lambda obj: rendered.append(obj))

    with text_mod.ptime():
        pass

    assert rendered == ["**Wall time:** 123.40 ms"]


def test_ptime_formats_seconds(monkeypatch):
    stamps = iter([5.0, 6.2345])
    rendered = []

    monkeypatch.setattr(text_mod.time, "perf_counter", lambda: next(stamps))
    monkeypatch.setattr(text_mod, "Markdown", lambda s: s)
    monkeypatch.setattr(text_mod, "display", lambda obj: rendered.append(obj))

    with text_mod.ptime():
        pass

    assert rendered == ["**Wall time:** 1.234 s"]
