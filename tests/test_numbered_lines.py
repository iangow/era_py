import re
from era_py import NumberedLines

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
