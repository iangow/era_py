from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Union
import re

@dataclass(frozen=True)
class NumberedLines(Sequence[str]):
    """
    A sequence of text lines that preserves original line numbers.

    `NumberedLines` behaves like a read-only sequence of strings, but
    carries an explicit `index` attribute that records the original
    (global) line numbers for each line. This is useful when working
    with text extracted from external sources (e.g., PDFs, logs, or
    scraped documents) where line numbers are meaningful for debugging,
    citation, or reproducibility.

    The object supports slicing and filtering while preserving original
    line numbers. Slicing returns a new `NumberedLines` object whose
    `index` reflects the corresponding subset of the original indices.
    The index can be reset explicitly using `reset_index()`.

    Parameters
    ----------
    lines : str or Iterable[str]
        Either a single string containing newline-separated text, or an
        iterable of strings. If a string is provided, it is split into
        lines using `str.splitlines()`. If an iterable of strings is
        provided (e.g., page-level text from a PDF), each element is
        split into lines and flattened.
    index : Iterable[int], optional
        Explicit line numbers to associate with `lines`. If not
        provided, line numbers are assigned sequentially starting from
        zero.

    Attributes
    ----------
    lines : list[str]
        The underlying text lines.
    index : list[int]
        The original line numbers corresponding to each line.

    Notes
    -----
    - `NumberedLines` is immutable: slicing and filtering return new
      objects rather than modifying the original.
    - The class does not perform any I/O. Display and printing are left
      to the caller.
    - The object can be passed directly to `pd.Series()` or other APIs
      that accept a sequence of strings.

    Examples
    --------
    Create from a single string:

    >>> nl = NumberedLines("a\\nb\\nc")
    >>> nl.index
    [0, 1, 2]

    Slice while preserving original line numbers:

    >>> nl[1:3].index
    [1, 2]

    Reset indices after slicing:

    >>> nl[1:3].reset_index().index
    [0, 1]
    """
    lines: List[str]
    index: List[int]

    def __init__(self, lines: str | Iterable[str], index: Optional[Iterable[int]] = None):
        # Case 1: a single string → split into lines
        if isinstance(lines, str):
            lines_list = lines.splitlines()
    
        # Case 2: an iterable of strings (pages or lines)
        elif isinstance(lines, Iterable):
            items = list(lines)
    
            # Split each string into lines and flatten
            lines_list = [
                ln
                for item in items
                if isinstance(item, str)
                for ln in item.splitlines()
            ]
    
        else:
            raise TypeError("lines must be a string or an iterable of strings")
    
        if index is None:
            index_list = list(range(len(lines_list)))
        else:
            index_list = list(index)
            if len(index_list) != len(lines_list):
                raise ValueError("index and lines must have the same length")
    
        object.__setattr__(self, "lines", lines_list)
        object.__setattr__(self, "index", index_list)

    # --- Sequence protocol (so it behaves like a list of lines) ---
    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self) -> Iterator[str]:
        return iter(self.lines)

    def __getitem__(self, key: Union[int, slice]) -> Union[str, "NumberedLines"]:
        if isinstance(key, slice):
            return NumberedLines(self.lines[key], self.index[key])
        return self.lines[key]

    # --- Display helpers ---
    def format(self, pad: Optional[int] = None) -> List[str]:
        """
        Return a list of strings with line numbers prefixed.
        """
        if pad is None:
            # Enough digits for the largest index value
            max_idx = max(self.index) if self.index else 0
            pad = max(2, len(str(max_idx)))
        return [f"{i:0{pad}d}: {line}" for i, line in zip(self.index, self.lines)]

    def __repr__(self) -> str:
        """
        Nice REPL display: show a short preview with numbering.
        """
        n = len(self)
        head = 5
        tail = 5
        buffer = 20
        if n <= head + tail + buffer:
            preview = self.format()
        else:
            preview = self[:head].format() + ["…"] + self[-tail:].format()

        return "NumberedLines([\n  " + "\n  ".join(preview) + "\n])"

    # --- Operations you asked for ---
    def reset_index(self) -> "NumberedLines":
        """
        Return a new NumberedLines with index reset to 0..n-1.
        """
        return NumberedLines(self.lines, range(len(self.lines)))

    def filter(self, regex: "re.Pattern[str]") -> "NumberedLines":
        """
        Return only lines matching regex (preserving original indices).
        """
        keep_lines = []
        keep_index = []
        for i, line in zip(self.index, self.lines):
            if regex.search(line):
                keep_lines.append(line)
                keep_index.append(i)
        return NumberedLines(keep_lines, keep_index)

    def filter_out(self, regex: re.Pattern[str]) -> "NumberedLines":
        keep_lines = []
        keep_index = []
        for i, line in zip(self.index, self.lines):
            if not regex.search(line):
                keep_lines.append(line)
                keep_index.append(i)
        return NumberedLines(keep_lines, keep_index)

    def to_list(self) -> List[str]:
        """
        Explicit escape hatch: get the underlying list.
        """
        return list(self.lines)
