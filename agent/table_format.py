"""Turn ConvFinQA `table` fields (list of rows, first row = header) into LLM-friendly text.

Two views of the same cleaned table:
- to_markdown: compact grid for the generator prompt.
- to_row_sentences: one self-contained sentence per row, for retrieval.

We use the `table` field (not `table_ori`) because every row in `table` has
the same number of cells; `table_ori` does not on ~16% of pages.
"""
import re
from typing import List, Optional

Table = List[List[str]]

# FinQA repeats numbers in brackets: "28% ( 28 % )", "-2913 ( 2913 )", and
# sometimes mid-text: "2.875% ( 2.875 % ) senior notes".
# Group 1 captures a number; "\1" (a backreference) requires the bracket to
# contain THE SAME number. So only true duplicates are removed, while
# footnote markers "plan ( 1 )" and accounting negatives "( 1.3 )" survive.
# (?<![\d.]) stops a match from starting mid-number (e.g. the "2" in "12").
# \+? allows a "+" inside the bracket: "+181% ( +181 % )".
_DUPLICATE = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?|\.\d+)(\s*%)?\s*\(\s*\+?\1\s*%?\s*\)")
# FinQA's empty duplicate after negative percents: "( 11.4 ) % (  % )".
# Brackets holding nothing (or only "%") carry no information.
_EMPTY_BRACKETS = re.compile(r"\s*\(\s*%?\s*\)")
# "14 %" -> "14%", "( 11.4 ) %" -> "( 11.4 )%". Only applied to numeric-looking
# cells (_NUMERIC_CELL); in text "%" is a word, and "2012 % of" must not
# become "2012% of", which reads like a percentage.
_SPACE_BEFORE_PERCENT = re.compile(r"\s+%")
_NUMERIC_CELL = re.compile(r"[\s$()+\-\d.,%]*")
_DOLLAR_SPACE = re.compile(r"\$\s+")
# A whole-string number: optional sign, then "12", "12.5", "12." or ".76".
_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)")
# A cell that is entirely "( ... )": accounting notation for a negative.
_BRACKETED = re.compile(r"\(\s*(.*?)\s*\)")


def clean_cell(cell: str) -> str:
    """Remove FinQA's bracketed duplicate and the space after '$'.

    How: _DUPLICATE finds "number [%] ( same number [%] )" anywhere in the
    cell and keeps only the first number (plus "%" if it had one).
    Brackets with text ("( in thousands )") or a different number
    ("plan ( 1 )") are kept. Then empty brackets "(  % )" are removed,
    "14 %" is tightened to "14%" (numeric cells only), and "$ 5742" to "$5742".
    """
    cell = _DUPLICATE.sub(lambda m: m.group(1) + ("%" if m.group(2) else ""), cell.strip())
    cell = _EMPTY_BRACKETS.sub("", cell)
    if _NUMERIC_CELL.fullmatch(cell):
        cell = _SPACE_BEFORE_PERCENT.sub("%", cell)
    return _DOLLAR_SPACE.sub("$", cell)


def parse_number(cell: str) -> Optional[float]:
    """Return the cell's numeric value, or None if it isn't a number.

    Percents stay on their displayed scale: "28%" -> 28.0.
    Accounting negatives: "( 156188 )", "( $ 41317 )", "$ ( 245 )" -> negative.

    How: clean the cell and drop "$". If the WHOLE cell is wrapped in
    brackets, unwrap it and remember it is negative; a bracket that is only
    part of the cell ("plan ( 1 )") is not unwrapped, so it stays text.
    Then drop "," and "%" and convert only if what is left matches _NUMBER
    exactly. We check with a regex instead of `try: float(text)` because
    float() also accepts words like "nan", "inf" and "infinity", which would
    turn text cells into bogus numbers.
    """
    # rstrip("%") before the bracket check so "( 11.4 )%" is seen as bracketed;
    # it is applied again below for "( 16% )", where the % is inside.
    text = clean_cell(cell).replace("$", "").rstrip("%").strip()
    bracketed = _BRACKETED.fullmatch(text)
    if bracketed:
        text = bracketed.group(1)
    text = text.replace(",", "").rstrip("%").strip()
    if not _NUMBER.fullmatch(text):
        return None
    value = float(text)
    return -value if bracketed else value


def _clean_table(table: Table) -> Table:
    """Apply clean_cell to every cell; shared by both output formats."""
    return [[clean_cell(c) for c in row] for row in table]


def to_markdown(table: Table) -> str:
    """Render the cleaned table as a Markdown grid for the LLM prompt.

    How: each row becomes "| a | b | c |"; a "| --- | --- |" separator is
    inserted after the first (header) row, which is what makes it a Markdown
    table rather than plain text.
    """
    rows = _clean_table(table)
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    lines.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")
    return "\n".join(lines)


def to_row_sentences(table: Table) -> List[str]:
    """One sentence per data row: "label: header1 = v1; header2 = v2."

    How: pair each value with its column header (skipping the corner cell),
    so every sentence carries its own context and can be retrieved alone.
    Empty values are skipped (no "2002 = ;" noise). A row with no values at
    all is usually a section/units label ("( dollars in millions )"), so its
    label is kept on its own: "( dollars in millions )."
    """
    header, *body = _clean_table(table)
    sentences = []
    for label, *values in body:
        pairs = "; ".join(f"{h} = {v}" for h, v in zip(header[1:], values) if v)
        sentences.append(f"{label}: {pairs}." if pairs else f"{label}.")
    return sentences


def has_duplicate_headers(table: Table) -> bool:
    """True if any non-empty header name appears more than once.

    How: clean the header cells, drop empty ones (a blank corner cell is
    normal), and compare the count of names to the count of unique names.
    A True result means column names are ambiguous for this table, e.g.
    the JKHY/2009 table where two columns are both "year ended june 30 2009 2008".
    """
    names = [c for c in (clean_cell(h) for h in table[0]) if c]
    return len(names) != len(set(names))
