"""Tests for agent.table_format: turning ConvFinQA `table` lists into LLM-friendly text."""
import json
from pathlib import Path

import pytest

from agent.table_format import (
    clean_cell,
    has_duplicate_headers,
    parse_number,
    to_markdown,
    to_row_sentences,
)

SMALL_TABLE = [
    ["", "2002", "2001"],
    ["net sales", "$ 5742", "$ 5363"],
    ["gross margin percentage", "28% ( 28 % )", "23% ( 23 % )"],
]


# --- clean_cell -------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("28% ( 28 % )", "28%"),            # duplicated percent
        ("-2913 ( 2913 )", "-2913"),        # negative with bracketed duplicate
        ("$ -4027 ( 4027 )", "$-4027"),     # dollar + negative + duplicate
        ("$ 5742", "$5742"),                # space after dollar sign
        ("  net sales ", "net sales"),      # plain text just gets trimmed
        # Gap 1: duplicate in the middle of a cell
        ("2.875% ( 2.875 % ) senior notes due may 2026", "2.875% senior notes due may 2026"),
        (".5% ( .5 % ) decrease in expected return", ".5% decrease in expected return"),
        ("( 16% ( 16 % ) )", "( 16% )"),    # inner duplicate removed, accounting brackets kept
        ("as of 2012 2.27% ( 2.27 % ) and 3.72% ( 3.72 % )", "as of 2012 2.27% and 3.72%"),
        # Brackets that are NOT duplicates must survive
        ("2004 incentive stock plan ( 1 )", "2004 incentive stock plan ( 1 )"),  # footnote marker
        ("( 1.3 ) pt", "( 1.3 ) pt"),       # accounting negative
        # Empty duplicate "(  % )" (FinQA's form for negative percents)
        ("( 11.4 ) % (  % )", "( 11.4 )%"),
        ("14 % (  % )", "14%"),
        # Leading "+" inside the duplicate
        ("+181% ( +181 % )", "+181%"),
        # In text cells "%" is a word: don't glue it onto a year
        ("december 31 2012 % (  % ) of fair value", "december 31 2012 % of fair value"),
    ],
)
def test_clean_cell(raw, expected):
    assert clean_cell(raw) == expected


# --- parse_number -----------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$ 5742", 5742.0),
        ("-261 ( 261 )", -261.0),
        ("28% ( 28 % )", 28.0),   # percent kept on its displayed scale, not 0.28
        ("$ 9362.2", 9362.2),
        ("net sales", None),      # text is not a number
        ("", None),
        # Gap 2: accounting-style negatives, (x) means -x
        ("( 156188 )", -156188.0),
        ("( $ 41317 )", -41317.0),
        ("$ ( 245 )", -245.0),
        ("( 1,234.5 )", -1234.5),
        ("( 16% ( 16 % ) )", -16.0),
        ("( 11.4 ) % (  % )", -11.4),   # negative percent, empty duplicate
        ("14 % (  % )", 14.0),
        ("+181% ( +181 % )", 181.0),
        # Not a whole-cell bracketed number -> still not a number
        ("2004 incentive stock plan ( 1 )", None),
        ("$ ( 245 ) 1", None),    # trailing footnote marker; left as text for now
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


# --- to_markdown ------------------------------------------------------------

def test_to_markdown_renders_header_separator_and_clean_cells():
    assert to_markdown(SMALL_TABLE) == (
        "|  | 2002 | 2001 |\n"
        "| --- | --- | --- |\n"
        "| net sales | $5742 | $5363 |\n"
        "| gross margin percentage | 28% | 23% |"
    )


# --- to_row_sentences -------------------------------------------------------

def test_to_row_sentences_one_self_contained_sentence_per_data_row():
    assert to_row_sentences(SMALL_TABLE) == [
        "net sales: 2002 = $5742; 2001 = $5363.",
        "gross margin percentage: 2002 = 28%; 2001 = 23%.",
    ]


def test_to_row_sentences_skips_empty_values():
    table = [["", "2009", "2008", ""], ["total", "5148881", "$ 21.96", ""]]
    assert to_row_sentences(table) == ["total: 2009 = 5148881; 2008 = $21.96."]


def test_to_row_sentences_keeps_label_of_row_with_no_values():
    # Section / units rows carry context, so keep the label alone.
    table = [["", "2009", "2008"], ["( dollars in millions )", "", ""]]
    assert to_row_sentences(table) == ["( dollars in millions )."]


# --- has_duplicate_headers --------------------------------------------------

def test_has_duplicate_headers_detects_repeated_names():
    # Real broken header from Single_JKHY/2009/page_28.pdf
    table = [
        ["2008", "year ended june 30 2009 2008", "year ended june 30 2009 2008", "year ended june 30 2009"],
        ["net income", "$ 103102", "$ 104222", "$ 104681"],
    ]
    assert has_duplicate_headers(table) is True


def test_has_duplicate_headers_false_for_normal_table():
    assert has_duplicate_headers(SMALL_TABLE) is False


def test_has_duplicate_headers_ignores_repeated_empty_cells():
    table = [["", "2015", "2014", ""], ["x", "1", "2", "3"]]
    assert has_duplicate_headers(table) is False


# --- whole-dataset sanity check ---------------------------------------------

TRAIN = Path(__file__).resolve().parent.parent / "data" / "train.json"


@pytest.mark.skipif(not TRAIN.exists(), reason="train.json not available")
def test_every_train_table_renders_one_markdown_line_per_row_plus_separator():
    records = json.loads(TRAIN.read_text())
    for r in records:
        lines = to_markdown(r["table"]).split("\n")
        assert len(lines) == len(r["table"]) + 1, r["id"]
