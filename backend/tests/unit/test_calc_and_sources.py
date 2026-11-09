from __future__ import annotations

import pytest

from app.rag.calc import CalcError, apply_calculations, evaluate
from app.rag.generator import check_answer, citing_text, merge_duplicate_sources
from app.rag.prompts import Evidence


def test_calc_markup_is_replaced_with_exact_results():
    result = apply_calculations("Safety stock is **[[calc: 11 * 6]] units** and cost [[calc: 1,850 × 4]] total.")
    assert result.text == "Safety stock is **66 units** and cost 7,400 total."
    assert (result.calculations, result.corrections, result.failures) == (2, 0, 0)


@pytest.mark.parametrize(("expr", "expected"), [("4 * 25", "100"), ("10 / 4", "2.5"), ("(3 + 2) * 2", "10"),
                                                 ("1/3", "0.33"), ("-2 + 5", "3"), ("7 x 3", "21")])
def test_evaluate(expr, expected):
    from app.rag.calc import format_number

    assert format_number(evaluate(expr)) == expected


@pytest.mark.parametrize("expr", ["__import__('os')", "2 ** 999", "open('x')", "1/0", "a + 1", "[1,2]", "9" * 40])
def test_evaluate_rejects_anything_but_arithmetic(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_unparseable_calc_is_left_readable_not_hidden():
    result = apply_calculations("Value: [[calc: eleven times six]].")
    assert result.text == "Value: eleven times six." and result.failures == 1


def test_wrong_explicit_equations_are_corrected():
    result = apply_calculations("11 units × 6 days... formally 11 × 6 = 72 units, and 2 + 2 = 4.")
    assert "11 × 6 = 66 units" in result.text and "2 + 2 = 4" in result.text
    assert result.corrections == 1


def test_dates_and_plain_numbers_are_untouched():
    text = "On 2026-11-09 the DC had 96 units; OTIF was 92.4%."
    assert apply_calculations(text).text == text


def test_check_answer_applies_calculations():
    ev = Evidence(1, "NH-205 11 units 6 days of average demand", "d", "doc.pdf", 1, "c1", 0.5)
    checked = check_answer("**[[calc: 11 * 6]] units** (11 units/day × 6 days [1]).", [ev])
    assert checked.text.startswith("**66 units**") and checked.calculations == 1


def source(n, excerpt, page=1, doc="d1", chunk="c"):
    return {"number": n, "document_id": doc, "page_number": page, "chunk_id": f"{chunk}{n}", "excerpt": excerpt}


def test_overlapping_passages_from_same_page_are_merged_and_renumbered():
    table = "Product Supplier Lead time NH-205 Compact Mixer Bright Appliances 14 days 96 units"
    sources = [source(1, "Bright Appliances ships NH-205 from Bengaluru with a lead time of 14 calendar days"),
               source(2, table), source(3, "… Product and Supplier Data " + table)]
    answer, merged = merge_duplicate_sources("Bright Appliances [1][2][3]. Lead time 14 days [3].", sources)
    assert [s["number"] for s in merged] == [1, 2]
    assert answer == "Bright Appliances [1][2]. Lead time 14 days [2]."


def test_different_pages_or_documents_are_never_merged():
    same_text = "Identical passage text about the policy."
    sources = [source(1, same_text, page=1), source(2, same_text, page=2), source(3, same_text, doc="d2")]
    answer, merged = merge_duplicate_sources("A [1]. B [2]. C [3].", sources)
    assert len(merged) == 3 and answer == "A [1]. B [2]. C [3]."


def test_citing_text_selects_the_sentences_for_one_source():
    answer = "The DC is in Pune [1]. Lead time is 14 days [2].\n- Stores are in India [1]."
    assert citing_text(answer, 1) == "The DC is in Pune [1]. - Stores are in India [1]."
    assert citing_text(answer, 9) == answer
