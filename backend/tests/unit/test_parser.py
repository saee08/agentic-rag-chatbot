from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.core.exceptions import DocumentParsingError, UnsupportedMediaTypeError
from app.rag.parser import parse_file, validate_file
from tests.helpers import make_docx, make_pdf, make_pptx


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_pdf_preserves_page_numbers(tmp_path):
    path = write(tmp_path, "a.pdf", make_pdf(["First page about revenue.", "Second page about factories."]))
    parsed = parse_file(path, ".pdf")
    assert parsed.page_count == 2
    assert [s.page for s in parsed.sections] == [1, 2]
    assert "revenue" in parsed.sections[0].text and "factories" in parsed.sections[1].text


def test_pdf_page_limit(tmp_path):
    path = write(tmp_path, "a.pdf", make_pdf(["one", "two", "three"]))
    with pytest.raises(DocumentParsingError, match="limit is 2"):
        parse_file(path, ".pdf", max_pages=2)


def test_pdf_without_text_is_rejected_with_ocr_hint(tmp_path):
    path = write(tmp_path, "blank.pdf", make_pdf([""]))
    with pytest.raises(DocumentParsingError, match="OCR"):
        parse_file(path, ".pdf")


def test_corrupted_pdf(tmp_path):
    path = write(tmp_path, "bad.pdf", b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(DocumentParsingError):
        parse_file(path, ".pdf")


def test_docx_includes_tables_in_order(tmp_path):
    data = make_docx(["Quarterly summary.", "See the KPI table."], table=[["KPI", "Value"], ["Retention", "94%"]])
    parsed = parse_file(write(tmp_path, "a.docx", data), ".docx")
    text = parsed.sections[0].text
    assert "Quarterly summary." in text
    assert "Retention | 94%" in text
    assert parsed.sections[0].page is None


def test_pptx_slides_and_notes(tmp_path):
    data = make_pptx([("Roadmap", "Launch in March"), ("Risks", "Supply chain")], notes={1: "Mention the port strike"})
    parsed = parse_file(write(tmp_path, "a.pptx", data), ".pptx")
    assert parsed.page_count == 2
    assert [s.page for s in parsed.sections] == [1, 2]
    assert "Launch in March" in parsed.sections[0].text
    assert "Speaker notes: Mention the port strike" in parsed.sections[1].text


def test_csv_rows_are_self_describing(tmp_path):
    data = b"region,revenue\nNorth,100\nSouth,250\n"
    parsed = parse_file(write(tmp_path, "a.csv", data), ".csv")
    text = parsed.sections[0].text
    assert "Columns: region, revenue" in text
    assert "Row 2: region: South; revenue: 250" in text


def test_csv_large_files_split_into_sections(tmp_path):
    rows = "\n".join(f"item{i},{i}" for i in range(100))
    parsed = parse_file(write(tmp_path, "a.csv", f"name,qty\n{rows}\n".encode()), ".csv")
    assert len(parsed.sections) == 3
    assert all(s.text.startswith("Columns: name, qty") for s in parsed.sections)


def test_text_encoding_fallback(tmp_path):
    path = write(tmp_path, "a.txt", "Café résumé naïve".encode("cp1252"))
    assert parse_file(path, ".txt").sections[0].text == "Café résumé naïve"


def test_markdown(tmp_path):
    path = write(tmp_path, "a.md", b"# Title\n\nSome *content* here.\n")
    assert "Some *content* here." in parse_file(path, ".md").sections[0].text


@pytest.mark.parametrize(
    ("name", "data", "error"),
    [
        ("a.exe", b"MZ....", UnsupportedMediaTypeError),
        ("a.pdf", b"not a pdf at all", UnsupportedMediaTypeError),
        ("a.docx", b"plain text pretending", UnsupportedMediaTypeError),
        ("a.txt", b"binary\x00\x01\x02", UnsupportedMediaTypeError),
        ("a.txt", b"", DocumentParsingError),
    ],
)
def test_validation_rejects_mismatched_or_empty_files(tmp_path, name, data, error):
    path = write(tmp_path, name, data)
    with pytest.raises(error):
        validate_file(path, path.suffix)


def test_zip_that_is_not_an_office_document(tmp_path):
    path = tmp_path / "a.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(UnsupportedMediaTypeError):
        validate_file(path, ".docx")


def test_zip_bomb_is_rejected(tmp_path):
    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "<w/>")
        zf.writestr("word/huge.xml", b"0" * (50 * 1024 * 1024))  # compresses ~1000:1
    with pytest.raises(DocumentParsingError, match="unreasonable size"):
        validate_file(path, ".docx")


def test_extracted_text_limit(tmp_path):
    path = write(tmp_path, "a.txt", b"word " * 5000)
    with pytest.raises(DocumentParsingError, match="more than"):
        parse_file(path, ".txt", max_chars=1000)
