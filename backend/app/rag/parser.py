"""File validation and text extraction.

Each parser returns ``Section`` objects so page/slide numbers survive into chunk metadata.
Parsers raise ``DocumentParsingError`` with a message that is safe to show to the uploader.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import DocumentParsingError, UnsupportedMediaTypeError

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".pptx", ".csv", ".txt", ".md")

_MAX_ZIP_UNCOMPRESSED = 300 * 1024 * 1024  # zip-bomb guard for DOCX/PPTX
_MAX_ZIP_RATIO = 200
_CSV_ROWS_PER_SECTION = 40
_TEXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


@dataclass(frozen=True)
class Section:
    text: str
    page: int | None = None  # 1-based page (PDF) or slide (PPTX); None when not applicable


@dataclass(frozen=True)
class ParsedDocument:
    sections: list[Section]
    page_count: int | None

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


def file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_file(path: Path, extension: str) -> None:
    """Check the bytes match the claimed type (not just the extension)."""
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type '{extension or 'none'}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    with path.open("rb") as fh:
        head = fh.read(8192)
    if not head:
        raise DocumentParsingError("The file is empty.")
    if extension == ".pdf":
        if not head.lstrip()[:5] == b"%PDF-":
            raise UnsupportedMediaTypeError("The file does not look like a PDF.")
    elif extension in (".docx", ".pptx"):
        if not head.startswith(b"PK\x03\x04"):
            raise UnsupportedMediaTypeError(f"The file does not look like a {extension[1:].upper()} document.")
        _validate_office_zip(path, extension)
    elif b"\x00" in head:
        raise UnsupportedMediaTypeError("The file appears to be binary, not text.")


def _validate_office_zip(path: Path, extension: str) -> None:
    required = "word/document.xml" if extension == ".docx" else "ppt/presentation.xml"
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            names = {i.filename for i in infos}
            total = sum(i.file_size for i in infos)
            compressed = sum(i.compress_size for i in infos) or 1
    except zipfile.BadZipFile as exc:
        raise DocumentParsingError("The file is corrupted (invalid ZIP container).") from exc
    if required not in names:
        raise UnsupportedMediaTypeError(f"The file is not a valid {extension[1:].upper()} document.")
    if total > _MAX_ZIP_UNCOMPRESSED or total / compressed > _MAX_ZIP_RATIO:
        raise DocumentParsingError("The document expands to an unreasonable size and was rejected.")


def _clean(text: str) -> str:
    # Normalise whitespace per line, drop NULs and empty runs; keep line structure for chunking.
    lines = (" ".join(line.replace("\x00", "").split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def parse_pdf(path: Path, max_pages: int) -> ParsedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise DocumentParsingError("The PDF is password-protected.")
            except NotImplementedError as exc:
                raise DocumentParsingError("The PDF uses unsupported encryption.") from exc
        pages = reader.pages
        if len(pages) > max_pages:
            raise DocumentParsingError(f"The PDF has {len(pages)} pages; the limit is {max_pages}.")
        sections = []
        for number, page in enumerate(pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                sections.append(Section(text, number))
        return ParsedDocument(sections, len(pages))
    except DocumentParsingError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise DocumentParsingError("The PDF is corrupted or could not be read.") from exc


def parse_docx(path: Path) -> ParsedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # python-docx raises a variety of lxml/zip/KeyError types
        raise DocumentParsingError("The DOCX file is corrupted or could not be read.") from exc

    parts: list[str] = []
    for block in document.iter_inner_content():  # paragraphs and tables in document order
        if isinstance(block, Paragraph):
            parts.append(block.text)
        elif isinstance(block, Table):
            parts.extend(_table_rows(block))
    for section in document.sections:  # headers/footers often carry titles and dates
        for part in (section.header, section.footer):
            parts.extend(p.text for p in part.paragraphs)
    text = _clean("\n".join(parts))
    return ParsedDocument([Section(text)] if text else [], None)


def _table_rows(table) -> list[str]:  # type: ignore[no-untyped-def]
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            value = " ".join(cell.text.split())
            if value and (not cells or cells[-1] != value):  # merged cells repeat their text
                cells.append(value)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def parse_pptx(path: Path) -> ParsedDocument:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    try:
        presentation = Presentation(str(path))
    except Exception as exc:  # python-pptx raises a variety of lxml/zip/KeyError types
        raise DocumentParsingError("The PPTX file is corrupted or could not be read.") from exc

    def shape_text(shape) -> list[str]:  # type: ignore[no-untyped-def]
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            return [t for s in shape.shapes for t in shape_text(s)]
        if getattr(shape, "has_table", False) and shape.has_table:
            return _table_rows(shape.table)
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            return [shape.text_frame.text]
        return []

    sections = []
    slides = list(presentation.slides)
    for number, slide in enumerate(slides, start=1):
        parts = [t for shape in slide.shapes for t in shape_text(shape)]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                parts.append(f"Speaker notes: {notes}")
        text = _clean("\n".join(parts))
        if text:
            sections.append(Section(text, number))
    return ParsedDocument(sections, len(slides))


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in _TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParsingError("Could not decode the text file.")  # unreachable with latin-1


def parse_text(path: Path) -> ParsedDocument:
    text = _clean(_read_text(path))
    return ParsedDocument([Section(text)] if text else [], None)


def parse_csv(path: Path) -> ParsedDocument:
    """Rows are grouped into sections that each repeat the header, so every chunk is
    self-describing ("col: value" pairs) instead of a wall of numbers."""
    text = _read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    except csv.Error as exc:
        raise DocumentParsingError("The CSV file is malformed.") from exc
    if not rows:
        return ParsedDocument([], None)
    header, body = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])], rows[1:]
    if not body:
        return ParsedDocument([Section("Columns: " + ", ".join(header))], None)
    sections = []
    for start in range(0, len(body), _CSV_ROWS_PER_SECTION):
        lines = [f"Columns: {', '.join(header)}"]
        for offset, row in enumerate(body[start : start + _CSV_ROWS_PER_SECTION], start=start + 1):
            pairs = [f"{header[i] if i < len(header) else f'column_{i + 1}'}: {v.strip()}"
                     for i, v in enumerate(row) if v.strip()]
            lines.append(f"Row {offset}: " + "; ".join(pairs))
        sections.append(Section("\n".join(lines)))
    return ParsedDocument(sections, None)


_PARSERS: dict[str, Callable[[Path], ParsedDocument]] = {
    ".docx": parse_docx,
    ".pptx": parse_pptx,
    ".csv": parse_csv,
    ".txt": parse_text,
    ".md": parse_text,
}


def parse_file(path: Path, extension: str, *, max_pages: int = 500, max_chars: int = 5_000_000) -> ParsedDocument:
    validate_file(path, extension)
    parsed = parse_pdf(path, max_pages) if extension == ".pdf" else _PARSERS[extension](path)
    if not parsed.sections:
        hint = " It may be a scanned document (OCR is not supported)." if extension == ".pdf" else ""
        raise DocumentParsingError("No extractable text was found in the document." + hint)
    if parsed.char_count > max_chars:
        raise DocumentParsingError(
            f"The document contains more than {max_chars:,} characters of text and was rejected."
        )
    return parsed
