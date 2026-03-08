"""
document_parser.py — Phase 1: Document Parsing Module
======================================================
Provides a unified `parse_document` interface that accepts a file path
and returns the extracted plain-text content. Supports:
  • PDF  → parsed via pdfplumber
  • DOCX → parsed via python-docx

This parser is used for both uploaded resumes and uploaded JD documents.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document as DocxDocument


# ──────────────────────────────────────────────
# Internal parsers
# ──────────────────────────────────────────────

def _parse_pdf(file_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber.

    Args:
        file_path: Absolute or relative path to the PDF.

    Returns:
        Concatenated text from every page, separated by newlines.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If pdfplumber fails to extract text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    pages_text: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF '{file_path}': {exc}") from exc

    return "\n\n".join(pages_text)


def _parse_docx(file_path: str) -> str:
    """Extract all text from a DOCX file using python-docx.

    Args:
        file_path: Absolute or relative path to the DOCX file.

    Returns:
        Concatenated paragraph text, separated by newlines.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If python-docx fails to read the document.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    try:
        doc = DocxDocument(str(path))
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    except Exception as exc:
        raise RuntimeError(f"Failed to parse DOCX '{file_path}': {exc}") from exc

    return "\n\n".join(paragraphs)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def parse_document(file_path: str) -> str:
    """Detect file type and route to the appropriate parser.

    Supports PDF (.pdf) and Word (.docx) documents.

    Args:
        file_path: Path to the document (resume or JD).

    Returns:
        Extracted plain-text content as a single string.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file type is not supported.
        RuntimeError: If the underlying parser encounters an error.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
