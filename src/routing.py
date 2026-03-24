"""
routing.py — Phase 1: JD Text Routing Workflow
===============================================
Provides the ``get_jd_text`` function that accepts EITHER an uploaded
file path OR a selected role id, determines which source to use, and
returns the clean extracted JD text.

Priority:
  • If an uploaded file path is provided → parse it and return its text.
  • If only a role id is provided → look it up in the knowledge base.
  • If both are provided → the uploaded file takes precedence.
  • If neither is provided → raise an error.
"""

from __future__ import annotations

from typing import Optional

from src.document_parser import parse_document
from src.kb_loader import KnowledgeBaseIndex


def _extract_jd_text_from_role(role: dict) -> str:
    """Extract a clean JD text string from a Kaggle role record.

    Synthesises a readable JD paragraph from the ``Title``,
    ``ExperienceLevel``, ``YearsOfExperience``, ``Skills``,
    ``Responsibilities``, and ``Keywords`` fields.
    """
    parts: list[str] = []

    title = role.get("Title") or "Unknown Role"
    level = role.get("ExperienceLevel") or ""
    yoe   = role.get("YearsOfExperience") or ""

    parts.append(f"Role: {title}")
    if level:
        parts.append(f"Experience Level: {level}")
    if yoe:
        parts.append(f"Years of Experience: {yoe}")

    # Skills
    skills = role.get("Skills") or []
    if isinstance(skills, list) and skills:
        parts.append(f"Required Skills: {', '.join(skills)}")
    elif isinstance(skills, str) and skills:
        parts.append(f"Required Skills: {skills}")

    # Responsibilities
    responsibilities = role.get("Responsibilities") or []
    if isinstance(responsibilities, list) and responsibilities:
        parts.append("Key Responsibilities:")
        for resp in responsibilities:
            parts.append(f"  • {resp}")
    elif isinstance(responsibilities, str) and responsibilities:
        parts.append(f"Key Responsibilities: {responsibilities}")

    # Keywords
    keywords = role.get("Keywords") or []
    if isinstance(keywords, list) and keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    return "\n".join(parts)


def get_jd_text(
    uploaded_file_path: Optional[str] = None,
    role_id: Optional[str] = None,
    kb_index: Optional[KnowledgeBaseIndex] = None,
) -> str:
    """Return clean JD text from the appropriate source.

    Workflow:
      1. If ``uploaded_file_path`` is given (and non-empty), parse the
         document and return extracted text. The uploaded file takes
         priority even if ``role_id`` is also provided.
      2. Otherwise, look up ``role_id`` in ``kb_index`` and return the
         JD text associated with that role.
      3. If neither argument is provided, raise ``ValueError``.

    Args:
        uploaded_file_path: Path to an uploaded PDF or DOCX file.
        role_id: JobID of a role in the knowledge base.
        kb_index: A loaded ``KnowledgeBaseIndex`` instance (required
                  when ``role_id`` is used).

    Returns:
        Extracted / synthesised JD text as a plain string.

    Raises:
        ValueError: If neither source is provided, or the role is not
                    found in the KB.
        FileNotFoundError: If the uploaded file does not exist.
        RuntimeError: If document parsing fails.
    """
    # ── Path 1: Uploaded document (takes priority) ───────
    if uploaded_file_path and uploaded_file_path.strip():
        print(f"[Router] Parsing uploaded document: {uploaded_file_path}")
        text = parse_document(uploaded_file_path.strip())
        if not text.strip():
            raise RuntimeError(
                f"Parsed document is empty: {uploaded_file_path}"
            )
        return text.strip()

    # ── Path 2: Knowledge-base look-up ───────────────────
    if role_id and role_id.strip():
        if kb_index is None:
            raise ValueError(
                "A KnowledgeBaseIndex must be provided when looking up "
                "a role by id."
            )
        role = kb_index.get_by_id(role_id.strip())
        if role is None:
            raise ValueError(
                f"Role ID '{role_id}' not found in the knowledge base."
            )
        print(f"[Router] Found role '{role.get('Title')}' ({role.get('ExperienceLevel')})")
        return _extract_jd_text_from_role(role)

    # ── Path 3: Nothing provided ─────────────────────────
    raise ValueError(
        "At least one of 'uploaded_file_path' or 'role_id' must be provided."
    )
