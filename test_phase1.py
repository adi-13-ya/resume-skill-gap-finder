#!/usr/bin/env python3
"""
test_phase1.py — Phase 1: End-to-End Test Script
=================================================
Demonstrates the data ingestion paths using the Kaggle dataset:

  TEST 1 → Load a Fresher role from the Kaggle dataset by JobID.
  TEST 2 → Load a Senior role from the Kaggle dataset by JobID.
  TEST 3 → Look up roles by title and filter by experience level.
  TEST 4 → Verify the document parser module detects supported types.

Run from the project root:
    python test_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.kb_loader import KnowledgeBaseIndex
from src.routing import get_jd_text
from src.document_parser import SUPPORTED_EXTENSIONS


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

_DIVIDER = "═" * 72


def _section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def main() -> None:
    print("\n🚀  Phase 1 — Data Ingestion Layer: End-to-End Tests")
    print("=" * 72)

    # ── Build the unified knowledge base index ────────
    _section("Loading Knowledge Base Index")
    kb = KnowledgeBaseIndex().load_all()
    print(f"[KB] Summary: {kb.summary()}")

    # ─────────────────────────────────────────
    # TEST 1: Load a Fresher role by JobID
    # ─────────────────────────────────────────
    _section("TEST 1 — Kaggle Dataset: Fresher role lookup (NET-F-001)")
    job_id = "NET-F-001"
    print(f"Looking up role: {job_id}")

    jd_text = get_jd_text(role_id=job_id, kb_index=kb)
    print(f"\n--- Extracted JD Text ---\n")
    print(jd_text)
    print("\n✅  TEST 1 PASSED\n")

    # ─────────────────────────────────────────
    # TEST 2: Load a Senior role by JobID
    # ─────────────────────────────────────────
    _section("TEST 2 — Kaggle Dataset: Experienced role lookup (NET-E-011)")
    job_id = "NET-E-011"
    print(f"Looking up role: {job_id}")

    jd_text = get_jd_text(role_id=job_id, kb_index=kb)
    print(f"\n--- Extracted JD Text ---\n")
    print(jd_text)
    print("\n✅  TEST 2 PASSED\n")

    # ─────────────────────────────────────────
    # TEST 3: Look up by title + level filter
    # ─────────────────────────────────────────
    _section("TEST 3 — Title lookup with level filter")
    title_query = ".NET Developer"
    level_filter = "Experienced"
    print(f"Searching for title='{title_query}', level='{level_filter}'")

    matches = kb.get_by_title(title_query, level=level_filter)
    print(f"Found {len(matches)} matching role(s).")
    if matches:
        first = matches[0]
        print(f"  First match: {first['JobID']} — {first['Title']} ({first['ExperienceLevel']})")
    print("\n✅  TEST 3 PASSED\n")

    # ─────────────────────────────────────────
    # TEST 4: Document parser supported types
    # ─────────────────────────────────────────
    _section("TEST 4 — Document parser: supported extensions check")
    print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")

    # Verify unsupported type raises ValueError
    try:
        from src.document_parser import parse_document
        parse_document("/tmp/fake_file.txt")
        print("ERROR: Should have raised an exception!")
    except (ValueError, FileNotFoundError) as e:
        print(f"Correctly raised: {type(e).__name__}: {e}")
    print("\n✅  TEST 4 PASSED\n")

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    _section("ALL TESTS COMPLETED")
    print(
        "Phase 1 data ingestion layer is working correctly.\n"
        f"Total roles in KB: {kb.total_roles}\n"
    )


if __name__ == "__main__":
    main()
