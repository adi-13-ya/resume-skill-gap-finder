"""
kb_loader.py — Phase 1: Knowledge Base Loader Module
=====================================================
Provides functions to:
  1. Load the Kaggle dataset (JSON format).
  2. Index all roles into a unified in-memory store.
  3. Retrieve a specific role by its JobID or title.

The index maps every role by both its ``JobID`` and a normalised version
of its title, so look-ups by either key are O(1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────
# Default path (project-root-relative)
# ──────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KAGGLE_JSON_PATH = _PROJECT_ROOT / "data" / "job_dataset.json"


# ──────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────

def load_kaggle_dataset(path: Optional[str] = None) -> list[dict[str, Any]]:
    """Load the Kaggle job-description dataset from JSON.

    Args:
        path: Explicit path to the dataset JSON file.  When *None*,
              defaults to ``data/job_dataset.json`` in the project root.

    Returns:
        List of role dictionaries.

    Raises:
        FileNotFoundError: If the dataset file cannot be found.
    """
    dataset_path = Path(path) if path else _DEFAULT_KAGGLE_JSON_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at: {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as fh:
        records: list[dict[str, Any]] = json.load(fh)

    return records


# ──────────────────────────────────────────────
# Unified Index
# ──────────────────────────────────────────────

class KnowledgeBaseIndex:
    """In-memory index over the Kaggle job-description dataset.

    Look-up is supported by:
      • JobID   (exact, case-insensitive)
      • Title   (normalised lowercase)
    """

    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_title: dict[str, list[dict[str, Any]]] = {}
        self._all_roles: list[dict[str, Any]] = []

    # ── Indexing helpers ─────────────────────

    def _normalise(self, text: str) -> str:
        """Lowercase & strip for consistent look-ups."""
        return text.strip().lower()

    def _index_record(self, record: dict[str, Any]) -> None:
        """Add a single record to both the id and title indices."""
        # ID key
        job_id = record.get("JobID") or ""
        if job_id:
            self._by_id[self._normalise(job_id)] = record

        # Title key — may have duplicates (e.g. Fresher vs Senior)
        title = record.get("Title") or ""
        if title:
            norm_title = self._normalise(title)
            self._by_title.setdefault(norm_title, []).append(record)

        self._all_roles.append(record)

    # ── Public methods ───────────────────────

    def load_all(
        self,
        kaggle_path: Optional[str] = None,
    ) -> "KnowledgeBaseIndex":
        """Load the Kaggle dataset into the index.

        Args:
            kaggle_path: Optional explicit path to the JSON file.

        Returns:
            ``self`` (for fluent chaining).
        """
        records = load_kaggle_dataset(kaggle_path)
        for rec in records:
            self._index_record(rec)
        print(f"[KB] Loaded {len(records)} roles from Kaggle dataset.")
        print(f"[KB] Total roles indexed: {len(self._all_roles)}")
        return self

    def get_by_id(self, role_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a role by its ``JobID`` (case-insensitive).

        Args:
            role_id: The unique JobID for the role.

        Returns:
            The matching role dict, or *None* if not found.
        """
        return self._by_id.get(self._normalise(role_id))

    def get_by_title(
        self,
        title: str,
        level: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve roles matching a title (case-insensitive).

        Because multiple entries may share the same title (e.g. Fresher
        vs Senior), a list is returned.

        Args:
            title: Role title to look up.
            level: Optional experience level filter
                   (e.g. ``"Fresher"`` or ``"Senior"``).

        Returns:
            List of matching role dicts (may be empty).
        """
        matches = self._by_title.get(self._normalise(title), [])

        if level:
            norm_level = self._normalise(level)
            matches = [
                r for r in matches
                if self._normalise(r.get("ExperienceLevel") or "") == norm_level
            ]

        return matches

    def search(self, query: str) -> list[dict[str, Any]]:
        """Simple substring search across JobIDs and titles.

        Args:
            query: Search term.

        Returns:
            List of matching role dicts.
        """
        q = self._normalise(query)
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for record in self._all_roles:
            rid = self._normalise(record.get("JobID") or "")
            title = self._normalise(record.get("Title") or "")
            if q in rid or q in title:
                if rid not in seen_ids:
                    results.append(record)
                    seen_ids.add(rid)

        return results

    @property
    def total_roles(self) -> int:
        """Total number of indexed roles."""
        return len(self._all_roles)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the indexed dataset."""
        levels: dict[str, int] = {}
        for rec in self._all_roles:
            lvl = rec.get("ExperienceLevel", "unknown")
            levels[lvl] = levels.get(lvl, 0) + 1
        return {"total": len(self._all_roles), "by_level": levels}
