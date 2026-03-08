"""
onet_normalizer.py — Phase 2, Component 2: O*NET Domain Normalization
======================================================================
Uses the O*NET database (Skills, Knowledge, Abilities tables) as the
authoritative skill taxonomy.  Raw extracted skills are mapped to their
closest O*NET category via fuzzy string matching (``rapidfuzz``).

Key capabilities:
  • Load O*NET tab-separated text files from ``data/onet/``.
  • Build a unified taxonomy of (Element Name → category, SOC codes).
  • Filter normalised results by domain context (Engineering, Commerce,
    Journalism) using SOC occupation-code prefixes.
  • Return domain-relevant normalised skill mappings with match scores.

Public API:
    load_onet_taxonomy()           → OnetTaxonomy
    normalize_skill(raw, taxonomy) → best match dict
    normalize_skills(raw_list, taxonomy, domain) → list of match dicts
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rapidfuzz import fuzz, process


# ──────────────────────────────────────────────────────
# Default O*NET data path
# ──────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ONET_DIR = _PROJECT_ROOT / "data" / "onet" / "db_29_1_text"


# ──────────────────────────────────────────────────────
# SOC code → domain mapping
# ──────────────────────────────────────────────────────
# The Standard Occupational Classification (SOC) system uses 2-digit
# major-group codes.  We map ScopeWise domains to relevant SOC groups.

DOMAIN_SOC_PREFIXES: dict[str, list[str]] = {
    "Engineering": [
        "15",  # Computer and Mathematical Occupations
        "17",  # Architecture and Engineering Occupations
        "19",  # Life, Physical, and Social Science Occupations
        "11",  # Management (tech leadership, CTO, VP Eng …)
    ],
    "Commerce": [
        "11",  # Management Occupations
        "13",  # Business and Financial Operations
        "41",  # Sales and Related Occupations
        "43",  # Office and Administrative Support
    ],
    "Journalism": [
        "27",  # Arts, Design, Entertainment, Sports, Media
        "25",  # Educational Instruction and Library
        "11",  # Management (editorial directors …)
        "23",  # Legal (media law …)
    ],
}


# ──────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────

@dataclass
class OnetElement:
    """A single O*NET taxonomy element (skill / knowledge / ability)."""
    element_id: str
    element_name: str
    category: str            # "Skill", "Knowledge", or "Ability"
    soc_codes: set[str] = field(default_factory=set)
    avg_importance: float = 0.0  # average IM score across occupations


@dataclass
class OnetTaxonomy:
    """Complete O*NET taxonomy loaded from disk.

    Attributes:
        elements:       Mapping of element name → OnetElement.
        element_names:  Flat sorted list of all unique element names
                        (used as the search space for fuzzy matching).
        soc_to_title:   Mapping of SOC code → occupation title.
    """
    elements: dict[str, OnetElement] = field(default_factory=dict)
    element_names: list[str] = field(default_factory=list)
    soc_to_title: dict[str, str] = field(default_factory=dict)


# ──────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────

def _load_onet_table(
    filepath: Path,
    category: str,
    taxonomy: OnetTaxonomy,
) -> int:
    """Parse a single O*NET table file and merge elements into the taxonomy.

    Args:
        filepath:  Path to the tab-separated O*NET file.
        category:  Label for the category ("Skill" / "Knowledge" / "Ability").
        taxonomy:  Taxonomy object to populate (mutated in-place).

    Returns:
        Number of rows processed.
    """
    if not filepath.exists():
        print(f"[O*NET] Warning — file not found: {filepath}")
        return 0

    row_count = 0
    with open(filepath, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            name = row.get("Element Name", "").strip()
            soc = row.get("O*NET-SOC Code", "").strip()
            scale_id = row.get("Scale ID", "").strip()

            if not name or not soc:
                continue

            # Only use IM (Importance) scale rows for average scoring
            if scale_id != "IM":
                continue

            if name not in taxonomy.elements:
                taxonomy.elements[name] = OnetElement(
                    element_id=row.get("Element ID", ""),
                    element_name=name,
                    category=category,
                )

            elem = taxonomy.elements[name]
            elem.soc_codes.add(soc)

            # Running average of importance
            try:
                val = float(row.get("Data Value", 0))
                n = len(elem.soc_codes)
                elem.avg_importance = (
                    elem.avg_importance * (n - 1) + val
                ) / n
            except (ValueError, ZeroDivisionError):
                pass

            row_count += 1

    return row_count


def load_onet_taxonomy(onet_dir: Optional[str] = None) -> OnetTaxonomy:
    """Load and merge Skills, Knowledge, and Abilities from O*NET.

    Args:
        onet_dir: Path to the extracted O*NET text-file directory.
                  Defaults to ``data/onet/db_29_1_text/``.

    Returns:
        A populated ``OnetTaxonomy`` object.

    Raises:
        FileNotFoundError: If the O*NET directory does not exist.
    """
    onet_path = Path(onet_dir) if onet_dir else _DEFAULT_ONET_DIR

    if not onet_path.exists():
        raise FileNotFoundError(
            f"O*NET data directory not found: {onet_path}\n"
            "Download from https://www.onetcenter.org/database.html "
            "and extract into data/onet/"
        )

    taxonomy = OnetTaxonomy()

    # Load the three core tables
    tables = [
        ("Skills.txt", "Skill"),
        ("Knowledge.txt", "Knowledge"),
        ("Abilities.txt", "Ability"),
    ]

    total = 0
    for filename, category in tables:
        count = _load_onet_table(onet_path / filename, category, taxonomy)
        print(f"[O*NET] Loaded {count:,} IM rows from {filename}")
        total += count

    # Build the flat name list for fuzzy matching
    taxonomy.element_names = sorted(taxonomy.elements.keys())

    # Load occupation titles for domain filtering context
    occ_file = onet_path / "Occupation Data.txt"
    if occ_file.exists():
        with open(occ_file, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                soc = row.get("O*NET-SOC Code", "").strip()
                title = row.get("Title", "").strip()
                if soc and title:
                    taxonomy.soc_to_title[soc] = title

    print(
        f"[O*NET] Taxonomy ready — {len(taxonomy.elements)} unique elements, "
        f"{len(taxonomy.soc_to_title)} occupations."
    )
    return taxonomy


# ──────────────────────────────────────────────────────
# Normalization (fuzzy matching)
# ──────────────────────────────────────────────────────

def normalize_skill(
    raw_skill: str,
    taxonomy: OnetTaxonomy,
    score_cutoff: int = 65,
) -> Optional[dict[str, Any]]:
    """Map a single raw skill string to its closest O*NET element.

    Uses ``rapidfuzz.process.extractOne`` with weighted-ratio scoring.

    Args:
        raw_skill:     Raw extracted skill string.
        taxonomy:      Loaded O*NET taxonomy.
        score_cutoff:  Minimum fuzzy match score (0–100) to accept.

    Returns:
        A dict with ``raw``, ``onet_match``, ``category``, ``score``,
        ``avg_importance`` — or *None* if no match above the cutoff.
    """
    if not raw_skill.strip() or not taxonomy.element_names:
        return None

    result = process.extractOne(
        raw_skill.strip(),
        taxonomy.element_names,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )

    if result is None:
        return None

    matched_name, score, _idx = result
    elem = taxonomy.elements[matched_name]

    return {
        "raw": raw_skill.strip(),
        "onet_match": matched_name,
        "category": elem.category,
        "score": round(score, 1),
        "avg_importance": round(elem.avg_importance, 2),
    }


def _get_domain_soc_prefixes(domain: Optional[str]) -> Optional[set[str]]:
    """Return the set of 2-digit SOC prefixes for a domain, or None."""
    if not domain:
        return None
    prefixes = DOMAIN_SOC_PREFIXES.get(domain)
    if prefixes:
        return set(prefixes)
    return None


def _is_domain_relevant(
    element: OnetElement,
    soc_prefixes: Optional[set[str]],
) -> bool:
    """Check if an O*NET element is relevant to the given domain.

    An element is considered domain-relevant if at least one of its
    associated SOC codes starts with one of the domain's SOC prefixes.
    If no domain filter is given, everything is relevant.
    """
    if soc_prefixes is None:
        return True
    return any(
        soc[:2] in soc_prefixes
        for soc in element.soc_codes
    )


def normalize_skills(
    raw_skills: list[str],
    taxonomy: OnetTaxonomy,
    domain: Optional[str] = None,
    score_cutoff: int = 65,
) -> dict[str, list[dict[str, Any]]]:
    """Normalise a list of raw skills against O*NET with domain filtering.

    Args:
        raw_skills:    List of raw extracted skill strings.
        taxonomy:      Loaded O*NET taxonomy.
        domain:        Domain context (``"Engineering"``, ``"Commerce"``,
                       ``"Journalism"``).  ``None`` means no filtering.
        score_cutoff:  Minimum fuzzy match score to accept.

    Returns:
        A dictionary with:
          - ``"matched"``   — list of successfully normalised skills
          - ``"unmatched"`` — list of raw skills with no O*NET match
    """
    soc_prefixes = _get_domain_soc_prefixes(domain)

    matched: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for raw in raw_skills:
        match = normalize_skill(raw, taxonomy, score_cutoff)
        if match is None:
            unmatched.append(raw)
            continue

        # Check domain relevance
        elem = taxonomy.elements.get(match["onet_match"])
        if elem and not _is_domain_relevant(elem, soc_prefixes):
            # Still include the match but flag it as out-of-domain
            match["domain_relevant"] = False
        else:
            match["domain_relevant"] = True

        matched.append(match)

    return {"matched": matched, "unmatched": unmatched}
