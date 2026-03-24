"""
analyzer.py — Phase 2: Master Orchestrator
===========================================
Provides the ``analyze_resume`` function that calls all three Phase 2
components in sequence and returns a single unified output:

  1. **NER Extraction** — extract entities from resume and JD text.
  2. **O*NET Normalization** — map extracted skills to O*NET taxonomy
     with domain-aware filtering.
  3. **LLM Pros & Cons** — intelligent strengths / weaknesses analysis (Ollama / Llama 3).

Public API:
    analyze_resume(resume_text, jd_text, domain, level)
        → unified result dictionary
"""

from __future__ import annotations

from typing import Any, Optional

from src.ner_extractor import extract_entities
from src.onet_normalizer import (
    OnetTaxonomy,
    load_onet_taxonomy,
    normalize_skills,
)
from src.gpt_pros_cons import extract_pros_cons


# Module-level taxonomy cache (loaded once per process)
_taxonomy_cache: OnetTaxonomy | None = None


def _get_taxonomy() -> OnetTaxonomy:
    """Load the O*NET taxonomy once and cache it."""
    global _taxonomy_cache
    if _taxonomy_cache is None:
        _taxonomy_cache = load_onet_taxonomy()
    return _taxonomy_cache


def analyze_resume(
    resume_text: str,
    jd_text: str,
    domain: str = "Engineering",
    level: str = "Fresher",
    skip_gpt: bool = False,
) -> dict[str, Any]:
    """Run the full Phase 2 analysis pipeline.

    Calls each component in sequence:
      1. NER entity extraction on both resume and JD text.
      2. O*NET normalization of extracted skills with domain filtering.
      3. LLM pros / cons analysis (can be skipped for testing).

    Args:
        resume_text: Plain-text content of the candidate's resume.
        jd_text:     Plain-text content of the job description.
        domain:      Job domain — ``"Engineering"``, ``"Commerce"``,
                     or ``"Journalism"``.
        level:       Experience level — ``"Fresher"``, ``"Experienced"``,
                     ``"Senior"``, etc.
        skip_gpt:    If ``True``, skip the LLM call (useful when
                     Ollama is not running or during offline tests).

    Returns:
        A unified dictionary with the following structure::

            {
                "domain": str,
                "level": str,
                "resume_entities": { skills: [...], tools: [...], ... },
                "jd_entities":     { skills: [...], tools: [...], ... },
                "onet_resume_skills": { matched: [...], unmatched: [...] },
                "onet_jd_skills":    { matched: [...], unmatched: [...] },
                "skill_gap": {
                    "jd_skills_missing_in_resume": [...],
                    "resume_extra_skills": [...],
                },
                "gpt_pros_cons": { pros: [...], cons: [...] },
            }
    """
    print(f"\n{'='*60}")
    print(f"  ScopeWise — Phase 2 Analysis")
    print(f"  Domain: {domain}  |  Level: {level}")
    print(f"{'='*60}\n")

    # ──────────────────────────────────────────
    # Step 1: NER Entity Extraction
    # ──────────────────────────────────────────
    print("▶ Step 1 / 3 — NER Entity Extraction")
    resume_entities = extract_entities(resume_text)
    jd_entities = extract_entities(jd_text)

    print(f"  Resume entities: { {k: len(v) for k, v in resume_entities.items()} }")
    print(f"  JD entities:     { {k: len(v) for k, v in jd_entities.items()} }")

    # ──────────────────────────────────────────
    # Step 2: O*NET Normalization
    # ──────────────────────────────────────────
    print("\n▶ Step 2 / 3 — O*NET Skill Normalization")
    taxonomy = _get_taxonomy()

    # Combine skills + tools for O*NET matching (most meaningful categories)
    resume_raw_skills = resume_entities["skills"] + resume_entities["tools"]
    jd_raw_skills = jd_entities["skills"] + jd_entities["tools"]

    onet_resume = normalize_skills(resume_raw_skills, taxonomy, domain)
    onet_jd = normalize_skills(jd_raw_skills, taxonomy, domain)

    print(f"  Resume: {len(onet_resume['matched'])} matched, {len(onet_resume['unmatched'])} unmatched")
    print(f"  JD:     {len(onet_jd['matched'])} matched, {len(onet_jd['unmatched'])} unmatched")

    # ── Compute skill gap ────────────────────
    resume_onet_names = {
        m["onet_match"] for m in onet_resume["matched"]
    }
    jd_onet_names = {
        m["onet_match"] for m in onet_jd["matched"]
    }

    jd_missing_in_resume = sorted(jd_onet_names - resume_onet_names)
    resume_extra = sorted(resume_onet_names - jd_onet_names)

    print(f"  Skill gap — JD skills missing in resume: {len(jd_missing_in_resume)}")
    print(f"  Skill gap — Resume extra skills:         {len(resume_extra)}")

    # ──────────────────────────────────────────
    # Step 3: LLM Pros & Cons (Ollama / Llama 3)
    # ──────────────────────────────────────────
    gpt_result: dict[str, Any] = {"pros": [], "cons": []}

    if skip_gpt:
        print("\n▶ Step 3 / 3 — LLM Pros & Cons [SKIPPED]")
    else:
        print("\n▶ Step 3 / 3 — LLM Pros & Cons")
        try:
            gpt_result = extract_pros_cons(resume_text, jd_text, domain, level)
        except (EnvironmentError, RuntimeError) as exc:
            print(f"  ⚠ LLM analysis failed: {exc}")
            gpt_result = {
                "pros": [],
                "cons": [],
                "_error": str(exc),
            }

    # ──────────────────────────────────────────
    # Assemble unified output
    # ──────────────────────────────────────────
    output: dict[str, Any] = {
        "domain": domain,
        "level": level,
        "resume_entities": resume_entities,
        "jd_entities": jd_entities,
        "onet_resume_skills": onet_resume,
        "onet_jd_skills": onet_jd,
        "skill_gap": {
            "jd_skills_missing_in_resume": jd_missing_in_resume,
            "resume_extra_skills": resume_extra,
        },
        "gpt_pros_cons": gpt_result,
    }

    print(f"\n{'='*60}")
    print("  ✅ Phase 2 analysis complete.")
    print(f"{'='*60}\n")

    return output
