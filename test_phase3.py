#!/usr/bin/env python3
"""
test_phase3.py — Phase 3: End-to-End Test Script
==================================================
Loads a real role from the Kaggle knowledge base, runs the full Phase 2
pipeline to generate real entity dictionaries, then immediately passes
that output (in memory) to Phase 3's ``evaluate_resume`` function.

Pretty-prints the complete Phase 3 output including all scores, the
full structural analysis JSON, and the final fuzzy verdict.

Run from the project root:
    python test_phase3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work correctly
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analyzer import analyze_resume
from src.phase3_evaluator import evaluate_resume
from src.kb_loader import KnowledgeBaseIndex
from src.routing import get_jd_text


# ──────────────────────────────────────────────
# Sample resume text (same realistic resume used in Phase 2 tests)
# ──────────────────────────────────────────────

SAMPLE_RESUME = """\
JANE DOE
Software Engineer | jane.doe@email.com | github.com/janedoe

SUMMARY
Passionate software engineer with 3 years of experience building scalable
backend systems in Python and Java. Strong foundation in data structures,
algorithms, and system design. Experienced with cloud infrastructure on
AWS and containerized deployments using Docker and Kubernetes.

TECHNICAL SKILLS
Languages: Python, Java, JavaScript, SQL
Frameworks: Django, Flask, Spring Boot, React
Databases: PostgreSQL, MongoDB, Redis
Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, Kubernetes, Jenkins, Git
Tools: Jira, Confluence, Postman, Grafana

EXPERIENCE
Backend Engineer — Acme Technologies (Jan 2022 – Present)
• Designed and built microservice APIs handling 5M+ requests/day using
  Django REST Framework and PostgreSQL.
• Implemented CI/CD pipelines with Jenkins and GitHub Actions, reducing
  deployment time by 60%.
• Led migration from monolithic architecture to Kubernetes-based
  microservices, improving system uptime from 99.5% to 99.95%.
• Mentored 2 junior engineers through code reviews and pair programming.

Junior Developer — StartupXYZ (Jun 2020 – Dec 2021)
• Built RESTful APIs for an e-commerce platform using Flask and MongoDB.
• Wrote comprehensive unit and integration tests, achieving 85% code coverage.
• Integrated Stripe payment gateway and SendGrid email service.

EDUCATION
B.Tech in Computer Science — Indian Institute of Technology Delhi (2020)

CERTIFICATIONS
AWS Certified Solutions Architect - Associate
"""

# The role ID to look up from the Kaggle dataset.
# NET-E-013 is an Experienced .NET Developer (5+ years) — a reasonable
# test case since the resume candidate is a Python/Java engineer applying
# for a .NET role (partial mismatch scenario).
SAMPLE_ROLE_ID = "NET-E-013"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

_DIVIDER = "═" * 72


def _section(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def _pp(obj: object, indent: int = 2) -> None:
    """Pretty-print a Python object as JSON."""
    print(json.dumps(obj, indent=indent, default=str))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    print("\n🚀  ScopeWise — Phase 3: Fuzzy Logic & Structural Evaluation Tests")
    print("=" * 72)

    # ── Step 0: Load the Kaggle knowledge base ───────────
    _section("0. Loading Knowledge Base")
    kb = KnowledgeBaseIndex().load_all()

    # Look up the chosen role to get JD metadata
    role = kb.get_by_id(SAMPLE_ROLE_ID)
    if role is None:
        print(f"  ❌ Role ID '{SAMPLE_ROLE_ID}' not found in the KB.")
        sys.exit(1)

    print(f"  Role: {role.get('Title')} ({role.get('ExperienceLevel')})")
    print(f"  YoE:  {role.get('YearsOfExperience')}")

    # Synthesise the JD text from the KB role record
    jd_text = get_jd_text(role_id=SAMPLE_ROLE_ID, kb_index=kb)

    # ── Step 1: Run Phase 2 to get real entities ─────────
    _section("1. Running Phase 2 Pipeline (NER + O*NET + LLM Pros/Cons)")

    # Check if Ollama is reachable for Phase 2's LLM component
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        has_ollama = True
    except Exception:
        has_ollama = False

    if not has_ollama:
        print("  ⚠  Ollama not reachable — Phase 2 LLM step will be skipped.")

    phase2_result = analyze_resume(
        resume_text=SAMPLE_RESUME,
        jd_text=jd_text,
        domain="Engineering",
        level=role.get("ExperienceLevel", "Experienced"),
        skip_gpt=not has_ollama,
    )

    # Display Phase 2 entity summaries for context
    _section("2. Phase 2 Output Summary")
    resume_ents = phase2_result["resume_entities"]
    jd_ents = phase2_result["jd_entities"]
    print("  Resume entities:")
    for cat, items in resume_ents.items():
        if items:
            print(f"    {cat}: {', '.join(items[:8])}{'…' if len(items) > 8 else ''}")
    print("  JD entities:")
    for cat, items in jd_ents.items():
        if items:
            print(f"    {cat}: {', '.join(items[:8])}{'…' if len(items) > 8 else ''}")

    # ── Step 2: Run Phase 3 with Phase 2 output ──────────
    _section("3. Running Phase 3 Pipeline")

    # Build the JD metadata dict from the knowledge base record.
    # This includes fields like YearsOfExperience, Title, ExperienceLevel
    # that Phase 3's Component 2 needs for alignment scoring.
    jd_metadata = {
        "Title": role.get("Title", ""),
        "ExperienceLevel": role.get("ExperienceLevel", ""),
        "YearsOfExperience": role.get("YearsOfExperience", "0"),
        "Skills": role.get("Skills", []),
        "Keywords": role.get("Keywords", []),
    }

    # Pass Phase 2 output DIRECTLY IN MEMORY to Phase 3
    phase3_result = evaluate_resume(
        resume_entities=resume_ents,
        jd_entities=jd_ents,
        jd_metadata=jd_metadata,
        resume_text=SAMPLE_RESUME,
    )

    # ── Pretty-print Phase 3 results ─────────────────────
    _section("4. Phase 3 Results — Scores")
    print(f"  Semantic Similarity Score:     {phase3_result['similarity_score']}")
    print(f"  Experience+Education Score:    {phase3_result['experience_score']}")
    print(f"  Structural Score (normalized): {phase3_result['structural_score']}")
    print(f"  Fuzzy Crisp Score:             {phase3_result['fuzzy_crisp_score']}")
    print(f"  Final Verdict:                 {phase3_result['verdict']}")

    _section("5. Phase 3 Results — Structural Analysis (LLM)")
    _pp(phase3_result["structural_analysis"])

    _section("6. Phase 3 Results — Full Output Dictionary")
    _pp(phase3_result)

    _section("ALL PHASE 3 TESTS COMPLETED")
    print("Pipeline executed successfully.\n")


if __name__ == "__main__":
    main()
