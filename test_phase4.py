#!/usr/bin/env python3
"""
test_phase4.py — Phase 4: End-to-End Test Script
==================================================
Loads a real role from the Kaggle knowledge base, runs the full Phase 2
and Phase 3 pipeline, then immediately passes that output (in memory)
to Phase 4's ``generate_career_advice`` function.

Pretty-prints the complete Phase 4 output.

Run from the project root:
    python test_phase4.py
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
from src.phase4_advisor import generate_career_advice
from src.kb_loader import KnowledgeBaseIndex
from src.routing import get_jd_text

# ──────────────────────────────────────────────
# Sample resume text (same realistic resume used in Phase 2 & 3 tests)
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

SAMPLE_ROLE_ID = "NET-E-013"

def _section(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")

def _pp(obj: object, indent: int = 2) -> None:
    """Pretty-print a Python object as JSON."""
    print(json.dumps(obj, indent=indent, default=str))

def main() -> None:
    print("\n🚀  ScopeWise — Phase 4: Final Recommendation & Advice Tests")
    print("=" * 72)

    _section("0. Loading Knowledge Base")
    kb = KnowledgeBaseIndex().load_all()
    role = kb.get_by_id(SAMPLE_ROLE_ID)
    if role is None:
        print(f"  ❌ Role ID '{SAMPLE_ROLE_ID}' not found in the KB.")
        sys.exit(1)

    print(f"  Role: {role.get('Title')} ({role.get('ExperienceLevel')})")
    jd_text = get_jd_text(role_id=SAMPLE_ROLE_ID, kb_index=kb)

    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        has_ollama = True
    except Exception:
        has_ollama = False

    if not has_ollama:
        print("  ⚠  Ollama not reachable — Phase 2/3/4 LLM steps might fail or mock.")

    _section("1. Running Phase 2 Pipeline (NER + O*NET + LLM Pros/Cons)")
    phase2_result = analyze_resume(
        resume_text=SAMPLE_RESUME,
        jd_text=jd_text,
        domain="Engineering",
        level=role.get("ExperienceLevel", "Experienced"),
        skip_gpt=not has_ollama,
    )

    _section("2. Running Phase 3 Pipeline")
    jd_metadata = {
        "Title": role.get("Title", ""),
        "ExperienceLevel": role.get("ExperienceLevel", ""),
        "YearsOfExperience": role.get("YearsOfExperience", "0"),
        "Skills": role.get("Skills", []),
        "Keywords": role.get("Keywords", []),
    }

    phase3_result = evaluate_resume(
        resume_entities=phase2_result["resume_entities"],
        jd_entities=phase2_result["jd_entities"],
        jd_metadata=jd_metadata,
        resume_text=SAMPLE_RESUME,
    )

    # Inject metadata into phase3 output specifically for phase 4
    phase3_result["jd_metadata"] = jd_metadata

    _section("3. Running Phase 4 Advisor Pipeline")
    phase4_result = generate_career_advice(
        phase2_output=phase2_result,
        phase3_output=phase3_result
    )

    _section("4. Phase 4 Results — Full Output JSON")
    _pp(phase4_result)

    _section("ALL PHASE 4 TESTS COMPLETED")
    print("Pipeline executed successfully.\n")

if __name__ == "__main__":
    main()
