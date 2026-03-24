#!/usr/bin/env python3
"""
test_phase2.py — Phase 2: End-to-End Test Script
=================================================
Loads sample resume and JD text, runs the full Phase 2 pipeline
(NER extraction → O*NET normalization → LLM pros/cons), and
pretty-prints the unified output to the console.

The test uses inline sample text (no external files needed).

If Ollama is not reachable, the LLM step is gracefully skipped.

Run from the project root:
    python test_phase2.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analyzer import analyze_resume


# ──────────────────────────────────────────────
# Sample data
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

SAMPLE_JD = """\
Role: Senior Software Engineer
Experience Level: Experienced
Years of Experience: 5+

We are looking for a Senior Software Engineer to lead backend development
for our cloud-native platform. You will architect scalable microservices,
define API contracts, and own end-to-end delivery of features.

Required Skills:
Python, Java, System Design, Microservices Architecture, REST APIs,
AWS or GCP, Docker, Kubernetes, CI/CD, SQL & NoSQL Databases,
Performance Optimization, Monitoring (Prometheus, Grafana)

Key Responsibilities:
• Design and implement scalable microservice architectures
• Lead technical design reviews and code reviews
• Mentor junior and mid-level engineers
• Build and maintain CI/CD pipelines
• Define SLOs and monitor system performance
• Collaborate with product management on technical feasibility
• Drive engineering best practices across the team

Required Qualifications:
• B.Tech/M.Tech in Computer Science or equivalent
• 5+ years of professional software engineering experience
• Deep expertise in Python or Java
• Hands-on experience with Kubernetes and Docker
• Strong system design and architecture skills
• Experience with observability tools (Prometheus, Grafana, Datadog)

Keywords: Python, Java, Microservices, Kubernetes, Docker, AWS, CI/CD,
System Design, REST APIs, PostgreSQL, MongoDB, Grafana, Prometheus
"""


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
    print("\n🚀  ScopeWise — Phase 2: Domain-Aware Skill Extraction Tests")
    print("=" * 72)

    # Decide whether to call the LLM (check if Ollama is reachable)
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        has_ollama = True
    except Exception:
        has_ollama = False
    skip_gpt = not has_ollama
    if skip_gpt:
        print("\n⚠  Ollama not reachable — LLM step will be skipped.")
        print("   Start Ollama to enable pros/cons extraction.\n")

    # ── Run full pipeline ────────────────────
    result = analyze_resume(
        resume_text=SAMPLE_RESUME,
        jd_text=SAMPLE_JD,
        domain="Engineering",
        level="Experienced",
        skip_gpt=skip_gpt,
    )

    # ── Pretty-print results section by section ──

    _section("1. Resume Entities (NER)")
    _pp(result["resume_entities"])

    _section("2. JD Entities (NER)")
    _pp(result["jd_entities"])

    _section("3. O*NET Normalized Resume Skills")
    _pp(result["onet_resume_skills"])

    _section("4. O*NET Normalized JD Skills")
    _pp(result["onet_jd_skills"])

    _section("5. Skill Gap Analysis")
    _pp(result["skill_gap"])

    _section("6. LLM Pros & Cons")
    if skip_gpt:
        print("  [Skipped — Ollama not reachable]")
    else:
        _pp(result["gpt_pros_cons"])

    _section("ALL PHASE 2 TESTS COMPLETED")
    print("Pipeline executed successfully.\n")


if __name__ == "__main__":
    main()
