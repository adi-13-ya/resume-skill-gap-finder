#!/usr/bin/env python3
"""
run_analysis.py — Run ScopeWise on your own resume PDF/DOCX
============================================================
Usage:
    # Analyse against a Kaggle KB role (by JobID):
    python run_analysis.py --resume path/to/resume.pdf --role-id NET-F-001

    # Analyse against a custom JD file (PDF or DOCX):
    python run_analysis.py --resume path/to/resume.pdf --jd path/to/jd.pdf

    # Specify domain and level (defaults: Engineering, Fresher):
    python run_analysis.py --resume resume.pdf --role-id NET-E-011 --domain Engineering --level Experienced

    # Skip LLM step (if Ollama is not running):
    python run_analysis.py --resume resume.pdf --role-id NET-F-001 --skip-gpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.document_parser import parse_document
from src.kb_loader import KnowledgeBaseIndex
from src.routing import get_jd_text
from src.analyzer import analyze_resume


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScopeWise — Analyse a resume against a job description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_analysis.py --resume my_resume.pdf --role-id NET-F-001
  python run_analysis.py --resume my_resume.pdf --jd job_description.pdf
  python run_analysis.py --resume my_resume.pdf --role-id NET-E-011 --domain Engineering --level Experienced --skip-gpt
        """,
    )

    parser.add_argument(
        "--resume", required=True,
        help="Path to your resume file (PDF or DOCX).",
    )
    parser.add_argument(
        "--jd", default=None,
        help="Path to a JD file (PDF or DOCX). If omitted, use --role-id instead.",
    )
    parser.add_argument(
        "--role-id", default=None,
        help="JobID from the Kaggle knowledge base (e.g. NET-F-001). "
             "Ignored if --jd is provided.",
    )
    parser.add_argument(
        "--domain", default="Engineering",
        help="Job domain (default: Engineering). "
             "Examples: Engineering, Commerce, Journalism, Healthcare, etc.",
    )
    parser.add_argument(
        "--level", default="Fresher",
        help="Experience level (default: Fresher). "
             "Examples: Fresher, Experienced, Senior, Mid-Level, Lead.",
    )
    parser.add_argument(
        "--skip-gpt", action="store_true",
        help="Skip the LLM pros/cons step (useful when Ollama is not running).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save full JSON output to this file (optional).",
    )

    args = parser.parse_args()

    # ── Validate resume path ─────────────────
    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"❌ Resume file not found: {resume_path}")
        sys.exit(1)

    # ── Parse resume ─────────────────────────
    print(f"\n📄 Parsing resume: {resume_path}")
    resume_text = parse_document(str(resume_path))
    print(f"   Extracted {len(resume_text)} characters.\n")

    # ── Get JD text ──────────────────────────
    if args.jd:
        jd_path = Path(args.jd)
        if not jd_path.exists():
            print(f"❌ JD file not found: {jd_path}")
            sys.exit(1)
        print(f"📄 Parsing JD: {jd_path}")
        jd_text = parse_document(str(jd_path))
    elif args.role_id:
        print(f"🔍 Looking up role: {args.role_id}")
        kb = KnowledgeBaseIndex().load_all()
        jd_text = get_jd_text(role_id=args.role_id, kb_index=kb)
    else:
        print("❌ You must provide either --jd or --role-id.")
        parser.print_help()
        sys.exit(1)

    print(f"   JD text: {len(jd_text)} characters.\n")

    # ── Run analysis ─────────────────────────
    result = analyze_resume(
        resume_text=resume_text,
        jd_text=jd_text,
        domain=args.domain,
        level=args.level,
        skip_gpt=args.skip_gpt,
    )

    # ── Print results ────────────────────────
    divider = "═" * 60

    print(f"\n{divider}")
    print("  RESUME ENTITIES")
    print(divider)
    for category, items in result["resume_entities"].items():
        if items:
            print(f"  {category}: {', '.join(items)}")

    print(f"\n{divider}")
    print("  JD ENTITIES")
    print(divider)
    for category, items in result["jd_entities"].items():
        if items:
            print(f"  {category}: {', '.join(items)}")

    print(f"\n{divider}")
    print("  O*NET SKILL MATCHES (Resume)")
    print(divider)
    for m in result["onet_resume_skills"]["matched"]:
        flag = "✓" if m["domain_relevant"] else "✗"
        print(f"  {flag} {m['raw']} → {m['onet_match']} (score: {m['score']}, {m['category']})")
    if result["onet_resume_skills"]["unmatched"]:
        print(f"\n  Unmatched: {', '.join(result['onet_resume_skills']['unmatched'])}")

    print(f"\n{divider}")
    print("  SKILL GAP")
    print(divider)
    gap = result["skill_gap"]
    if gap["jd_skills_missing_in_resume"]:
        print("  Missing from resume (required by JD):")
        for s in gap["jd_skills_missing_in_resume"]:
            print(f"    ⚠ {s}")
    else:
        print("  No critical skill gaps detected!")
    if gap["resume_extra_skills"]:
        print("  Extra skills in resume (not in JD):")
        for s in gap["resume_extra_skills"]:
            print(f"    + {s}")

    print(f"\n{divider}")
    print("  LLM PROS & CONS")
    print(divider)
    gpt = result["gpt_pros_cons"]
    if gpt.get("_error"):
        print(f"  ⚠ Skipped or failed: {gpt['_error'][:100]}...")
    else:
        if gpt.get("pros"):
            print("  PROS:")
            for p in gpt["pros"]:
                print(f"    ✅ {p.get('point', p)}")
                if isinstance(p, dict) and p.get("justification"):
                    print(f"       {p['justification']}")
        if gpt.get("cons"):
            print("  CONS:")
            for c in gpt["cons"]:
                print(f"    ❌ {c.get('point', c)}")
                if isinstance(c, dict) and c.get("justification"):
                    print(f"       {c['justification']}")

    # ── Save JSON output ─────────────────────
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Full results saved to: {out_path}")

    print()


if __name__ == "__main__":
    main()
