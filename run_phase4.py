#!/usr/bin/env python3
"""
run_phase4.py — Run ScopeWise Phase 4 on your own resume PDF/DOCX
==================================================================
This wrapper uses Phase 1, 2, and 3 to parse and extract your custom
uploaded resume and Job Description, evaluate it, and finally produce
the Phase 4 actionable advice.

Usage:
    # Evaluate a custom PDF resume against a specific JD file:
    python run_phase4.py --resume my_resume.pdf --jd my_jd_file.pdf

    # Evaluate against a Kaggle KB role using its JobID:
    python run_phase4.py --resume my_resume.pdf --role-id NET-E-013
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
from src.phase3_evaluator import evaluate_resume
from src.phase4_advisor import generate_career_advice


def _pp(obj: object, indent: int = 2) -> None:
    print(json.dumps(obj, indent=indent, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScopeWise Phase 4 — Final LLM Career Advice.",
    )
    parser.add_argument("--resume", required=True, help="Path to your resume PDF/DOCX.")
    parser.add_argument("--jd", default=None, help="Path to your JD PDF/DOCX.")
    parser.add_argument("--role-id", default=None, help="JobID from the Kaggle KB (e.g. NET-E-013).")
    args = parser.parse_args()

    # 1. Parse Resume
    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"❌ Resume file not found: {resume_path}")
        sys.exit(1)
    
    print(f"\n📄 Parsing Resume: {resume_path}")
    resume_text = parse_document(str(resume_path))

    # 2. Parse JD or Load from KB
    jd_metadata = {
        "Title": "Custom Role",
        "ExperienceLevel": "Experienced",
        "YearsOfExperience": "3+",
    }
    
    kb = KnowledgeBaseIndex().load_all()

    if args.jd:
        jd_path = Path(args.jd)
        if not jd_path.exists():
            print(f"❌ JD file not found: {jd_path}")
            sys.exit(1)
        print(f"📄 Parsing JD: {jd_path}")
        jd_text = parse_document(str(jd_path))
    elif args.role_id:
        print(f"🔍 Looking up role: {args.role_id}")
        role = kb.get_by_id(args.role_id)
        if role is None:
            print(f"❌ Role ID '{args.role_id}' not found.")
            sys.exit(1)
        jd_metadata = {
            "Title": role.get("Title", ""),
            "ExperienceLevel": role.get("ExperienceLevel", ""),
            "YearsOfExperience": role.get("YearsOfExperience", "0"),
        }
        jd_text = get_jd_text(role_id=args.role_id, kb_index=kb)
    else:
        print("❌ You must provide either --jd or --role-id.\n")
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("▶ Executing Phase 2 Pipeline (NER Extractor + O*NET)")
    print("=" * 60)
    phase2_result = analyze_resume(
        resume_text=resume_text,
        jd_text=jd_text,
        domain="Engineering",
        level=jd_metadata["ExperienceLevel"],
        skip_gpt=False, # We assume Ollama is running since we are running Phase 4
    )

    print("\n" + "=" * 60)
    print("▶ Executing Phase 3 Pipeline (Fuzzy Logic & Structure)")
    print("=" * 60)
    phase3_output = evaluate_resume(
        resume_entities=phase2_result["resume_entities"],
        jd_entities=phase2_result["jd_entities"],
        jd_metadata=jd_metadata,
        resume_text=resume_text,
    )
    
    phase3_output["jd_metadata"] = jd_metadata

    print("\n" + "=" * 60)
    print("▶ Executing Phase 4 Pipeline (LLM Explanation & Advice)")
    print("=" * 60)
    phase4_output = generate_career_advice(
        phase2_output=phase2_result,
        phase3_output=phase3_output,
    )

    print("\n" + "═" * 60)
    print("  PHASE 4 RESULTS SUMMARY")
    print("═" * 60)
    
    print("\n  1. Structural Advice:")
    for i, advice in enumerate(phase4_output.get("structural_advice", []), 1):
        print(f"     {i}. {advice}")
        
    print("\n  2. Top Skill Explanations:")
    for exp in phase4_output.get("skill_explanations", []):
        print(f"     • {exp.get('skill', 'Unknown')}: {exp.get('criticality', '')}")
        print(f"       Action: {exp.get('next_steps', '')}")
        
    print("\n  3. Recommended Resources (Top Skills):")
    resources = phase4_output.get("recommended_resources", {})
    if not resources:
        print("     (No missing skills to recommend resources for)")
    for skill, resource_info in resources.items():
        print(f"     ▶ {skill}")
        for res in resource_info.get("resources", []):
            print(f"       - [{res.get('platform')}] {res.get('title')}:  {res.get('url')}")

    print("\n✅ Run complete. Insights written to outputs/phase4/")

if __name__ == "__main__":
    main()
