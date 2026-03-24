"""
phase4_advisor.py — Phase 4: Final Recommendation & Advice Orchestrator
=======================================================================
Orchestrates Phase 4 by taking the outputs from Phase 2 and Phase 3 to
provide human-centric career advice, skill gap explanations, and resources.

Public API:
    generate_career_advice(phase2_output, phase3_output)
        → unified Phase 4 recommendation dictionary
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.explanation_layer import explain_skill_gap
from src.resource_engine import get_resources_for_skill
from src.structural_advice import generate_structural_advice

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_career_advice(
    phase2_output: dict[str, Any],
    phase3_output: dict[str, Any]
) -> dict[str, Any]:
    """Run the Phase 4 advisor pipeline.
    
    Args:
        phase2_output: The complete output dictionary from Phase 2 analysis.
        phase3_output: The complete output dictionary from Phase 3 evaluation.
        
    Returns:
        A dictionary containing the Phase 4 recommendations.
    """
    print(f"\n{'='*60}")
    print(f"  ScopeWise — Phase 4: LLM Explanation & Recommendation")
    print(f"{'='*60}\n")
    
    # ── 1. Gather Inputs ─────────────────────────────────
    role_title = phase3_output.get("jd_metadata", {}).get("Title", "Target Role")
    
    # Extract missing skills from Phase 2 output
    skill_gap = phase2_output.get("skill_gap", {})
    missing_skills = skill_gap.get("jd_skills_missing_in_resume", [])
    
    # Extract current candidate skills from Phase 2 output
    resume_entities = phase2_output.get("resume_entities", {})
    current_skills = resume_entities.get("skills", []) + resume_entities.get("tools", [])
    
    # Extract structural analysis from Phase 3 output
    structural_analysis = phase3_output.get("structural_analysis", {})
    
    # ── 2. Structural Advice ─────────────────────────────
    print("▶ Step 1 / 3 — Generating Structural Advice via LLM")
    advice_list = generate_structural_advice(structural_analysis=structural_analysis)
    
    # ── 3. Skill Gap Explanations ────────────────────────
    print("\n▶ Step 2 / 3 — Explaining Skill Gaps via LLM")
    explanations = explain_skill_gap(
        missing_skills=missing_skills,
        current_skills=current_skills,
        role_title=role_title
    )
    
    # ── 4. Resource Engine ───────────────────────────────
    print("\n▶ Step 3 / 3 — Generating Resource Links for Missing Skills")
    resources = {}
    for missing_skill in missing_skills[:5]: # Top 5 to match explanations
        resources[missing_skill] = get_resources_for_skill(skill_name=missing_skill)
        
    # ── 5. Assemble and Persist ──────────────────────────
    output = {
        "role_title": role_title,
        "structural_advice": advice_list,
        "skill_explanations": explanations,
        "recommended_resources": resources
    }
    
    output_dir = _PROJECT_ROOT / "outputs" / "phase4"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"phase4_result_{timestamp}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
        
    print(f"\n💾 Phase 4 recommendations saved to: {output_path}")
    print(f"\n{'='*60}")
    print(f"  ✅ Phase 4 pipeline complete.")
    print(f"{'='*60}\n")
    
    return output
