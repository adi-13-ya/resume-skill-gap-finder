"""
explanation_layer.py — Phase 4: Explanation Layer
===================================================
Provides the LLM-powered explanation of why a missing skill is critical
for the target role, and how the candidate's existing skills serve as a foundation.

Public API:
    explain_skill_gap(missing_skills, current_skills, role_title)
        → list of explanations for the missing skills
"""

import os
import json
import re
from typing import Any
from openai import OpenAI

# Default fallback explanation
_DEFAULT_EXPLANATION = {
    "skill": "Unknown",
    "criticality": "This skill is highly valued in the industry.",
    "foundation": "Your existing experience is a great starting point, but specific knowledge is required.",
    "next_steps": "Consider taking an introductory course."
}

# Ollama client singleton for Phase 4
_ollama_client: OpenAI | None = None

def _get_ollama_client() -> OpenAI:
    """Lazy-initialise the OpenAI-compatible client pointing to Ollama."""
    global _ollama_client
    if _ollama_client is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        _ollama_client = OpenAI(
            base_url=base_url,
            api_key="ollama", # Placeholder for Ollama
        )
    return _ollama_client

_EXPLANATION_PROMPT = """\
You are an expert career advisor. The user is applying for the role of '{role_title}'.
They are currently missing the following critical skill: '{missing_skill}'.

Here are some of their existing skills:
{current_skills}

Explain why '{missing_skill}' is critical for a '{role_title}', and explain how their current skills provide a good foundation but are not enough. provide actionable next steps.

Return ONLY a valid JSON object with exactly these four keys:
1. "skill": The name of the missing skill.
2. "criticality": A short explanation of why it is critical for this role.
3. "foundation": How their existing skills relate to it.
4. "next_steps": 1-2 sentences on what they should do next to acquire it.

Do NOT wrap the JSON in markdown code fences. Do NOT add any explanation, preamble, or commentary. Your entire response must start with {{ and end with }}. Return raw JSON only."""

def explain_skill_gap(missing_skills: list[str], current_skills: list[str], role_title: str) -> list[dict[str, str]]:
    """Generate explanations for a list of missing skills.

    Args:
        missing_skills: List of skills missing from the resume.
        current_skills: List of skills the candidate currently possesses.
        role_title: The title of the target role.

    Returns:
        A list of explanation dictionaries, one for each missing skill.
    """
    if not missing_skills:
        return []

    client = _get_ollama_client()
    model = os.environ.get("OLLAMA_MODEL", "llama3")
    
    current_skills_str = ", ".join(current_skills[:10]) if current_skills else "General Professional Experience"
    
    explanations = []
    
    # We will only explain up to the top 5 missing skills to save processing time
    for missing_skill in missing_skills[:5]:
        prompt = _EXPLANATION_PROMPT.format(
            role_title=role_title,
            missing_skill=missing_skill,
            current_skills=current_skills_str
        )
        
        print(f"[Phase4] Generating explanation for missing skill: '{missing_skill}' ...")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or ""
            parsed = _parse_llm_json(raw_content, missing_skill)
            explanations.append(parsed)
        except Exception as exc:
            print(f"[Phase4] ERROR generating explanation for {missing_skill}: {exc}")
            fallback = dict(_DEFAULT_EXPLANATION)
            fallback["skill"] = missing_skill
            explanations.append(fallback)
            
    return explanations

def _parse_llm_json(raw: str, missing_skill: str) -> dict[str, Any]:
    """Parse the LLM JSON response safely."""
    text = raw.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict) and "criticality" in result:
            return result
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            result = json.loads(text[brace_start : brace_end + 1])
            if isinstance(result, dict) and "criticality" in result:
                return result
        except json.JSONDecodeError:
            pass

    print(f"[Phase4] WARNING — Could not parse explanation response for {missing_skill}.")
    fallback = dict(_DEFAULT_EXPLANATION)
    fallback["skill"] = missing_skill
    return fallback
