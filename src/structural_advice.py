"""
structural_advice.py — Phase 4: Structural Advice Layer
=======================================================
Takes the structural analysis metrics produced by Phase 3 and uses an LLM
to generate specific, actionable advice (e.g., suggesting the STAR method
for low quantification usage).

Public API:
    generate_structural_advice(structural_analysis)
        → list of actionable advice strings
"""

import os
import json
import re
from typing import Any
from openai import OpenAI

# Default fallback advice
_DEFAULT_ADVICE = [
    "Ensure your sections (Education, Experience, Skills) are clearly separated with standard headings.",
    "Try to use more numbers and metrics to quantify your achievements (e.g., 'reduced latency by 20%').",
    "Use strong action verbs (e.g., 'Architected', 'Spearheaded') to start bullet points."
]

# Ollama client singleton
_ollama_client: OpenAI | None = None

def _get_ollama_client() -> OpenAI:
    """Lazy-initialise the OpenAI-compatible client pointing to Ollama."""
    global _ollama_client
    if _ollama_client is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        _ollama_client = OpenAI(
            base_url=base_url,
            api_key="ollama", # Placeholder
        )
    return _ollama_client

_ADVICE_PROMPT = """\
You are an expert resume reviewer. You have just evaluated a candidate's resume structure and given it the following scores (out of 10):

- Section Clarity: {section_clarity}
- Quantification Usage: {quantification_usage}
- Action Language Strength: {action_language_strength}
- Completeness: {completeness}
- Overall Structural Score: {overall_structural_score}

A brief summary from the evaluation: "{structural_feedback}"

Based on the scores, provide 2 to 3 very specific and actionable bullet points on how the candidate can improve their resume structure.
For example, if Quantification Usage is low, suggest using the STAR method (Situation, Task, Action, Result) with specific metric examples.
Do not flatter the candidate. Be direct and constructive.

Return ONLY a valid JSON object with a single key "advice" which contains a list of strings (each string is one piece of advice).
Do NOT wrap the JSON in markdown code fences. Do NOT add any preamble. Return raw JSON only."""

def generate_structural_advice(structural_analysis: dict[str, Any]) -> list[str]:
    """Generate structural advice based on Phase 3 analysis.
    
    Args:
        structural_analysis: The raw JSON object from Phase 3 structural evaluation.
        
    Returns:
        A list of actionable advice strings.
    """
    if not structural_analysis or not isinstance(structural_analysis, dict):
        return _DEFAULT_ADVICE

    # Extract metrics safely
    section_clarity = structural_analysis.get("section_clarity", 0)
    quantification_usage = structural_analysis.get("quantification_usage", 0)
    action_language = structural_analysis.get("action_language_strength", 0)
    completeness = structural_analysis.get("completeness", 0)
    overall_score = structural_analysis.get("overall_structural_score", 0)
    feedback = structural_analysis.get("structural_feedback", "Not provided.")

    # Only ask for advice if the score is not perfect (avoid unnecessary API calls)
    if float(overall_score) >= 9.5:
        return ["Your resume structure is excellent. Keep it up!"]
        
    client = _get_ollama_client()
    model = os.environ.get("OLLAMA_MODEL", "llama3")
    
    prompt = _ADVICE_PROMPT.format(
        section_clarity=section_clarity,
        quantification_usage=quantification_usage,
        action_language_strength=action_language,
        completeness=completeness,
        overall_structural_score=overall_score,
        structural_feedback=feedback
    )
    
    print("[Phase4] Generating actionable structural advice via LLM...")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content or ""
        parsed = _parse_advice_json(raw_content)
        if parsed and isinstance(parsed.get("advice"), list):
            return parsed["advice"]
    except Exception as exc:
        print(f"[Phase4] ERROR generating structural advice: {exc}")
        
    return _DEFAULT_ADVICE

def _parse_advice_json(raw: str) -> dict[str, Any]:
    """Parse the LLM JSON response safely."""
    text = raw.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict) and "advice" in result:
            return result
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            result = json.loads(text[brace_start : brace_end + 1])
            if isinstance(result, dict) and "advice" in result:
                return result
        except json.JSONDecodeError:
            pass

    print("[Phase4] WARNING — Could not parse structural advice response.")
    return {"advice": _DEFAULT_ADVICE}
