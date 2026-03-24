"""
phase3_evaluator.py — Phase 3: Fuzzy Logic & Structural Evaluation
===================================================================
Receives the Phase 2 output dictionary **in memory** and computes a
multi-dimensional suitability assessment using four components:

  1. **Semantic Similarity Score** — Sentence Transformers (all-MiniLM-L6-v2)
  2. **Experience & Education Alignment** — rule-based with rapidfuzz
  3. **Structural Analysis** — Ollama / Llama 3 LLM evaluation
  4. **Fuzzy Inference System** — scikit-fuzzy triangular MFs + rule base

Public API:
    evaluate_resume(resume_entities, jd_entities, jd_metadata, resume_text)
        → unified Phase 3 result dictionary

The output is both returned in memory (for Phase 4) and persisted to
``outputs/phase3/`` as a timestamped JSON file.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# scikit-fuzzy imports
import skfuzzy as fuzz_sk
from skfuzzy import control as ctrl

# ──────────────────────────────────────────────────────
# Environment and model singletons
# ──────────────────────────────────────────────────────

# Load .env so OLLAMA_BASE_URL and OLLAMA_MODEL are available
load_dotenv()

# SentenceTransformer model — loaded once at module import time for
# efficiency; subsequent calls reuse the same in-memory model.
_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    """Lazy-load the Sentence Transformer model (all-MiniLM-L6-v2).

    This model produces 384-dimensional embeddings and is lightweight
    enough to run on CPU without issues.
    """
    global _st_model
    if _st_model is None:
        print("[Phase3] Loading Sentence Transformer model (all-MiniLM-L6-v2) …")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Phase3] Sentence Transformer model loaded.")
    return _st_model


# ──────────────────────────────────────────────────────
# COMPONENT 1 — Semantic Similarity Score
# ──────────────────────────────────────────────────────

def compute_similarity_score(
    resume_entities: dict[str, list[str]],
    jd_entities: dict[str, list[str]],
) -> float:
    """Compute semantic similarity between resume and JD skill+tool lists.

    Combines the ``skills`` and ``tools`` lists from both the resume and
    JD entity dictionaries into single strings, encodes them into dense
    embeddings using ``all-MiniLM-L6-v2``, and returns the cosine
    similarity as a float in [0.0, 1.0].

    Args:
        resume_entities: Phase 2 resume entities dict with keys
                         ``skills``, ``tools``, etc.
        jd_entities:     Phase 2 JD entities dict with the same shape.

    Returns:
        Cosine similarity score between 0.0 and 1.0.
    """
    # Merge skills + tools into a single descriptive string per side
    resume_skills = resume_entities.get("skills", []) + resume_entities.get("tools", [])
    jd_skills = jd_entities.get("skills", []) + jd_entities.get("tools", [])

    # Guard: if either side is empty, similarity is 0
    if not resume_skills or not jd_skills:
        print("[Phase3] WARNING — Empty skill/tool list; similarity = 0.0")
        return 0.0

    # Join into single strings for embedding
    resume_text = ", ".join(resume_skills)
    jd_text = ", ".join(jd_skills)

    # Encode both strings into 384-d embeddings
    model = _get_st_model()
    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)

    # Cosine similarity via sentence_transformers.util
    cosine_score = float(util.cos_sim(embeddings[0], embeddings[1]).item())

    # Clamp to [0, 1] just in case of floating-point edge cases
    cosine_score = max(0.0, min(1.0, cosine_score))

    print(f"[Phase3] Semantic similarity score: {cosine_score:.4f}")
    return cosine_score


# ──────────────────────────────────────────────────────
# COMPONENT 2 — Experience & Education Alignment Score
# ──────────────────────────────────────────────────────

def _parse_years_from_strings(experience_strings: list[str]) -> float:
    """Extract the maximum numeric years value from a list of experience strings.

    Handles formats like:
        "3 years", "5+ years", "0-1", "3+", "5–7 years", "10+ years experience"
    as well as level keywords like "Fresher" (0), "Senior" (7), etc.

    Returns the best-estimate years as a float.
    """
    max_years = 0.0

    for s in experience_strings:
        text = s.lower().strip()

        # Try to find patterns like "5+", "3-5", "10+ years"
        # Match ranges like "3-5" or "0-1" — take the higher end
        range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
        if range_match:
            high = float(range_match.group(2))
            max_years = max(max_years, high)
            continue

        # Match "5+" or "5+ years"
        plus_match = re.search(r"(\d+)\s*\+", text)
        if plus_match:
            max_years = max(max_years, float(plus_match.group(1)))
            continue

        # Match standalone numbers like "3 years"
        num_match = re.search(r"(\d+)\s*(?:years?|yrs?|yr)?", text)
        if num_match:
            max_years = max(max_years, float(num_match.group(1)))
            continue

        # Handle seniority-level keywords as rough experience proxies
        level_map = {
            "fresher": 0, "intern": 0, "trainee": 0,
            "entry": 1, "junior": 1,
            "mid": 3, "associate": 2,
            "senior": 7, "lead": 8, "principal": 10,
            "manager": 8, "director": 12, "vp": 15, "cto": 15,
        }
        for keyword, yrs in level_map.items():
            if keyword in text:
                max_years = max(max_years, float(yrs))
                break

    return max_years


def _parse_required_years(yoe_string: str) -> float:
    """Parse the JD's YearsOfExperience field into a numeric value.

    Handles formats like "0-1", "3+", "5+", "0–1 year", "4+ years".
    Returns the minimum required years (lower bound of range or the
    number before '+').
    """
    text = yoe_string.lower().strip()

    # Range: "3-5" → take lower bound as the requirement
    range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
    if range_match:
        return float(range_match.group(1))

    # Plus: "5+" → the minimum is 5
    plus_match = re.search(r"(\d+)\s*\+", text)
    if plus_match:
        return float(plus_match.group(1))

    # Standalone number
    num_match = re.search(r"(\d+)", text)
    if num_match:
        return float(num_match.group(1))

    return 0.0


def compute_experience_education_score(
    resume_entities: dict[str, list[str]],
    jd_metadata: dict[str, Any],
) -> float:
    """Compute a weighted alignment score for experience + education.

    Weights:
        - Experience years alignment: 60%
        - Education/degree alignment:  40%

    Experience scoring:
        ratio = resume_years / required_years  (capped at 1.0)
    Education scoring:
        Best fuzzy-match ratio between any resume degree and the JD's
        required degree string, using rapidfuzz ``token_sort_ratio``.

    Args:
        resume_entities: Phase 2 resume entities dict.
        jd_metadata:     JD metadata dict from the knowledge base, with
                         keys like ``YearsOfExperience``, ``Title``,
                         ``ExperienceLevel``, etc.

    Returns:
        Weighted combined score between 0.0 and 1.0.
    """
    # ── Experience alignment (60% weight) ────────────────
    resume_years = _parse_years_from_strings(
        resume_entities.get("experience", [])
    )
    required_years_str = jd_metadata.get("YearsOfExperience", "0")
    required_years = _parse_required_years(str(required_years_str))

    if required_years <= 0:
        # If JD doesn't specify a requirement, treat as fully met
        exp_score = 1.0
    else:
        # Ratio capped at 1.0 — exceeding the requirement is not penalized
        exp_score = min(resume_years / required_years, 1.0)

    print(f"[Phase3] Experience: resume={resume_years}y, required={required_years}y → score={exp_score:.3f}")

    # ── Education alignment (40% weight) ─────────────────
    resume_degrees = resume_entities.get("degrees", [])
    # The Kaggle dataset doesn't have a dedicated 'required_degree' field,
    # so we fall back to checking the Title or ExperienceLevel as a proxy.
    # If a 'required_degree' key is present, use it; otherwise use Title.
    required_degree = jd_metadata.get("required_degree", "")
    if not required_degree:
        # Use the role title + experience level as a rough proxy —
        # in the real world, the JD text itself would be parsed.
        required_degree = jd_metadata.get("Title", "")

    if not resume_degrees or not required_degree:
        # If either side has no degree info, default to 50% match
        edu_score = 0.5
    else:
        # Find the best fuzzy match between any resume degree and
        # the required degree using token_sort_ratio for robustness
        best = 0.0
        for deg in resume_degrees:
            ratio = fuzz.token_sort_ratio(
                deg.lower(), required_degree.lower()
            ) / 100.0  # rapidfuzz returns 0–100
            best = max(best, ratio)
        edu_score = best

    print(f"[Phase3] Education: best_match={edu_score:.3f}")

    # ── Weighted combination ─────────────────────────────
    combined = 0.6 * exp_score + 0.4 * edu_score
    combined = max(0.0, min(1.0, combined))

    print(f"[Phase3] Experience+Education alignment score: {combined:.4f}")
    return combined


# ──────────────────────────────────────────────────────
# COMPONENT 3 — Structural Analysis via Ollama / Llama 3
# ──────────────────────────────────────────────────────

# Default zero-scored structural result used as fallback on errors
_DEFAULT_STRUCTURAL_RESULT: dict[str, Any] = {
    "section_clarity": 0,
    "quantification_usage": 0,
    "action_language_strength": 0,
    "completeness": 0,
    "overall_structural_score": 0,
    "structural_feedback": "Structural analysis could not be completed.",
}

# Ollama client singleton for Phase 3
_ollama_client: OpenAI | None = None


def _get_ollama_client() -> OpenAI:
    """Lazy-initialise the OpenAI-compatible client pointing to Ollama.

    Reads OLLAMA_BASE_URL from the environment (defaults to
    ``http://localhost:11434/v1``).  The api_key is a placeholder
    because Ollama does not require authentication.
    """
    global _ollama_client
    if _ollama_client is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        _ollama_client = OpenAI(
            base_url=base_url,
            api_key="ollama",
        )
    return _ollama_client


# The prompt for Llama 3 structural analysis — asks for raw JSON only,
# no markdown fences, no extra text.
_STRUCTURAL_PROMPT = """\
You are an expert resume evaluator. Analyse the following resume text \
and evaluate its structural quality. Score each dimension from 0 to 10.

Dimensions to evaluate:

1. **section_clarity** (0–10): Are standard resume sections (Education, \
Skills, Work Experience, Projects, Contact information) clearly \
identifiable and logically organized? Higher scores for resumes with \
clear headings and logical section order.

2. **quantification_usage** (0–10): Does the candidate use numbers, \
metrics, and measurable outcomes to describe achievements? For example \
"increased revenue by 30%" or "handled 5M+ requests/day" scores \
higher than "responsible for revenue improvement".

3. **action_language_strength** (0–10): Does the resume use strong, \
active verbs (built, led, designed, delivered, architected, optimized) \
rather than weak, passive phrases (was responsible for, assisted with, \
helped in)?

4. **completeness** (0–10): Are all essential resume sections present \
and sufficiently detailed? Essential sections include Contact Info, \
Summary/Objective, Skills, Work Experience, and Education. Extra \
sections like Certifications, Projects, or Publications add value.

5. **overall_structural_score** (0–10): A holistic summary score \
reflecting the overall structural quality of the resume.

6. **structural_feedback**: A single concise sentence summarizing the \
resume's structural quality.

Return ONLY a valid JSON object with exactly these six keys. \
Do NOT wrap the JSON in markdown code fences. Do NOT add any \
explanation, preamble, or commentary. Your entire response must \
start with {{ and end with }}. Return raw JSON only.

=== RESUME TEXT ===
{resume_text}"""


def analyze_structure_with_llm(resume_text: str) -> dict[str, Any]:
    """Send the raw resume text to Llama 3 for structural evaluation.

    Constructs a detailed prompt and calls Ollama (Llama 3) via the
    OpenAI-compatible SDK.  Parses the JSON response, handling
    malformed output gracefully by returning a zero-scored default
    dictionary instead of crashing.

    Args:
        resume_text: The full raw text of the resume.

    Returns:
        A dict with keys ``section_clarity``, ``quantification_usage``,
        ``action_language_strength``, ``completeness``,
        ``overall_structural_score``, and ``structural_feedback``.
    """
    client = _get_ollama_client()
    model = os.environ.get("OLLAMA_MODEL", "llama3")

    prompt = _STRUCTURAL_PROMPT.format(resume_text=resume_text.strip())

    print(f"[Phase3] Sending resume to {model} for structural analysis …")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # Log the error but do NOT crash — return default zeros
        print(f"[Phase3] ERROR — Ollama structural analysis failed: {exc}")
        return dict(_DEFAULT_STRUCTURAL_RESULT)

    raw_content = response.choices[0].message.content or ""
    print(f"[Phase3] Structural analysis response received ({len(raw_content)} chars).")

    # ── Parse JSON response ──────────────────────────────
    return _parse_structural_response(raw_content)


def _parse_structural_response(raw: str) -> dict[str, Any]:
    """Parse the LLM's structural analysis JSON response.

    Handles edge cases:
      - Markdown code fences (```json ... ```)
      - Preamble text before the JSON object
      - Completely malformed JSON (returns default zeros)
    """
    text = raw.strip()

    # Strip markdown code fences if the LLM added them
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Attempt 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            print("[Phase3] Structural JSON parsed successfully.")
            return result
    except json.JSONDecodeError:
        pass

    # Attempt 2: find the first { ... } object in the string
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            result = json.loads(text[brace_start : brace_end + 1])
            if isinstance(result, dict):
                print("[Phase3] Structural JSON parsed (fallback extraction).")
                return result
        except json.JSONDecodeError:
            pass

    # All parsing failed — log error and return safe defaults
    print("[Phase3] WARNING — Could not parse structural analysis response.")
    print(f"[Phase3] Raw response (first 300 chars): {raw[:300]}")
    return dict(_DEFAULT_STRUCTURAL_RESULT)


# ──────────────────────────────────────────────────────
# COMPONENT 4 — Fuzzy Inference System (scikit-fuzzy)
# ──────────────────────────────────────────────────────

def _build_fuzzy_system() -> ctrl.ControlSystem:
    """Build the scikit-fuzzy ControlSystem with membership functions and rules.

    Inputs (Antecedents):
        - similarity:  universe [0, 1] — semantic skill similarity score
        - experience:  universe [0, 1] — experience+education alignment

    Output (Consequent):
        - suitability: universe [0, 1] — overall suitability verdict

    Membership functions:
        - Inputs:  low / medium / high  (triangular)
        - Output:  poor / weak / close / good / excellent  (triangular)

    Returns:
        A configured ``ctrl.ControlSystem`` ready for simulation.
    """
    # Define universes of discourse with fine granularity (0.01 steps)
    similarity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "similarity")
    experience = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "experience")
    suitability = ctrl.Consequent(np.arange(0, 1.01, 0.01), "suitability")

    # ── Input membership functions (triangular) ──────────
    # "low" peaks at 0, falls to 0 by 0.4
    similarity["low"]    = fuzz_sk.trimf(similarity.universe, [0.0, 0.0, 0.4])
    # "medium" spans 0.2–0.8, peaks at 0.5
    similarity["medium"] = fuzz_sk.trimf(similarity.universe, [0.2, 0.5, 0.8])
    # "high" rises from 0.6, peaks at 1.0
    similarity["high"]   = fuzz_sk.trimf(similarity.universe, [0.6, 1.0, 1.0])

    experience["low"]    = fuzz_sk.trimf(experience.universe, [0.0, 0.0, 0.4])
    experience["medium"] = fuzz_sk.trimf(experience.universe, [0.2, 0.5, 0.8])
    experience["high"]   = fuzz_sk.trimf(experience.universe, [0.6, 1.0, 1.0])

    # ── Output membership functions (5 levels, triangular) ──
    suitability["poor"]      = fuzz_sk.trimf(suitability.universe, [0.0, 0.0, 0.2])
    suitability["weak"]      = fuzz_sk.trimf(suitability.universe, [0.1, 0.25, 0.4])
    suitability["close"]     = fuzz_sk.trimf(suitability.universe, [0.3, 0.45, 0.6])
    suitability["good"]      = fuzz_sk.trimf(suitability.universe, [0.5, 0.65, 0.8])
    suitability["excellent"] = fuzz_sk.trimf(suitability.universe, [0.7, 1.0, 1.0])

    # ── Fuzzy rules ──────────────────────────────────────
    # Rule 1: Both high → Excellent
    r1 = ctrl.Rule(similarity["high"] & experience["high"], suitability["excellent"])
    # Rule 2: High similarity + medium experience → Good
    r2 = ctrl.Rule(similarity["high"] & experience["medium"], suitability["good"])
    # Rule 3: Medium similarity + high experience → Good
    r3 = ctrl.Rule(similarity["medium"] & experience["high"], suitability["good"])
    # Rule 4: Both medium → Close
    r4 = ctrl.Rule(similarity["medium"] & experience["medium"], suitability["close"])
    # Rule 5: High similarity + low experience → Close
    r5 = ctrl.Rule(similarity["high"] & experience["low"], suitability["close"])
    # Rule 6: Low similarity + high experience → Weak
    r6 = ctrl.Rule(similarity["low"] & experience["high"], suitability["weak"])
    # Rule 7: Medium similarity + low experience → Weak
    r7 = ctrl.Rule(similarity["medium"] & experience["low"], suitability["weak"])
    # Rule 8: Low similarity + medium experience → Weak
    r8 = ctrl.Rule(similarity["low"] & experience["medium"], suitability["weak"])
    # Rule 9: Both low → Poor
    r9 = ctrl.Rule(similarity["low"] & experience["low"], suitability["poor"])

    # Assemble the control system from all rules
    system = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8, r9])
    return system


# Module-level cached control system (built once per process)
_fuzzy_system: ctrl.ControlSystem | None = None


def _get_fuzzy_system() -> ctrl.ControlSystem:
    """Lazy-build and cache the fuzzy control system."""
    global _fuzzy_system
    if _fuzzy_system is None:
        _fuzzy_system = _build_fuzzy_system()
    return _fuzzy_system


def _score_to_verdict(score: float) -> str:
    """Map a crisp fuzzy output score to a human-readable verdict string.

    Thresholds:
        ≥ 0.75  →  Excellent Fit
        ≥ 0.55  →  Good Match
        ≥ 0.35  →  Close Match
        ≥ 0.15  →  Weak Match
        < 0.15  →  Poor Fit
    """
    if score >= 0.75:
        return "Excellent Fit"
    elif score >= 0.55:
        return "Good Match"
    elif score >= 0.35:
        return "Close Match"
    elif score >= 0.15:
        return "Weak Match"
    else:
        return "Poor Fit"


def run_fuzzy_evaluation(
    similarity_score: float,
    experience_score: float,
) -> tuple[float, str]:
    """Run the fuzzy inference system and return a crisp score + verdict.

    Feeds the two input scores into the scikit-fuzzy simulation,
    computes the defuzzified output, and maps it to a verdict string.

    Args:
        similarity_score:  Semantic similarity in [0, 1] from Component 1.
        experience_score:  Exp+Edu alignment in [0, 1] from Component 2.

    Returns:
        A tuple of (crisp_score rounded to 3 decimals, verdict_string).
    """
    system = _get_fuzzy_system()
    simulation = ctrl.ControlSystemSimulation(system)

    # Clamp inputs to valid universe range [0, 1]
    similarity_score = max(0.0, min(1.0, similarity_score))
    experience_score = max(0.0, min(1.0, experience_score))

    # Feed inputs into the simulation
    simulation.input["similarity"] = similarity_score
    simulation.input["experience"] = experience_score

    # Compute the defuzzified crisp output
    simulation.compute()
    crisp_score = float(simulation.output["suitability"])

    # Round to 3 decimal places for clean output
    crisp_score = round(crisp_score, 3)

    # Map to a linguistic verdict
    verdict = _score_to_verdict(crisp_score)

    print(f"[Phase3] Fuzzy evaluation: crisp_score={crisp_score}, verdict=\"{verdict}\"")
    return crisp_score, verdict


# ──────────────────────────────────────────────────────
# MASTER ORCHESTRATOR — evaluate_resume()
# ──────────────────────────────────────────────────────

# Project root for output file persistence
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate_resume(
    resume_entities: dict[str, list[str]],
    jd_entities: dict[str, list[str]],
    jd_metadata: dict[str, Any],
    resume_text: str,
) -> dict[str, Any]:
    """Run the full Phase 3 evaluation pipeline.

    Calls all four components in sequence:
      1. Semantic similarity (Sentence Transformers)
      2. Experience + education alignment (rule-based + rapidfuzz)
      3. Structural analysis (Ollama / Llama 3)
      4. Fuzzy inference system (scikit-fuzzy)

    The result is:
      - Returned as an in-memory dict for direct use by Phase 4.
      - Saved as a timestamped JSON file in ``outputs/phase3/``.

    Args:
        resume_entities: Phase 2 resume entities dict (skills, tools,
                         certifications, degrees, experience).
        jd_entities:     Phase 2 JD entities dict (same shape).
        jd_metadata:     JD metadata from the knowledge base (Title,
                         ExperienceLevel, YearsOfExperience, etc.).
        resume_text:     The original raw resume text string.

    Returns:
        A unified Phase 3 result dictionary::

            {
                "similarity_score": float,
                "experience_score": float,
                "structural_score": float,      # normalized 0–1
                "structural_analysis": dict,     # full LLM JSON output
                "fuzzy_crisp_score": float,
                "verdict": str,
            }
    """
    print(f"\n{'='*60}")
    print(f"  ScopeWise — Phase 3: Fuzzy Logic & Structural Evaluation")
    print(f"{'='*60}\n")

    # ── Component 1: Semantic Similarity ─────────────────
    print("▶ Component 1 / 4 — Semantic Similarity Score")
    similarity_score = compute_similarity_score(resume_entities, jd_entities)

    # ── Component 2: Experience & Education Alignment ────
    print("\n▶ Component 2 / 4 — Experience & Education Alignment")
    experience_score = compute_experience_education_score(
        resume_entities, jd_metadata
    )

    # ── Component 3: Structural Analysis via Llama 3 ─────
    print("\n▶ Component 3 / 4 — Structural Analysis (Ollama / Llama 3)")
    structural_analysis = analyze_structure_with_llm(resume_text)

    # Normalize the overall_structural_score from 0–10 to 0.0–1.0
    raw_structural = structural_analysis.get("overall_structural_score", 0)
    try:
        structural_score = float(raw_structural) / 10.0
    except (TypeError, ValueError):
        structural_score = 0.0
    structural_score = max(0.0, min(1.0, structural_score))
    print(f"[Phase3] Normalized structural score: {structural_score:.3f}")

    # ── Component 4: Fuzzy Inference System ──────────────
    print("\n▶ Component 4 / 4 — Fuzzy Inference System")
    fuzzy_crisp_score, verdict = run_fuzzy_evaluation(
        similarity_score, experience_score
    )

    # ── Assemble unified output ──────────────────────────
    output: dict[str, Any] = {
        "similarity_score": round(similarity_score, 4),
        "experience_score": round(experience_score, 4),
        "structural_score": round(structural_score, 4),
        "structural_analysis": structural_analysis,
        "fuzzy_crisp_score": fuzzy_crisp_score,
        "verdict": verdict,
    }

    # ── Persist to disk ──────────────────────────────────
    # Create output directory if it doesn't exist
    output_dir = _PROJECT_ROOT / "outputs" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped filename for traceability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"phase3_result_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n💾 Phase 3 results saved to: {output_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ Phase 3 evaluation complete.")
    print(f"  📊 Verdict: {verdict} (score: {fuzzy_crisp_score})")
    print(f"{'='*60}\n")

    return output
