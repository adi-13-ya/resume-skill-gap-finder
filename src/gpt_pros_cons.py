"""
gpt_pros_cons.py — Phase 2, Component 3: LLM Pros & Cons Extraction
=====================================================================
Uses the OpenAI-compatible Python SDK to call Ollama (Llama 3) and
extract strengths (pros) and weaknesses (cons) from a resume relative
to a job description.

Configuration is loaded from a ``.env`` file via ``python-dotenv``.

Public API:
    extract_pros_cons(resume_text, jd_text, domain, level)
        → dict with "pros", "cons" lists
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ──────────────────────────────────────────────────────
# Load .env and initialise Ollama client
# ──────────────────────────────────────────────────────

# load_dotenv() searches for .env in the current directory and parents
load_dotenv()

_client: OpenAI | None = None


def _get_ollama_client() -> OpenAI:
    """Lazy-initialise the OpenAI-compatible client pointing to Ollama."""
    global _client
    if _client is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        _client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Placeholder — Ollama doesn't require a real key
        )
        print("[LLM] Ollama client initialised.")
    return _client


# ──────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert HR analyst and resume screener with deep expertise \
across a wide range of professional domains including but not limited to: \
Engineering & Technology, Commerce & Finance, Journalism & Media, \
Healthcare & Life Sciences, Legal & Compliance, Education & Academia, \
Creative Arts & Design, Government & Public Administration, Hospitality \
& Tourism, Manufacturing & Supply Chain, Real Estate, Retail, \
Nonprofit & Social Impact, and any other industry the candidate or job \
may belong to. You dynamically adapt your evaluation criteria to the \
specific domain and seniority level provided. You evaluate candidate \
resumes against job descriptions with rigorous, objective analysis \
grounded in industry-specific standards and expectations. You always \
respond with valid JSON only — no markdown, no explanations outside \
the JSON. Your entire response must start with { and end with }. \
Do not include any text before or after the JSON object."""

_USER_PROMPT_TEMPLATE = """\
Compare the following RESUME against the JOB DESCRIPTION.

Domain: {domain}
Experience Level Expected: {level}

=== JOB DESCRIPTION ===
{jd_text}

=== RESUME ===
{resume_text}

Analyse the resume and return a JSON object with exactly this structure:

{{
  "pros": [
    {{"point": "<strength>", "justification": "<one-line reason>"}},
    ...
  ],
  "cons": [
    {{"point": "<weakness>", "justification": "<one-line reason>"}},
    ...
  ]
}}

For PROS, identify:
- Directly relevant technical skills that match the JD
- Quantified achievements and measurable impact
- Relevant certifications or qualifications
- Leadership and initiative indicators
- Domain-relevant tools and technologies
- Strong educational background or prestigious institutions

For CONS, identify:
- Skill gaps relative to the JD requirements
- Missing qualifications or certifications the JD requires
- Weak or vague language (e.g. "familiar with" instead of "expert in")
- Short tenures at multiple roles (job hopping signals)
- Missing contact information or incomplete sections
- Lack of measurable outcomes or achievements
- Domain mismatch (e.g. applying for {domain} with unrelated background)
- Experience level mismatch (expected {level})

Provide 4-8 pros and 4-8 cons. Each must have a brief, one-line justification.
Return ONLY the JSON object, nothing else. Do not wrap it in markdown \
code fences. Start your response with {{ and end with }}."""


# ──────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────

def _parse_gpt_response(raw: str) -> dict[str, Any]:
    """Parse the LLM's response into a structured dict.

    Handles cases where GPT wraps JSON in markdown code fences or adds
    preamble text before the actual JSON.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first JSON object in the string
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: return a structured error
    return {
        "pros": [{"point": "Parse error", "justification": "Could not parse GPT response."}],
        "cons": [{"point": "Parse error", "justification": raw[:200]}],
        "_raw_response": raw,
    }


# ──────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────

def extract_pros_cons(
    resume_text: str,
    jd_text: str,
    domain: str = "Engineering",
    level: str = "Fresher",
    model: str | None = None,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Send resume + JD to Ollama (Llama 3) and extract structured pros & cons.

    Args:
        resume_text: Full plain-text of the candidate's resume.
        jd_text:     Full plain-text of the job description.
        domain:      Job domain (``"Engineering"`` / ``"Commerce"`` /
                     ``"Journalism"``).
        level:       Expected experience level (``"Fresher"`` /
                     ``"Senior"`` / ``"Experienced"`` etc.).
        model:       Ollama model name. Defaults to the ``OLLAMA_MODEL``
                     env var, or ``"llama3"`` if unset.
        temperature: Sampling temperature. Lower = more deterministic.

    Returns:
        A dict with ``"pros"`` and ``"cons"`` lists. Each item is a dict
        with ``"point"`` and ``"justification"`` keys.

    Raises:
        RuntimeError: If the Ollama API call fails.
    """
    client = _get_ollama_client()
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", "llama3")

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        domain=domain,
        level=level,
        jd_text=jd_text.strip(),
        resume_text=resume_text.strip(),
    )

    print(f"[LLM] Sending resume + JD to {model} for pros/cons analysis …")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama API call failed: {exc}\n"
            f"Check that Ollama is running at {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}."
        ) from exc

    raw_content = response.choices[0].message.content or ""
    print(f"[LLM] Received response ({len(raw_content)} chars).")

    result = _parse_gpt_response(raw_content)

    # Ensure expected keys exist
    result.setdefault("pros", [])
    result.setdefault("cons", [])

    return result
