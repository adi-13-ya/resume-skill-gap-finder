"""
ner_extractor.py — Phase 2, Component 1: Two-Stage Entity Extraction
=====================================================================
A two-model pipeline that extracts and classifies entities from resume
and JD text:

  STAGE 1 — BERT Span Detection
      Uses ``dslim/bert-base-NER`` (HuggingFace) purely as a span
      detector.  BERT scans the raw text and outputs candidate entity
      phrases.  Its generic CoNLL labels (PER, ORG, LOC, MISC) are
      completely ignored — we only care about the text spans it finds.

  STAGE 2 — LLM Span Classification (Ollama / Llama 3)
      All unique spans from Stage 1 are batched into a single LLM
      call via Ollama (Llama 3).  The model classifies every span
      into one of six categories:
          Skill | Tool | Certification | Degree | Experience | Other
      Spans labeled "Other" are noise and are discarded.

Public API:
    extract_entities(text: str) -> dict[str, list[str]]
        Keys: "skills", "tools", "certifications", "degrees", "experience"
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline, Pipeline


# ──────────────────────────────────────────────────────
# Load environment variables (for OLLAMA_BASE_URL, OLLAMA_MODEL)
# ──────────────────────────────────────────────────────
load_dotenv()


# ──────────────────────────────────────────────────────
# STAGE 1 — BERT NER span detector (singleton)
# ──────────────────────────────────────────────────────

_ner_pipeline: Pipeline | None = None


def _get_ner_pipeline() -> Pipeline:
    """Lazy-load the HuggingFace NER pipeline.

    Uses ``aggregation_strategy="simple"`` so that BERT's internal
    sub-word tokenizer automatically merges fragment tokens (e.g.
    "Ku" + "##ber" + "##netes") into clean whole-word spans before
    they ever reach our code.  This is the key to eliminating
    sub-word artifacts at the source.
    """
    global _ner_pipeline
    if _ner_pipeline is None:
        print("[NER] Loading dslim/bert-base-NER model …")
        _ner_pipeline = pipeline(
            task="ner",
            model="dslim/bert-base-NER",
            # "simple" merges sub-word tokens into whole-word spans
            aggregation_strategy="simple",
        )
        print("[NER] Model loaded successfully.")
    return _ner_pipeline


# ──────────────────────────────────────────────────────
# Text chunking (BERT has a 512-token limit)
# ──────────────────────────────────────────────────────

_MAX_CHUNK_CHARS = 1500  # ~400 tokens — safely within the 512 limit


def _chunk_text(text: str) -> list[str]:
    """Split long text into chunks that fit BERT's context window.

    Splits on paragraph boundaries first, then sentence boundaries,
    so that entity spans are not cut in half.
    """
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    chunks: list[str] = []
    paragraphs = re.split(r"\n{2,}", text)
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= _MAX_CHUNK_CHARS:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds the limit, split by sentence
            if len(para) > _MAX_CHUNK_CHARS:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) + 1 <= _MAX_CHUNK_CHARS:
                        sub = f"{sub} {sent}" if sub else sent
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = sent
                current = sub or ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


# ──────────────────────────────────────────────────────
# STAGE 1 — Run BERT and collect raw spans
# ──────────────────────────────────────────────────────

def _extract_bert_spans(text: str) -> list[str]:
    """Run BERT NER over (possibly chunked) text and return unique spans.

    BERT's entity-group labels (PER, ORG, LOC, MISC) are intentionally
    ignored — we treat every detected span as a candidate entity and
    leave the real classification to the LLM in Stage 2.

    Returns:
        A deduplicated list of cleaned span strings.
    """
    ner = _get_ner_pipeline()
    raw_spans: list[str] = []

    for chunk in _chunk_text(text):
        try:
            entities = ner(chunk)
            for ent in entities:
                # Extract the span text; ignore the "entity_group" label
                word = ent.get("word", "").strip()
                # Clean up residual tokenizer artifacts
                word = word.replace("##", "")
                word = re.sub(r"\s+", " ", word).strip()
                word = word.strip(".,;:!?()[]{}\"'")
                if word and len(word) >= 2:
                    raw_spans.append(word)
        except Exception as exc:
            print(f"[NER] Warning — BERT chunk failed: {exc}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for span in raw_spans:
        key = span.lower()
        if key not in seen:
            seen.add(key)
            unique.append(span)

    return unique


# ──────────────────────────────────────────────────────
# STAGE 2 — LLM classification of spans (Ollama / Llama 3)
# ──────────────────────────────────────────────────────

# Ollama client singleton (OpenAI-compatible SDK)
_ollama_client: OpenAI | None = None


def _get_ollama_client() -> OpenAI:
    """Lazy-initialise the OpenAI-compatible client pointing to Ollama."""
    global _ollama_client
    if _ollama_client is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        _ollama_client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Placeholder — Ollama doesn't require a real key
        )
    return _ollama_client


# The prompt instructs the LLM to classify each span into exactly one
# category and return raw JSON only — no markdown, no explanation.
_CLASSIFICATION_PROMPT = """\
You are an expert resume and job-description parser with deep knowledge \
across all professional domains — technology, healthcare, finance, law, \
media, education, manufacturing, government, hospitality, and more.

I will give you a list of text spans extracted from a professional \
document (a resume or a job description). Your task is to classify \
EACH span into exactly ONE of the following six categories. Read the \
detailed descriptions and examples carefully before classifying.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Skill
  A technical skill, soft skill, methodology, or area of expertise \
that a person can possess or a job can require. This includes \
domain-specific competencies, analytical methods, interpersonal \
abilities, and professional disciplines.
  Examples:
    Technology  — Machine Learning, System Design, Data Analysis, \
Cloud Architecture, DevOps, CI/CD, Agile, API Development, \
Natural Language Processing, Computer Vision, Cybersecurity
    Healthcare  — Patient Care, Clinical Research, Phlebotomy, \
Medical Coding, Pharmacology, Radiology, Triage
    Finance     — Financial Modeling, Risk Analysis, Auditing, \
Tax Planning, Portfolio Management, Compliance, Underwriting
    Legal       — Contract Drafting, Litigation, Legal Research, \
Regulatory Compliance, Intellectual Property
    Media       — Content Strategy, SEO, Copywriting, Video Editing, \
Broadcast Journalism, Investigative Reporting
    Soft Skills — Leadership, Communication, Problem Solving, \
Team Management, Critical Thinking, Public Speaking, Negotiation, \
Time Management, Stakeholder Management, Mentoring

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tool
  A specific named software product, programming language, framework, \
library, platform, cloud service, hardware system, or instrument that \
someone uses to perform work. If it has a product name or version, \
it is likely a Tool.
  Examples:
    Languages   — Python, Java, C++, JavaScript, TypeScript, R, Go, \
Rust, SQL, MATLAB, Scala, Kotlin, Swift, Ruby, PHP
    Frameworks  — React, Angular, Django, Flask, Spring Boot, .NET, \
Express.js, TensorFlow, PyTorch, Keras, LangChain, Hadoop, Spark
    Cloud       — AWS, Azure, GCP, Docker, Kubernetes, Terraform, \
Lambda, EC2, S3, BigQuery, Snowflake, Databricks, Heroku
    Databases   — PostgreSQL, MongoDB, MySQL, Redis, Elasticsearch, \
Cassandra, DynamoDB, Oracle, SQL Server, Neo4j
    DevOps      — Jenkins, GitHub Actions, GitLab CI, CircleCI, \
Ansible, Puppet, Prometheus, Grafana, Datadog, Splunk
    Design      — Figma, Sketch, Adobe Photoshop, Illustrator, \
InDesign, Canva, Adobe Premiere Pro, Final Cut Pro
    Business    — Salesforce, SAP, Tableau, Power BI, Jira, \
Confluence, Slack, HubSpot, QuickBooks, Tally, Excel, Google Analytics
    Healthcare  — Epic, Cerner, MEDITECH, SPSS, HL7 FHIR
    Legal       — Westlaw, LexisNexis, Relativity, DocuSign

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Certification
  A professional certification, accreditation, license, or formally \
awarded credential from a recognised body. Typically includes the \
issuing authority's name or a well-known acronym.
  Examples:
    Technology  — AWS Certified Solutions Architect, Google Cloud \
Professional Data Engineer, Certified Kubernetes Administrator (CKA), \
CompTIA Security+, Cisco CCNA, CISSP, CEH, TOGAF
    Finance     — CPA, CFA, FRM, Chartered Accountant, Series 7, \
ACCA, CFP
    Project Mgt — PMP, PRINCE2, Certified ScrumMaster (CSM), SAFe \
Agilist, Six Sigma Green Belt, Six Sigma Black Belt
    Healthcare  — RN License, Board Certified, USMLE, BLS, ACLS, CNA
    Legal       — Bar Admission, Certified Mediator
    HR          — SHRM-CP, PHR, SHRM-SCP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Degree
  An academic degree, diploma, educational qualification, or field \
of study from a university or educational institution.
  Examples: B.Tech, M.Tech, B.Sc, M.Sc, B.A., M.A., B.Com, M.Com, \
BCA, MCA, MBA, Ph.D, M.D., J.D., LL.B, LL.M, BBA, Associate Degree, \
Diploma in Nursing, Bachelor of Computer Science, \
Master of Business Administration, Doctor of Medicine, \
Computer Science, Information Technology, Electrical Engineering, \
Mechanical Engineering, Economics, Journalism, Psychology

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experience
  A mention of work duration, years of experience, or a seniority / \
role-level indicator.
  Examples: 3 years, 5+ years, 10+ years experience, 0-1 years, \
Fresher, Entry-Level, Junior, Mid-Level, Senior, Lead, Principal, \
Manager, Director, VP, CTO, Intern, Trainee, Associate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Other
  The span is noise — a person's name, a city or country, a company \
or organisation name, a date, a URL, an email address, a phone number, \
or any text that does not fit the above five categories and is not \
useful for resume screening.
  Examples: John Doe, Jane Smith, New York, India, San Francisco, \
Acme Corp, Google, Microsoft, 2020-2023, jan2023, @gmail.com, \
linkedin.com/in/..., References Available Upon Request

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES:
1. Classify EVERY span — do not skip any.
2. Each span gets exactly ONE category.
3. When in doubt between Skill and Tool, ask yourself: "Is this a \
named product/language I can download or subscribe to?" If yes → Tool. \
If it is a general competency → Skill.
4. Partial or truncated spans (e.g. "Ku", "Postg", "Mon") should be \
classified as Other since they are not meaningful on their own.

Return ONLY a valid JSON object where each key is the original span \
text exactly as given, and the value is its category label. Example:

{{"Python": "Tool", "Machine Learning": "Skill", "AWS Certified \
Solutions Architect": "Certification", "B.Tech": "Degree", \
"5+ years": "Experience", "John Doe": "Other"}}

Do NOT wrap the JSON in markdown code fences. Do NOT add any \
explanation, preamble, or commentary. Return raw JSON only.

CRITICAL: Your entire response must be a single valid JSON object \
and nothing else. Do not write any text before or after the JSON. \
Start your response with {{ and end it with }}.

Here are the spans to classify:
{spans_json}"""


def _classify_spans_with_llm(spans: list[str]) -> dict[str, str]:
    """Send all spans to the LLM (Ollama/Llama 3) in one batched call for classification.

    Args:
        spans: List of unique entity span strings from BERT.

    Returns:
        A dict mapping each span to its category label.

    Raises:
        RuntimeError: If the Ollama API call fails.
    """
    client = _get_ollama_client()
    model = os.environ.get("OLLAMA_MODEL", "llama3")

    # Format the spans as a JSON list for the prompt
    spans_json = json.dumps(spans, ensure_ascii=False)

    prompt = _CLASSIFICATION_PROMPT.format(spans_json=spans_json)

    print(f"[NER] Sending {len(spans)} spans to {model} for classification …")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            # Temperature 0 for deterministic, consistent classification
            temperature=0,
            max_tokens=4000,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama API call failed during span classification: {exc}\n"
            f"Check that Ollama is running at {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}."
        ) from exc

    raw_content = response.choices[0].message.content or ""
    print(f"[NER] {model} response received ({len(raw_content)} chars).")

    # Parse the JSON response, handling cases where the LLM might wrap it
    # in markdown code fences despite our instructions
    return _parse_classification_response(raw_content)


def _parse_classification_response(raw: str) -> dict[str, str]:
    """Parse the LLM's JSON response into a span → category mapping.

    Handles edge cases:
      - Markdown code fences (```json ... ```)
      - Preamble text before the JSON object
      - Malformed JSON (logs warning, returns empty dict)
    """
    text = raw.strip()

    # Strip markdown code fences if the LLM added them anyway
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find the first { ... } JSON object in the string
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            result = json.loads(text[brace_start : brace_end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # If all parsing fails, log the error and return empty
    print(f"[NER] WARNING — Could not parse LLM classification response.")
    print(f"[NER] Raw response (first 300 chars): {raw[:300]}")
    return {}


# ──────────────────────────────────────────────────────
# Public API — the two-stage pipeline
# ──────────────────────────────────────────────────────

# The five output categories (lowercase keys)
_CATEGORY_KEYS = {
    "Skill": "skills",
    "Tool": "tools",
    "Certification": "certifications",
    "Degree": "degrees",
    "Experience": "experience",
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract and classify entities from raw text using BERT + Llama 3.

    Stage 1: BERT (dslim/bert-base-NER) detects candidate entity spans.
    Stage 2: Llama 3 (via Ollama) classifies each span into a meaningful category.

    Args:
        text: Raw resume or JD text.

    Returns:
        A dictionary with five keys — ``"skills"``, ``"tools"``,
        ``"certifications"``, ``"degrees"``, ``"experience"`` — each
        mapping to a deduplicated list of entity strings.
        Spans classified as "Other" by the LLM are discarded.
    """
    # Initialise the output structure with empty lists
    result: dict[str, list[str]] = {
        "skills": [],
        "tools": [],
        "certifications": [],
        "degrees": [],
        "experience": [],
    }

    # Guard: empty input → return immediately, no API calls
    if not text or not text.strip():
        return result

    # ── STAGE 1: BERT span detection ─────────────────────
    spans = _extract_bert_spans(text)

    # Guard: if BERT found nothing, return empty without calling the LLM
    if not spans:
        print("[NER] BERT found no entity spans — skipping LLM.")
        return result

    print(f"[NER] BERT detected {len(spans)} unique spans.")

    # ── STAGE 2: LLM classification (Ollama / Llama 3) ───
    try:
        classifications = _classify_spans_with_llm(spans)
    except (EnvironmentError, RuntimeError) as exc:
        print(f"[NER] LLM classification failed: {exc}")
        # On failure, put all spans under "skills" as a safe fallback
        # so downstream components still have data to work with
        result["skills"] = spans
        return result

    # Group classified spans into the output categories.
    # Anything labeled "Other" is discarded as noise.
    for span, category in classifications.items():
        output_key = _CATEGORY_KEYS.get(category)
        if output_key:
            result[output_key].append(span)
        # If category is "Other" or unrecognised, the span is silently dropped

    classified_count = sum(len(v) for v in result.values())
    discarded_count = len(classifications) - classified_count
    print(
        f"[NER] Classification complete — "
        f"{classified_count} entities kept, {discarded_count} discarded as 'Other'."
    )

    return result
