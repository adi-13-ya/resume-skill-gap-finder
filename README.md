# ✦ ScopeWise — AI-Powered Resume-Job Matching & Explainable Recommendation System

> Go beyond keyword matching. ScopeWise uses Fuzzy Inference, Sentence-BERT semantic embeddings, and an LLM explanation layer to tell you *why* a resume fits a role — not just *if* it does.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20Sentence--BERT-orange)](https://sbert.net/)
[![Fuzzy Logic](https://img.shields.io/badge/Fuzzy%20Logic-scikit--fuzzy-purple)](https://pythonhosted.org/scikit-fuzzy/)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20OpenAI--compatible-red)](https://ollama.com/)

---

## 🧠 What is ScopeWise?

Traditional Applicant Tracking Systems (ATS) reject qualified candidates for the wrong reasons — wrong formatting, missing exact keywords, synonyms unrecognised. A candidate who writes *"team coordination"* when the JD says *"collaboration"* gets filtered out, not because they're underqualified, but because the system was never built to understand language.

ScopeWise fixes this with a four-stage intelligent pipeline:

1. **Document Parsing** — Ingests resumes and JDs in PDF or DOCX format
2. **NER Skill Extraction** — Extracts technical and soft skills using HuggingFace Transformers, normalised against the O\*NET framework via RapidFuzz
3. **Fuzzy Inference Matching** — Scores semantic similarity using Sentence-BERT, then classifies fit as **Strong / Moderate / Weak Match** using a Fuzzy Inference System (FIS) — no hard binary thresholds
4. **LLM Explainability (XAI)** — Generates plain-language explanations for every decision, including a full Skill Gap Analysis of what's missing and why it matters

---

## 🏗️ Architecture

```
Resume (PDF/DOCX)          Job Description (PDF/DOCX or KB Role ID)
       │                                │
       ▼                                ▼
 ┌─────────────────────────────────────────────┐
 │         Phase 1: Document Parsing           │
 │   pdfplumber · python-docx · text cleaning  │
 └───────────────────┬─────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────┐
 │      Phase 2: NER Skill Extraction          │
 │  HuggingFace Transformers · RapidFuzz       │
 │  O*NET normalization · Skill dictionaries   │
 └───────────────────┬─────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────┐
 │    Phase 3: Semantic Similarity + FIS       │
 │  Sentence-BERT embeddings · scikit-fuzzy    │
 │  Inputs: Semantic Score + Skill Overlap %   │
 │  Output: Strong / Moderate / Weak Match     │
 └───────────────────┬─────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────┐
 │     Phase 4: LLM Explainability (XAI)       │
 │  Ollama (OpenAI-compatible) · LangChain     │
 │  Pros/Cons · Skill Gap Analysis · Verdict   │
 └─────────────────────────────────────────────┘
```

---

## ⚡ Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/yashvardhandebas/resume-skill-gap-finder-main.git
cd resume-skill-gap-finder-main
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your Ollama base URL or OpenAI-compatible API key
```

### 4. Run the analysis

```bash
# Analyse your resume against a Kaggle KB role by Job ID
python run_analysis.py --resume path/to/resume.pdf --role-id NET-F-001

# Analyse against a custom JD file
python run_analysis.py --resume path/to/resume.pdf --jd path/to/jd.pdf

# Specify domain and experience level
python run_analysis.py --resume resume.pdf --role-id NET-E-011 \
  --domain Engineering --level Experienced

# Skip the LLM step (if Ollama is not running)
python run_analysis.py --resume resume.pdf --role-id NET-F-001 --skip-gpt

# Save full JSON output
python run_analysis.py --resume resume.pdf --jd jd.pdf --output result.json
```

---

## 📊 Sample Output

```
════════════════════════════════════════════════════════════
 SKILL GAP
════════════════════════════════════════════════════════════
  Missing from resume (required by JD):
  ⚠ Docker
  ⚠ Kubernetes
  ⚠ System Design

  Extra skills in resume (not in JD):
  + RAG Pipelines
  + Vector Embeddings
  + LangChain

════════════════════════════════════════════════════════════
 LLM PROS & CONS
════════════════════════════════════════════════════════════
  PROS:
  ✅ Strong Python and ML fundamentals align with core JD requirements
     Candidate demonstrates hands-on experience with PyTorch and Scikit-learn
  ✅ Project experience with LLM pipelines shows genuine AI engineering depth

  CONS:
  ❌ No cloud infrastructure experience (Docker, Kubernetes)
     Role requires containerised deployment — this is a critical gap
```

---

## 🗂️ Project Structure

```
resume-skill-gap-finder-main/
│
├── src/
│   ├── document_parser.py     # Phase 1: PDF/DOCX text extraction
│   ├── kb_loader.py           # Kaggle knowledge base loader & indexer
│   ├── routing.py             # JD source routing (file vs KB)
│   └── analyzer.py            # Core pipeline orchestrator
│
├── data/                      # Job description knowledge base (Kaggle)
├── outputs/                   # Analysis results (JSON)
│
├── run_analysis.py            # Main CLI entry point
├── run_phase3.py              # Run FIS matching in isolation
├── run_phase4.py              # Run LLM explanation layer in isolation
│
├── test_phase1.py             # Unit tests: document parsing
├── test_phase2.py             # Unit tests: NER extraction
├── test_phase3.py             # Unit tests: fuzzy inference
├── test_phase4.py             # Unit tests: LLM explanation
│
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Document Parsing | `pdfplumber`, `python-docx` |
| NER & Skill Extraction | `HuggingFace Transformers`, `torch` |
| O\*NET Normalization | `RapidFuzz` |
| Semantic Similarity | `Sentence-BERT` (sentence-transformers) |
| Fuzzy Inference System | `scikit-fuzzy`, `scipy`, `numpy` |
| LLM Explainability | `Ollama` via OpenAI-compatible SDK, `openai`, `python-dotenv` |

---

## 📈 Results

Evaluated on benchmark resume-JD pairs across Engineering, Commerce, and Journalism domains:

- **75.7% match accuracy** on Excellent Fit profiles (21 matched core skills, 17 critical gaps identified)
- FIS successfully distinguishes Strong / Moderate / Weak matches without hard binary thresholds
- LLM explanation layer generates actionable skill gap rationale, reducing the need for manual recruiter interpretation

---

## 🔬 Research Context

Built as part of the **Soft Computing** course at Vellore Institute of Technology (VIT Vellore) under **Dr. Chiranji Lal Chowdhary**.

Informed by literature including:
- *Smart-Hiring: An Explainable End-to-End Pipeline for CV Matching* (NER + semantic embeddings)
- *When LLM Meets Fuzzy-TOPSIS for Personnel Selection* (Hoque et al., 2026)
- *Towards Bias-Free Recruitment: Adversarial Contrastive Tuning* (Wu et al., 2026)
- *ESCOX: Skill and Occupation Extraction Using LLMs* (Kavargyris et al., 2025)

---

## 👥 Team

| Name | Roll No |
|---|---|
| Aditya Gujar | 23BIT0224 |
| Yashvardhan Debas | 23BIT0109 |
| Hamza Raza Khan | 23BIT0252 |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with Python · Fuzzy Logic · Sentence-BERT · LLMs · ❤️
</p>
