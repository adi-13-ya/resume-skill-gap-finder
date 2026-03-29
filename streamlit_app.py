import os
import sys
import tempfile
import pandas as pd
import streamlit as st
from pathlib import Path
from streamlit_extras.metric_cards import style_metric_cards

# Ensure backend modules can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.document_parser import parse_document
from src.analyzer import analyze_resume
from src.phase3_evaluator import evaluate_resume
from src.kb_loader import KnowledgeBaseIndex
from src.routing import get_jd_text

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Resume Skill Gap Finder",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Custom CSS
st.markdown("""
    <style>
    /* Premium background and fonts */
    .stApp {
        background-color: #1a1a1a;
        color: #f0ebd8;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Elegant Hero typography */
    h1 {
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f7f1e3;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #d1ccc0;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards customization */
    div[data-testid="metric-container"] {
        background-color: #262626 !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px 0 rgba(0,0,0,0.3) !important;
        padding: 1rem !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px 0 rgba(0,0,0,0.4) !important;
        border-color: #b59868 !important; /* Soft gold hover */
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #b59868 0%, #8e7345 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 10px rgba(181, 152, 104, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #c7a771 0%, #9e7f4c 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(181, 152, 104, 0.5);
    }
    
    /* Card Containers */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
        background: #212121;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Floating background animation */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-divider {
        height: 3px;
        background: linear-gradient(270deg, #b59868, #3498db, #e74c3c, #b59868);
        background-size: 800% 800%;
        animation: gradient-shift 8s ease infinite;
        margin: 1rem 0 2rem 0;
        border-radius: 2px;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# HELPER FUNCTIONS & CACHING
# ==========================================
@st.cache_resource(show_spinner=False)
def get_kb_index():
    """Load the knowledge base index once globally."""
    kb = KnowledgeBaseIndex().load_all()
    return kb

@st.cache_data(show_spinner=False)
def load_supported_roles():
    """Extract list of roles from the Knowledge Base."""
    kb = get_kb_index()
    roles = []
    for r in kb._all_roles: # Access internal list safely
        roles.append({
            "JobID": r.get("JobID", ""),
            "Title": r.get("Title", "Unknown Role"),
            "Level": r.get("ExperienceLevel", "Experienced"),
            "Domain": r.get("Domain", "Engineering")
        })
    # Sort for dropdown
    return sorted(roles, key=lambda x: f"{x['Domain']} - {x['Title']} ({x['Level']})")

def extract_text(uploaded_file, fallback_text: str) -> str:
    """Robust document extraction mapped safely via backend `parse_document`."""
    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            parsed_text = parse_document(tmp_path)
            return parsed_text.strip()
        except Exception as e:
            st.error(f"File Parsing Error ({uploaded_file.name}): {str(e)}")
            return ""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return fallback_text.strip()

@st.cache_data(show_spinner=False)
def run_backend_analysis(resume_text: str, jd_text: str, fast_mode: bool, jd_metadata: dict):
    """Execute Phase 2 and Phase 3 caching identical payloads."""
    # Phase 2 Analysis
    phase2 = analyze_resume(
        resume_text=resume_text,
        jd_text=jd_text,
        domain=jd_metadata.get("Domain", "Engineering"),
        level=jd_metadata.get("ExperienceLevel", "Experienced"),
        skip_gpt=fast_mode
    )

    # Phase 3 Scoring
    phase3 = evaluate_resume(
        resume_entities=phase2.get("resume_entities", {}),
        jd_entities=phase2.get("jd_entities", {}),
        jd_metadata=jd_metadata,
        resume_text=resume_text
    )
    
    return phase2, phase3

def safe_extract_missing_skills(phase2_result):
    """Computes exact missing skills dynamically to avoid blanks."""
    jd_skills_raw = phase2_result.get("jd_entities", {}).get("skills", []) + phase2_result.get("jd_entities", {}).get("tools", [])
    resume_skills_raw = phase2_result.get("resume_entities", {}).get("skills", []) + phase2_result.get("resume_entities", {}).get("tools", [])
    
    # 1. Start with O*NET explicit missing
    missing = set(phase2_result.get("skill_gap", {}).get("jd_skills_missing_in_resume", []))
    
    # 2. Backfill with raw text un-matched to guarantee no blind spots
    resume_lower = {s.lower() for s in resume_skills_raw}
    for js in jd_skills_raw:
        if js.lower() not in resume_lower:
            missing.add(js)
            
    return sorted(list(missing), key=str.casefold)

def render_pill_tags(items, icon="•", color_hex="#ffffff"):
    """Renders text cleanly grouped like chips/bullets."""
    if not items:
        st.markdown("*None identified.*")
        return
    for item in items:
        # Markdown prevents text truncation natively
        st.markdown(f"<span style='color:{color_hex}; font-weight:500;'>{icon}</span> {item}", unsafe_allow_html=True)


# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # --- HERO SECTION ---
    st.markdown("<h1>✨ ScopeWise Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>An AI-powered SaaS tool mapping your trajectory to the perfect role. Upload your resume and discover your exact skill gaps instantly.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-divider'></div>", unsafe_allow_html=True)

    # --- MINIMAL SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Preferences")
        fast_mode = st.checkbox("⚡ Fast Mode", value=False, help="Skips deep LLM reasoning. Analysis runs near-instantly without tailored recommendations.")

    # --- INPUT SECTION ---
    col_input1, col_input2 = st.columns(2, gap="large")

    with col_input1:
        st.subheader("📄 Candidate Profile")
        with st.container(border=True):
            resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"], key="rf_up")
            with st.expander("Or paste raw text"):
                resume_text_input = st.text_area("Paste Resume Text", height=200, key="rf_tx", label_visibility="collapsed")

    with col_input2:
        st.subheader("🎯 Target Position")
        with st.container(border=True):
            kb_roles = load_supported_roles()
            role_options = ["Custom Job Description"] + [f"[{r['JobID']}] {r['Domain']} - {r['Title']} ({r['Level']})" for r in kb_roles]
            
            target_selection = st.selectbox("Select Benchmark Role", options=role_options)
            
            jd_file, jd_text_input = None, ""
            if target_selection == "Custom Job Description":
                jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"], key="jd_up")
                with st.expander("Or paste raw text"):
                    jd_text_input = st.text_area("Paste JD Text", height=130, key="jd_tx", label_visibility="collapsed")

    st.write("")
    
    # --- ANALYZE ACTION ---
    action_c1, action_c2, action_c3 = st.columns([1, 2, 1])
    with action_c2:
        analyze_btn = st.button("🚀 Execute Analysis", type="primary", use_container_width=True)

    # --- ANALYSIS EXECUTION ---
    if analyze_btn:
        # 1. Resolve Resume Text
        resume_text = extract_text(resume_file, resume_text_input)
        if not resume_text:
            st.warning("⚠️ Please provide a Candidate Profile (upload or paste text).")
            return

        # 2. Resolve Job Description Text
        kb_index = get_kb_index()
        target_role_data = None
        jd_metadata = {"Title": "Custom User Job", "ExperienceLevel": "Experienced", "YearsOfExperience": "0", "Domain": "Engineering"}

        if target_selection != "Custom Job Description":
            # Parse JobID safely from the selection string (e.g. "[NET-F-001] ...")
            job_id = target_selection.split("]")[0].replace("[", "").strip()
            target_role_data = kb_index.get_by_id(job_id)
            if target_role_data:
                jd_text = get_jd_text(role_id=job_id, kb_index=kb_index)
                jd_metadata.update({
                    "Title": target_role_data.get("Title", ""),
                    "ExperienceLevel": target_role_data.get("ExperienceLevel", "Experienced"),
                    "YearsOfExperience": target_role_data.get("YearsOfExperience", "0"),
                    "Domain": target_role_data.get("Domain", "Engineering")
                })
            else:
                st.error("Failed to load selected role from Knowledge Base.")
                return
        else:
            jd_text = extract_text(jd_file, jd_text_input)
            if not jd_text:
                st.warning("⚠️ Please provide a Target Position (upload or paste text).")
                return

        # 3. Execution Wrapper
        with st.spinner("🧠 Running deep structural and semantic analysis..."):
            try:
                # Backend Trigger
                phase2_result, phase3_result = run_backend_analysis(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    fast_mode=fast_mode,
                    jd_metadata=jd_metadata
                )

                # Extracts
                score = phase3_result.get("fuzzy_crisp_score", 0.0)
                verdict = phase3_result.get("verdict", "Unknown")
                feedback = phase3_result.get("structural_analysis", {}).get("structural_feedback", "Evaluated successfully.")
                
                matched_items = phase2_result.get("onet_resume_skills", {}).get("matched", [])
                matched_skills = [m.get("onet_match", m.get("raw", "")) for m in matched_items]
                missing_skills = safe_extract_missing_skills(phase2_result)

                # --- RESULT DASHBOARD ---
                st.markdown("<div class='hero-divider' style='height: 1px; opacity: 0.3; margin-top: 3rem;'></div>", unsafe_allow_html=True)
                st.subheader("📊 Intelligence Synthesis")
                st.info(f"**Final Verdict: {verdict}** — {feedback}")

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Match Score", f"{score * 100:.1f}%")
                m2.metric("Matched Core Skills", len(matched_skills))
                m3.metric("Critical Gaps", len(missing_skills))
                style_metric_cards(background_color="transparent", border_left_color="#b59868", border_color="#3d3d3d")

                st.write("")
                st.progress(score, text="Strategic Alignment Probability")
                st.write("")

                # --- SKILL COMPARISON ---
                st.subheader("🔬 Competency Delta")
                
                chart_col, delta_col = st.columns([1, 2], gap="large")
                
                with chart_col:
                    st.write("##### Profile Breakdown")
                    chart_df = pd.DataFrame({
                        "Category": ["Matched", "Missing"],
                        "Count": [len(matched_skills), len(missing_skills)]
                    })
                    st.bar_chart(chart_df.set_index("Category"), color=["#b59868"])

                with delta_col:
                    st.write("##### Granular Requirements")
                    skill_c1, skill_c2 = st.columns(2)
                    
                    with skill_c1:
                        st.markdown("<p style='color:#a3e4d7; font-weight:600;'>✓ Confirmed Competencies</p>", unsafe_allow_html=True)
                        with st.container(border=True):
                            render_pill_tags(matched_skills, icon="✓", color_hex="#a3e4d7")

                    with skill_c2:
                        st.markdown("<p style='color:#f1948a; font-weight:600;'>✕ Deficient Areas</p>", unsafe_allow_html=True)
                        with st.container(border=True):
                            render_pill_tags(missing_skills, icon="✕", color_hex="#f1948a")

                st.markdown("<br>", unsafe_allow_html=True)

                # --- RECOMMENDATIONS ---
                st.subheader("💡 Actionable Directives")
                gpt_res = phase2_result.get("gpt_pros_cons", {})
                
                if fast_mode:
                    st.info("⚡ Fast Mode active. Deep LLM tailored recommendations were bypassed.")
                elif gpt_res.get("_error"):
                    st.warning("Tailored AI directives currently unavailable (LLM connection failed).")
                else:
                    rec1, rec2 = st.columns(2, gap="large")
                    with rec1:
                        with st.container(border=True):
                            st.markdown("#### 🌟 Profile Strengths")
                            pros = gpt_res.get("pros", [])
                            if pros:
                                for p in pros:
                                    text = p.get('point', p) if isinstance(p, dict) else p
                                    st.success(f"{text}", icon="🎯")
                            else:
                                st.write("*No distinct strengths surfaced.*")
                            
                    with rec2:
                        with st.container(border=True):
                            st.markdown("#### 🔧 Strategic Remediation")
                            cons = gpt_res.get("cons", [])
                            if cons:
                                for c in cons:
                                    text = c.get('point', c) if isinstance(c, dict) else c
                                    st.error(f"{text}", icon="🚨")
                            else:
                                st.write("*No critical vulnerabilities detected.*")

                st.markdown("<br>", unsafe_allow_html=True)

                # --- DETAILED DATA VIEWER ---
                st.subheader("🗂️ Telemetry & Extraction Logs")
                
                # Rendered purely using clean tabs to avoid letter squishing and massive chunks
                t1, t2, t3, t4 = st.tabs([
                    "Extracted Resume DNA", 
                    "Extracted Target DNA",
                    "Raw Resume Payload", 
                    "Raw Target Payload"
                ])
                
                resume_raw_skills = phase2_result.get("resume_entities", {}).get("skills", []) + phase2_result.get("resume_entities", {}).get("tools", [])
                jd_raw_skills = phase2_result.get("jd_entities", {}).get("skills", []) + phase2_result.get("jd_entities", {}).get("tools", [])

                with t1:
                    st.markdown("*(Independently validated skills & tools)*")
                    if resume_raw_skills:
                        st.markdown(", ".join(resume_raw_skills))
                    else:
                        st.write("No skills identified natively.")

                with t2:
                    st.markdown("*(Required core capabilities)*")
                    if jd_raw_skills:
                        st.markdown(", ".join(jd_raw_skills))
                    else:
                        st.write("No skills identified natively.")

                with t3:
                    st.markdown("*(Analyzed source structure)*")
                    with st.container(height=400):
                        st.text(resume_text)

                with t4:
                    st.markdown("*(Analyzed benchmark structure)*")
                    with st.container(height=400):
                        st.text(jd_text)

            except Exception as e:
                st.error("System Failure Mode Engaged: Unexpected Backend Exception")
                st.exception(e)

if __name__ == "__main__":
    main()
