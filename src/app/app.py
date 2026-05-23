"""
app.py - MyCareerGPT Main Application
Streamlit UI | Weeks 4-6

Run with: streamlit run src/app/app.py
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import sys
import os
import json
import time
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.database import save_user, save_recommendations, db_status, save_feedback
from src.rag.retrieval import RAGRetriever
from src.llm.llm_interface import LLMInterface, check_hallucinations
from src.resume_parser import parse_resume, split_skills, ALL_TECH_SKILLS, ALL_SOFT_SKILLS

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MyCareerGPT",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background: #021a14 !important; font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #073B3A 0%, #042820 100%) !important; border-right: 1px solid #0B6E4F !important; }
[data-testid="stSidebar"] * { color: #a8d5b5 !important; }
.main-header { background: linear-gradient(135deg, #073B3A 0%, #0B6E4F 60%, #08A045 100%); padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem; text-align: center; border: 1px solid #08A045; box-shadow: 0 0 40px rgba(8,160,69,0.2); position: relative; overflow: hidden; }
.main-header h1 { color: #21D375; font-size: 2.5rem; margin: 0; font-weight: 700; letter-spacing: -0.5px; }
.main-header p { color: #6BBF59; margin: 0.5rem 0 0 0; font-size: 1.05rem; font-weight: 300; }
.match-card { background: linear-gradient(135deg, #0a2a20 0%, #0d3526 100%); border: 1px solid #0B6E4F; border-left: 4px solid #08A045; padding: 1.4rem; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.match-card h3 { color: #21D375; margin: 0 0 0.3rem 0; }
.skill-badge-have { background: rgba(107,191,89,0.15); color: #6BBF59; border: 1px solid rgba(107,191,89,0.4); padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; display: inline-block; margin: 2px; font-family: 'JetBrains Mono', monospace; }
.skill-badge-gap { background: rgba(8,160,69,0.1); color: #08A045; border: 1px solid rgba(8,160,69,0.3); padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; display: inline-block; margin: 2px; font-family: 'JetBrains Mono', monospace; }
.confidence-high { color: #21D375; font-weight: 700; }
.confidence-medium { color: #6BBF59; font-weight: 700; }
.confidence-low { color: #08A045; font-weight: 700; }
.metric-box { background: linear-gradient(135deg, #0a2a20 0%, #073B3A 100%); border: 1px solid #0B6E4F; padding: 1.2rem; border-radius: 12px; text-align: center; box-shadow: 0 2px 15px rgba(0,0,0,0.2); }
.metric-box .value { font-size: 2.2rem; color: #21D375; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-box .label { color: #6BBF59; font-size: 0.85rem; margin-top: 4px; }
.stProgress > div > div > div { background-color: #08A045 !important; }
.stProgress > div > div { background-color: #073B3A !important; border-radius: 8px; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #08A045 0%, #0B6E4F 100%) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; letter-spacing: 0.3px !important; box-shadow: 0 4px 15px rgba(8,160,69,0.3) !important; transition: all 0.3s ease !important; }
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #21D375 0%, #08A045 100%) !important; box-shadow: 0 4px 25px rgba(33,211,117,0.4) !important; transform: translateY(-1px) !important; }
.stButton > button[kind="secondary"] { background: transparent !important; color: #6BBF59 !important; border: 1px solid #0B6E4F !important; border-radius: 10px !important; }
.stButton > button[kind="secondary"]:hover { border-color: #08A045 !important; color: #21D375 !important; }
.stAlert { border-radius: 10px !important; border-left-width: 4px !important; }
[data-baseweb="notification"] { background: #0a2a20 !important; border-color: #08A045 !important; }
[data-testid="stExpander"] { background: #0a2a20 !important; border: 1px solid #0B6E4F !important; border-radius: 10px !important; }
[data-testid="stExpander"]:hover { border-color: #08A045 !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div { background: #0a2a20 !important; border-color: #0B6E4F !important; color: #d4f5e0 !important; border-radius: 8px !important; }
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: #08A045 !important; box-shadow: 0 0 0 2px rgba(8,160,69,0.2) !important; }
[data-baseweb="tag"] { background: rgba(8,160,69,0.2) !important; color: #6BBF59 !important; border: 1px solid rgba(8,160,69,0.4) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child { background: #073B3A !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: #08A045 !important; border-color: #21D375 !important; }
[data-testid="stToggle"] [data-checked="true"] { background-color: #08A045 !important; }
hr { border-color: #0B6E4F !important; opacity: 0.4; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #021a14; }
::-webkit-scrollbar-thumb { background: #0B6E4F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #08A045; }
h1, h2, h3 { color: #21D375 !important; }
h4, h5, h6 { color: #6BBF59 !important; }
p, li, span, label { color: #c8ecd4 !important; }
.stMarkdown p { color: #c8ecd4 !important; }
[data-testid="stMarkdownContainer"] p { color: #c8ecd4 !important; }
[data-testid="stFileUploader"] { background: #0a2a20 !important; border: 2px dashed #0B6E4F !important; border-radius: 12px !important; }
[data-testid="stFileUploader"]:hover { border-color: #08A045 !important; }
[data-testid="stCheckbox"] [data-checked="true"] svg { color: #08A045 !important; fill: #08A045 !important; }
.resume-section-box { background: #0a2a20; border: 1px solid #0B6E4F; border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem; }
.ats-resume-box { background: #0a2a20; border: 1px solid #21D375; border-radius: 12px; padding: 1.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #c8ecd4; white-space: pre-wrap; line-height: 1.7; }
.feedback-card { background: linear-gradient(135deg, #0a2a20, #073B3A); border: 1px solid #0B6E4F; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "profile"
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "parsed_resume" not in st.session_state:
    st.session_state.parsed_resume = None
if "resume_sections" not in st.session_state:
    st.session_state.resume_sections = {}
if "generated_resume" not in st.session_state:
    st.session_state.generated_resume = ""
if "retriever" not in st.session_state:
    with st.spinner("Loading job database..."):
        st.session_state.retriever = RAGRetriever()
if "llm" not in st.session_state:
    st.session_state.llm = LLMInterface()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇲🇾 MyCareerGPT")
    st.markdown("---")

    pages = {
        "👤 My Profile":      "profile",
        "🎯 Recommendations": "recommendations",
        "📄 My ATS Resume":   "ats_resume",
        "💬 Feedback":        "feedback",
    }
    for label, page_id in pages.items():
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.page == page_id else "secondary"):
            st.session_state.page = page_id
            st.rerun()

    st.markdown("---")
    st.markdown("**System:**")
    st.markdown("🔍 TF-IDF Matching")
    st.markdown("✨ Gemini 2.5 Flash Reranking")
    st.markdown("🛡️ RAG (No Hallucinations)")
    st.markdown("🇲🇾 Malaysian Job Market")


# ═══════════════════════════════════════════════════════════════════════════════
# RESUME SECTION EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def extract_resume_sections(raw_text: str) -> dict:
    """
    Extract structured sections from resume raw text:
      - profile   : name, email, phone
      - skills    : tech + soft
      - experience: work history entries
      - projects  : project entries
      - education : degree entries with CGPA
    """
    import re

    sections = {
        "profile":    {},
        "skills":     {"tech": [], "soft": [], "raw": []},
        "experience": [],
        "projects":   [],
        "education":  [],
        "raw_text":   raw_text,
    }

    lines      = raw_text.split("\n")
    text_lower = raw_text.lower()

    # ── Profile ────────────────────────────────────────────────────────────
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", raw_text)
    if email_match:
        sections["profile"]["email"] = email_match.group(0)

    phone_match = re.search(r"(\+?6?01[0-9][-\s]?\d{7,8}|\+?\d[\d\s\-]{8,14})", raw_text)
    if phone_match:
        sections["profile"]["phone"] = phone_match.group(0).strip()

    for line in lines[:10]:
        line = line.strip()
        if line and re.match(r'^[A-Za-z ]{3,50}$', line) and len(line.split()) >= 2:
            sections["profile"]["name"] = line
            break

    # ── Skills ────────────────────────────────────────────────────────────
    from src.resume_parser import KNOWN_SKILLS, ALL_TECH_SKILLS, ALL_SOFT_SKILLS
    found_skills = []
    for skill in KNOWN_SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    soft_lower = {s.lower() for s in ALL_SOFT_SKILLS}
    sections["skills"]["tech"] = [s for s in found_skills if s.lower() not in soft_lower]
    sections["skills"]["soft"] = [s for s in found_skills if s.lower() in soft_lower]
    sections["skills"]["raw"]  = found_skills

    # ── Experience & Projects ─────────────────────────────────────────────
    exp_pattern  = re.compile(r'(experience|employment|work history|career history)', re.IGNORECASE)
    proj_pattern = re.compile(r'(project|portfolio)', re.IGNORECASE)

    current_section = None
    exp_text_lines  = []
    proj_text_lines = []

    for line in lines:
        ll = line.strip().lower()
        if exp_pattern.search(line) and len(ll) < 40:
            current_section = "experience"
            continue
        elif proj_pattern.search(line) and len(ll) < 40:
            current_section = "projects"
            continue
        elif re.search(r'\b(skills?|certification|summary|objective|education|reference)\b', ll) and len(ll) < 40:
            current_section = None

        if current_section == "experience" and line.strip():
            exp_text_lines.append(line.strip())
        elif current_section == "projects" and line.strip():
            proj_text_lines.append(line.strip())

    date_re = re.compile(
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|'
        r'April|June|July|August|September|October|November|December)[\s,]*\d{4}',
        re.IGNORECASE,
    )
    year_re = re.compile(r'\b(20\d{2}|19\d{2})\b')

    def parse_entries(text_lines):
        entries = []
        current   = None
        desc_lines = []
        for line in text_lines:
            is_header = (
                (date_re.search(line) or year_re.search(line))
                and len(line) < 120
                and not line.startswith("•")
                and not line.startswith("-")
            )
            if is_header and current is None:
                current    = {"header": line, "description": ""}
                desc_lines = []
            elif is_header and current:
                current["description"] = " ".join(desc_lines)
                entries.append(current)
                current    = {"header": line, "description": ""}
                desc_lines = []
            elif current:
                desc_lines.append(line)
        if current:
            current["description"] = " ".join(desc_lines)
            entries.append(current)
        return entries

    sections["experience"] = parse_entries(exp_text_lines)
    sections["projects"]   = parse_entries(proj_text_lines)

    # ── Education ─────────────────────────────────────────────────────────
    cgpa_re = re.compile(r'(?:cgpa|gpa|pointer)[:\s]+([0-9]\.[0-9]{1,2})', re.IGNORECASE)
    degree_keywords = ["bachelor", "master", "phd", "diploma", "degree",
                       "msc", "bsc", "sarjana", "sarjana muda"]
    edu_entries = []
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in degree_keywords):
            entry = {"text": line.strip(), "cgpa": None}
            for offset in range(5):
                if i + offset < len(lines):
                    m = cgpa_re.search(lines[i + offset])
                    if m:
                        entry["cgpa"] = float(m.group(1))
                        break
            edu_entries.append(entry)
    sections["education"] = edu_entries

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# ATS RESUME GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ats_resume(profile: dict, recs: list, llm: LLMInterface) -> str:
    """
    Use Gemini 2.5 Flash to generate an ATS-friendly resume tailored to the
    top job recommendation, based on the user's profile and resume sections.
    """
    top_job     = recs[0] if recs else {}
    all_skills  = profile.get("skills", "")
    matched     = ", ".join(top_job.get("matched_skills", []))
    gaps        = ", ".join(top_job.get("skill_gaps", []))
    work_history = profile.get("work_history", "Not provided")
    projects     = profile.get("projects", "Not provided")

    prompt = f"""You are a professional resume writer specialising in ATS (Applicant Tracking System) optimisation for the Malaysian job market.

CRITICAL RULES — YOU MUST FOLLOW THESE:
1. NEVER change, invent, or modify the candidate's company names in Work Experience.
2. Use the EXACT company name from the candidate's work history: {work_history}
3. The target job company ({top_job.get('company', '')}) is where they are APPLYING — NOT where they worked.
4. Only add skills to the Skills section — never fabricate job titles or companies.
5. Return ONLY the resume text, no preamble or explanation.

CANDIDATE PROFILE:
- Name         : {profile.get('name', 'Candidate')}
- Email        : {profile.get('email', '')}
- Education    : {profile.get('education', '')} in {profile.get('field', '')}
- University   : {profile.get('university', '')}
- CGPA         : {profile.get('cgpa', '')}
- Experience   : {profile.get('experience', 0)} years
- All Skills   : {all_skills}
- Work History (USE EXACT COMPANY NAMES): {work_history}
- Projects     : {projects}

TARGET JOB (this is where they are APPLYING, not where they worked):
- Title        : {top_job.get('job_title', '')}
- Company      : {top_job.get('company', '')}
- Matched Skills: {matched}
- Skills to Highlight: {gaps}

INSTRUCTIONS:
1. Write a complete resume with these sections: Summary, Skills, Work Experience, Projects, Education
2. Use KEYWORDS from the target job naturally throughout the summary and skills
3. Start bullet points with strong action verbs (Developed, Implemented, Analysed, etc.)
4. Quantify achievements where possible (e.g. "Improved accuracy by 15%")
5. Write a FULL and COMPLETE resume — do not cut it short
6. Use clean plain-text formatting with clear section headers
7. Write in third-person implied style (no "I" statements)
8. Make it ATS-parseable — no tables, no columns
9. Do NOT use markdown formatting — no asterisks (**bold**), no # headers, no underscores
10. Use PLAIN TEXT only — every section header should just be written in CAPITALS
11. Do not truncate or summarise — write the full content for every section

OUTPUT FORMAT — use this exact structure:

[FULL NAME]
[Email] | Malaysia

PROFESSIONAL SUMMARY
[2-3 sentences tailored to the target job: {top_job.get('job_title', '')}]

TECHNICAL SKILLS
[Comma-separated relevant skills, prioritising matched ones]

SOFT SKILLS
[Key soft skills]

WORK EXPERIENCE
[Exact job title from work history] | [EXACT company from work history — do NOT use target company here] | [Duration]
• [Achievement bullet using action verb]
• [Achievement bullet with quantified result]

PROJECTS
[Project Name] | [Technologies Used]
• [What you built and impact]

EDUCATION
[Degree] in [Field]
[University] | [Year] | CGPA: {profile.get('cgpa', '')}
"""

    try:
        result = llm.generate_recommendations(prompt, max_tokens=10000, temperature=0.4)
        raw = result.get("raw_text", "")
        # Strip any RECOMMENDATION blocks if the parser ran on it
        if "RECOMMENDATION" in raw:
            raw = raw.split("RECOMMENDATION")[0].strip()
        # Strip markdown formatting Gemini sometimes adds
        import re as _re
        raw = _re.sub(r"\*\*(.*?)\*\*", r"", raw)   # Remove **bold**
        raw = _re.sub(r"\*(.*?)\*",   r"", raw)         # Remove *italic*
        raw = _re.sub(r"^#+\s+", "", raw, flags=_re.MULTILINE) # Remove # headers
        raw = raw.strip()
        return raw if raw else "❌ Could not generate resume. Please try again."
    except Exception as e:
        return f"❌ Resume generation failed: {e}"


def _build_ats_resume_pdf(resume_text: str, name: str) -> bytes:
    """Convert the generated ATS resume text into a downloadable PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("Name", fontSize=16, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#0B6E4F"),
                                 alignment=TA_CENTER, spaceAfter=4)
    contact_style = ParagraphStyle("Contact", fontSize=9, alignment=TA_CENTER,
                                    textColor=colors.HexColor("#444444"), spaceAfter=10)
    section_style = ParagraphStyle("Section", fontSize=11, fontName="Helvetica-Bold",
                                    textColor=colors.HexColor("#08A045"),
                                    spaceBefore=10, spaceAfter=4)
    body_style    = ParagraphStyle("Body", fontSize=9, leading=14,
                                    textColor=colors.HexColor("#222222"), spaceAfter=3)

    story = []
    lines = resume_text.strip().split("\n")
    i = 0

    # Section header keywords
    SECTIONS = {"PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "SOFT SKILLS",
                "WORK EXPERIENCE", "PROJECTS", "EDUCATION", "SKILLS",
                "EXPERIENCE", "CERTIFICATIONS"}

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            story.append(Spacer(1, 0.15*cm))
            i += 1
            continue

        # First non-empty line = name
        if i == 0 or (i <= 2 and not any(s in line.upper() for s in SECTIONS)):
            if i == 0:
                story.append(Paragraph(line, name_style))
            else:
                story.append(Paragraph(line, contact_style))
            i += 1
            continue

        # Section headers
        if any(line.upper().startswith(s) for s in SECTIONS):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#08A045")))
            story.append(Paragraph(line, section_style))
            i += 1
            continue

        # Bullet points
        if line.startswith("•") or line.startswith("-"):
            clean = line.lstrip("•- ").strip()
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;• {clean}", body_style))
            i += 1
            continue

        # Regular text
        story.append(Paragraph(line, body_style))
        i += 1

    doc.build(story)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# PDF RECOMMENDATIONS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _build_recommendations_pdf(profile: dict, recs: list) -> bytes:
    """Build a styled PDF report of the recommendations."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle,
    )
    from reportlab.lib.enums import TA_CENTER

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    GREEN_DARK  = colors.HexColor("#08A045")
    GREEN_MID   = colors.HexColor("#0B6E4F")
    GREEN_LIGHT = colors.HexColor("#21D375")
    WHITE       = colors.white

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], fontSize=20, textColor=WHITE,
        backColor=GREEN_MID, alignment=TA_CENTER, spaceAfter=4,
        leftIndent=-1*cm, rightIndent=-1*cm, borderPad=12,
    )
    subtitle_style  = ParagraphStyle("Sub",   parent=styles["Normal"], fontSize=10,
                                      textColor=GREEN_LIGHT, alignment=TA_CENTER, spaceAfter=16)
    section_style   = ParagraphStyle("Sec",   parent=styles["Heading2"], fontSize=13,
                                      textColor=WHITE, backColor=GREEN_DARK,
                                      spaceAfter=6, spaceBefore=10,
                                      leftIndent=-0.5*cm, rightIndent=-0.5*cm, borderPad=6)
    rec_title_style = ParagraphStyle("RecT",  parent=styles["Heading3"], fontSize=12,
                                      textColor=GREEN_DARK, spaceAfter=4, spaceBefore=8)
    label_style     = ParagraphStyle("Lbl",   parent=styles["Normal"], fontSize=9,
                                      textColor=GREEN_MID, fontName="Helvetica-Bold", spaceAfter=2)
    body_style      = ParagraphStyle("Body",  parent=styles["Normal"], fontSize=9,
                                      textColor=colors.HexColor("#333333"), spaceAfter=4, leading=14)
    small_style     = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8,
                                      textColor=colors.HexColor("#666666"), spaceAfter=2)

    story = []

    # Header
    story.append(Paragraph("MyCareerGPT — Career Recommendations", title_style))
    story.append(Paragraph(
        f"Generated for {profile.get('name','Candidate')} on {time.strftime('%d %b %Y, %H:%M')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN_DARK))
    story.append(Spacer(1, 0.3*cm))

    # Profile table
    story.append(Paragraph("Candidate Profile", section_style))
    story.append(Spacer(1, 0.2*cm))
    profile_data = [
        ["Name",       profile.get("name", ""),           "Field",      profile.get("field", "")],
        ["Email",      profile.get("email", ""),          "Education",  profile.get("education", "")],
        ["CGPA",       str(profile.get("cgpa", "")),      "Experience", f"{profile.get('experience',0)} years"],
        ["University", profile.get("university", ""),     "RIASEC",     profile.get("riasec_type", "")],
    ]
    if profile.get("skills"):
        sk = profile["skills"][:80] + ("..." if len(profile["skills"]) > 80 else "")
        profile_data.append(["Skills", sk, "", ""])

    pt = Table(profile_data, colWidths=[3*cm, 7*cm, 3*cm, 4*cm])
    pt.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",       (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",      (0, 0), (0, -1), GREEN_MID),
        ("TEXTCOLOR",      (2, 0), (2, -1), GREEN_MID),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f0faf4"), WHITE]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#c8ecd4")),
        ("PADDING",        (0, 0), (-1, -1), 6),
        ("SPAN",           (1, -1), (3, -1)),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.4*cm))

    # Recommendations
    story.append(Paragraph("Your Top Job Matches", section_style))
    for i, rec in enumerate(recs, 1):
        story.append(Spacer(1, 0.3*cm))
        match_pct  = rec.get("match_percent", 0)
        confidence = rec.get("confidence", "Medium")

        rh = Table([[
            Paragraph(f"#{i}  {rec.get('job_title','')} @ {rec.get('company','')}", rec_title_style),
            Paragraph(f"{confidence} Match — {match_pct:.1f}%", body_style),
        ]], colWidths=[11*cm, 6*cm])
        rh.setStyle(TableStyle([
            ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING",  (0, 0), (-1, -1), 4),
            ("LINEBELOW",(0, 0), (-1, 0), 1, GREEN_LIGHT),
        ]))
        story.append(rh)
        story.append(Spacer(1, 0.15*cm))

        matched_str = ", ".join(rec.get("matched_skills", [])) or "N/A"
        gaps_str    = ", ".join(rec.get("skill_gaps", [])) or "None — you match all!"
        details = [
            [Paragraph("<b>Location</b>",       label_style), Paragraph(rec.get("location","Malaysia"), body_style)],
            [Paragraph("<b>Why It Matches</b>", label_style), Paragraph(rec.get("why_match","Based on your skill profile."), body_style)],
            [Paragraph("<b>Skills You Have</b>",label_style), Paragraph(matched_str, body_style)],
            [Paragraph("<b>Skills to Learn</b>",label_style), Paragraph(gaps_str, body_style)],
            [Paragraph("<b>Learning Roadmap</b>",label_style),Paragraph(rec.get("learning_path","N/A"), body_style)],
        ]
        dt = Table(details, colWidths=[4*cm, 13*cm])
        dt.setStyle(TableStyle([
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ("PADDING",        (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f7fdf9"), WHITE]),
            ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#d4f5e0")),
        ]))
        story.append(dt)
        if i < len(recs):
            story.append(HRFlowable(width="100%", thickness=0.5, color=GREEN_LIGHT))

    # Footer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN_DARK))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Generated by MyCareerGPT | TF-IDF + Gemini 2.5 Flash | FYP02-DS-T2610-0382",
        small_style,
    ))

    doc.build(story)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PROFILE WIZARD
# ═══════════════════════════════════════════════════════════════════════════════

def show_profile_page():
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Build Your Career Profile</h1>
        <p>Upload your resume — AI will extract your profile, skills and experience,
           then generate personalised recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Resume Upload ──────────────────────────────────────────────────────
    st.subheader("📄 Upload Resume (Optional)")
    st.caption("Supported formats: PDF, DOCX — your details will be auto-filled below")

    uploaded_resume = st.file_uploader(
        "Drop your resume here",
        type=["pdf", "docx", "doc", "txt"],
        label_visibility="collapsed",
    )

    parsed = st.session_state.parsed_resume

    if uploaded_resume is not None:
        file_bytes = uploaded_resume.read()
        with st.spinner("📖 Reading and extracting resume sections..."):
            parsed = parse_resume(file_bytes, uploaded_resume.name)
            st.session_state.parsed_resume = parsed
            raw_text = parsed.get("raw_text", "")
            if raw_text:
                sections = extract_resume_sections(raw_text)
                st.session_state.resume_sections = sections

    if parsed:
        sections   = st.session_state.resume_sections
        conf_color = {"high": "#1a4d3a", "medium": "#4d3a00", "low": "#4d1a1a"}.get(
            parsed.get("confidence", "low"), "#4d1a1a"
        )
        conf_icon = {"high": "✅", "medium": "⚠️", "low": "❌"}.get(
            parsed.get("confidence", "low"), "❌"
        )
        skills_found = len(parsed.get("skills", []))
        st.markdown(
            f'<div style="background:{conf_color};border-radius:8px;padding:0.8rem 1rem;'
            f'margin-bottom:1rem;color:#fff">'
            f'{conf_icon} Resume parsed — found <b>{skills_found} skills</b>, '
            f'education: <b>{parsed.get("education") or "not detected"}</b>, '
            f'CGPA: <b>{parsed.get("cgpa") or "not detected"}</b> · '
            f'Confidence: <b>{parsed.get("confidence","low").upper()}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for w in parsed.get("warnings", []):
            st.warning(w)

        # ── Extracted Sections Tabs ────────────────────────────────────────
        with st.expander("📋 View Extracted Resume Sections", expanded=False):
            tabs = st.tabs(["👤 Profile", "💡 Skills", "💼 Experience & Projects", "🎓 Education"])

            with tabs[0]:
                profile_info = sections.get("profile", {})
                if profile_info:
                    for k, v in profile_info.items():
                        st.markdown(f"**{k.title()}:** {v}")
                else:
                    st.info("No profile details automatically detected.")

            with tabs[1]:
                tech = sections.get("skills", {}).get("tech", [])
                soft = sections.get("skills", {}).get("soft", [])
                if tech:
                    st.markdown("**Technical Skills:**")
                    st.markdown(", ".join(f"`{s}`" for s in tech))
                if soft:
                    st.markdown("**Soft Skills:**")
                    st.markdown(", ".join(f"`{s}`" for s in soft))
                if not tech and not soft:
                    st.info("No skills detected.")

            with tabs[2]:
                exp_entries  = sections.get("experience", [])
                proj_entries = sections.get("projects", [])
                if exp_entries:
                    st.markdown("**Work Experience:**")
                    for i, e in enumerate(exp_entries[:5], 1):
                        st.markdown(f"**{i}.** {e.get('header','')}")
                        if e.get("description"):
                            desc = e["description"]
                            st.caption(desc[:200] + ("..." if len(desc) > 200 else ""))
                else:
                    st.info("No work experience entries detected.")
                if proj_entries:
                    st.markdown("**Projects:**")
                    for i, e in enumerate(proj_entries[:5], 1):
                        st.markdown(f"**{i}.** {e.get('header','')}")
                        if e.get("description"):
                            desc = e["description"]
                            st.caption(desc[:200] + ("..." if len(desc) > 200 else ""))
                else:
                    st.info("No project entries detected.")

            with tabs[3]:
                edu_entries = sections.get("education", [])
                if edu_entries:
                    for e in edu_entries[:3]:
                        cgpa_str = f" — CGPA: {e['cgpa']}" if e.get("cgpa") else ""
                        st.markdown(f"• {e['text']}{cgpa_str}")
                else:
                    st.info("No education entries detected.")

            with st.expander("📝 Raw extracted text", expanded=False):
                st.text(parsed.get("raw_text", "")[:2000] + "...")

        if st.button("✕ Clear resume & start fresh"):
            st.session_state.parsed_resume = None
            st.session_state.resume_sections = {}
            st.rerun()

    st.markdown("---")

    # ── Pre-fill helpers ───────────────────────────────────────────────────
    def _default(key, fallback):
        if parsed and parsed.get(key):
            return parsed[key]
        return fallback

    def _default_skills(skill_list, parsed_skills):
        if not parsed_skills:
            return []
        parsed_lower = {s.lower() for s in parsed_skills}
        return [s for s in skill_list if s.lower() in parsed_lower]

    # ── Form ───────────────────────────────────────────────────────────────
    with st.form("profile_form"):
        st.subheader("📋 Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            name  = st.text_input("Full Name *", placeholder="Please insert your full name")
            email = st.text_input("Email *",     placeholder="Please insert your email")
        with col2:
            edu_options = ["Bachelor's Degree", "Master's Degree", "PhD",
                           "Diploma", "Foundation", "SPM", "Other"]
            edu_default = _default("education", "Bachelor's Degree")
            edu_idx     = edu_options.index(edu_default) if edu_default in edu_options else 0
            education   = st.selectbox("Highest Education", edu_options, index=edu_idx)
            education_other = st.text_input(
                "If 'Other', specify your education level:",
                placeholder="e.g. Professional Certificate, Short Course",
            )
            cgpa_default = float(_default("cgpa", 3.0) or 3.0)
            cgpa = st.slider("CGPA", min_value=0.0, max_value=4.0,
                             value=cgpa_default, step=0.01, format="%.2f")

        col3, col4 = st.columns(2)
        with col3:
            field_options = [
                "Computer Science", "Data Science", "Software Engineering",
                "Information Technology", "Business Administration",
                "Electrical Engineering", "Mechanical Engineering",
                "Mathematics", "Statistics", "Finance & Accounting",
                "Psychology", "Other",
            ]
            field_default = _default("field", "Computer Science")
            field_idx     = field_options.index(field_default) if field_default in field_options else 0
            field         = st.selectbox("Field of Study", field_options, index=field_idx)
            field_other   = st.text_input(
                "If 'Other', specify your field of study:",
                placeholder="e.g. Actuarial Science, Mass Communication",
            )
        with col4:
            uni_options = [
                "Multimedia University (MMU)", "University of Malaya (UM)",
                "Universiti Teknologi Malaysia (UTM)", "Universiti Putra Malaysia (UPM)",
                "Universiti Kebangsaan Malaysia (UKM)", "Universiti Teknologi MARA (UiTM)",
                "UTAR", "Taylor's University", "Sunway University", "APU",
                "Monash University Malaysia", "Other",
            ]
            uni_default      = _default("university", "Multimedia University (MMU)")
            uni_idx          = uni_options.index(uni_default) if uni_default in uni_options else 0
            university       = st.selectbox("University", uni_options, index=uni_idx)
            university_other = st.text_input(
                "If 'Other', specify your university:",
                placeholder="e.g. Universiti Malaysia Sabah, Lincoln University",
            )

        st.markdown("---")
        st.subheader("💡 Skills & Experience")

        parsed_skills = parsed.get("skills", []) if parsed else []
        if parsed_skills:
            st.info(f"📄 **{len(parsed_skills)} skills auto-detected from resume.** "
                    f"Review and adjust below.")

        parsed_tech, parsed_soft = split_skills(parsed_skills)

        tech_skills = st.multiselect(
            "Technical Skills", options=ALL_TECH_SKILLS,
            default=_default_skills(ALL_TECH_SKILLS, parsed_tech) or ["Python", "SQL"],
            help="Auto-filled from resume. Add or remove as needed.",
        )
        tech_other = st.text_input(
            "Other Technical Skills (comma separated):",
            placeholder="e.g. MATLAB, Tableau, Power BI, SAP",
        )
        soft_skills = st.multiselect(
            "Soft Skills", options=ALL_SOFT_SKILLS,
            default=_default_skills(ALL_SOFT_SKILLS, parsed_soft) or ["Communication", "Teamwork"],
        )
        soft_other = st.text_input(
            "Other Soft Skills (comma separated):",
            placeholder="e.g. Mentoring, Conflict Resolution, Empathy",
        )

        if parsed_skills:
            known_lower = {s.lower() for s in ALL_TECH_SKILLS + ALL_SOFT_SKILLS}
            extra       = [s for s in parsed_skills if s.lower() not in known_lower]
            if extra:
                st.caption(f"📌 Additional skills from resume (included automatically): "
                           f"{', '.join(extra)}")
        else:
            extra = []

        # Show detected experience summary
        sections     = st.session_state.resume_sections
        exp_entries  = sections.get("experience", [])
        proj_entries = sections.get("projects", [])
        if exp_entries or proj_entries:
            parts = [e.get("header", "") for e in exp_entries[:3]]
            parts += [f"[Project] {e.get('header','')}" for e in proj_entries[:2]]
            if parts:
                st.markdown("**📋 Detected Experience from Resume:**")
                st.caption(" | ".join(parts))

        col5, col6 = st.columns(2)
        with col5:
            exp_default = int(_default("experience", 0) or 0)
            experience  = st.number_input(
                "Years of Work Experience", min_value=0, max_value=20,
                value=min(exp_default, 20),
            )
        with col6:
            interests = st.text_area(
                "Career Interests / Goals",
                placeholder="E.g. I want to work in fintech, AI research, "
                            "or data-driven companies in Malaysia...",
                height=80,
            )

        st.markdown("---")
        st.subheader("🧭 Career Personality (RIASEC)")
        st.caption("Select 2–3 types that best describe you:")
        st.markdown(
            "💡 Don't know your RIASEC type? "
            "[Take the free test here](https://openpsychometrics.org/tests/RIASEC/)",
            unsafe_allow_html=True,
        )

        riasec_options = {
            "R - Realistic":     "Hands-on, technical, mechanical",
            "I - Investigative": "Analytical, research, problem-solving",
            "A - Artistic":      "Creative, expressive, design-oriented",
            "S - Social":        "Helping, teaching, teamwork",
            "E - Enterprising":  "Leadership, business, entrepreneurial",
            "C - Conventional":  "Detail-oriented, organized, data entry",
        }
        riasec_cols    = st.columns(3)
        selected_riasec = []
        for i, (code, desc) in enumerate(riasec_options.items()):
            with riasec_cols[i % 3]:
                if st.checkbox(f"**{code}** \n_{desc}_"):
                    selected_riasec.append(code[0])

        st.markdown("---")
        submitted = st.form_submit_button(
            "🔍 Find My Career Matches", use_container_width=True
        )

    if submitted:
        if not name or not email:
            st.error("Please fill in your Name and Email.")
            return

        final_education  = education_other.strip()  if education  == "Other" else education
        final_field      = field_other.strip()       if field      == "Other" else field
        final_university = university_other.strip()  if university == "Other" else university

        if education  == "Other" and not final_education:
            st.error("Please specify your education level."); return
        if field      == "Other" and not final_field:
            st.error("Please specify your field of study."); return
        if university == "Other" and not final_university:
            st.error("Please specify your university."); return

        extra_tech = [s.strip() for s in tech_other.split(",") if s.strip()]
        extra_soft = [s.strip() for s in soft_other.split(",") if s.strip()]
        all_skills = tech_skills + extra_tech + soft_skills + extra_soft + extra
        seen       = set()
        all_skills = [s for s in all_skills if not (s.lower() in seen or seen.add(s.lower()))]

        riasec_str  = "".join(selected_riasec[:3]) or "IRA"
        sections    = st.session_state.resume_sections
        exp_context = " | ".join(e.get("header","") for e in sections.get("experience",[])[:3])
        proj_context= " | ".join(e.get("header","") for e in sections.get("projects",[])[:2])

        profile = {
            "name":         name,
            "email":        email,
            "education":    final_education,
            "field":        final_field,
            "university":   final_university,
            "cgpa":         cgpa,
            "skills":       ", ".join(all_skills),
            "experience":   experience,
            "riasec_type":  riasec_str,
            "interests":    interests,
            "resume_used":  uploaded_resume is not None or parsed is not None,
            "work_history": exp_context,
            "projects":     proj_context,
        }

        st.session_state.user_profile = profile
        st.session_state.page         = "recommendations"

        try:
            save_user(name, email, final_education, final_field, final_university,
                      cgpa, ", ".join(all_skills), experience, riasec_str)
        except Exception:
            pass

        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def show_recommendations_page():
    profile = st.session_state.user_profile
    if not profile:
        st.warning("Please complete your profile first.")
        if st.button("Go to Profile"):
            st.session_state.page = "profile"
            st.rerun()
        return

    st.markdown(f"""
    <div class="main-header">
        <h1>🎯 Your Career Recommendations</h1>
        <p>Matched from Malaysian jobs using TF-IDF + Gemini 2.5 Flash</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👤 Your Profile Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Name:** {profile.get('name')}")
            st.markdown(f"**Field:** {profile.get('field')}")
            st.markdown(f"**CGPA:** {profile.get('cgpa')}")
        with col2:
            st.markdown(f"**Education:** {profile.get('education')}")
            st.markdown(f"**Experience:** {profile.get('experience')} years")
            st.markdown(f"**RIASEC:** {profile.get('riasec_type')}")
        with col3:
            skills_list = profile.get("skills", "").split(", ")[:6]
            st.markdown("**Top Skills:**")
            for s in skills_list:
                st.markdown(f" • {s}")
        if profile.get("work_history"):
            st.markdown(f"**Work History (from resume):** {profile['work_history']}")
        if profile.get("projects"):
            st.markdown(f"**Projects (from resume):** {profile['projects']}")

    st.markdown("---")

    col_btn, col_mode = st.columns([3, 1])
    with col_mode:
        use_llm = st.toggle("Use LLM Reranking", value=True)
    with col_btn:
        generate = st.button("🚀 Generate Recommendations", use_container_width=True, type="primary")

    if generate or st.session_state.recommendations:
        if generate:
            with st.spinner("Step 1/2: TF-IDF matching across Malaysia Job Market..."):
                candidates = st.session_state.retriever.retrieve(profile, top_n=20)
            st.success(f"✅ Found {len(candidates)} candidate jobs")

            if use_llm:
                with st.spinner("Step 2/2: AI reranking & explanation generation..."):
                    prompt     = st.session_state.retriever.build_prompt(profile, candidates)
                    llm_result = st.session_state.llm.generate_recommendations(prompt)
                    recs       = llm_result.get("recommendations", [])
                    inference_time = llm_result.get("inference_time", 0)

                if "error" in llm_result:
                    err = llm_result["error"]
                    if "memory" in err.lower() or "ram" in err.lower():
                        st.error("❌ Not enough RAM for LLM. Showing TF-IDF results instead.")
                    elif "connect" in err.lower() or "refused" in err.lower():
                        st.error("❌ Cannot connect to Gemini 2.5 Flash.\n\n**Fix:** Check Docker is running: docker compose up -d")
                    else:
                        st.warning(f"⚠️ LLM error: {err}\n\nShowing TF-IDF results instead.")

                hall_check  = check_hallucinations(recs, candidates)
                user_skills = [s.strip().lower() for s in profile.get("skills", "").split(",")]

                for rec in recs:
                    for c in candidates:
                        rec_title  = rec["job_title"].lower().split(" at ")[0].strip()
                        cand_title = c["title"].lower().strip()
                        if rec_title in cand_title or cand_title in rec_title:
                            rec["tfidf_score"]     = c.get("cv_tfidf_score", 0) or c.get("tfidf_score", 0)
                            rec["match_percent"]   = c.get("match_percent", 0)
                            rec["location"]        = c.get("location", "Malaysia")
                            rec["skills_required"] = c.get("skills_required", "")
                            rec["job_id"]          = c.get("id", 0)
                            matched = c.get("matched_skills", [])
                            gaps    = c.get("skill_gaps", [])
                            if not matched:
                                required = [s.strip() for s in (c.get("skills_required","") or "").split(",") if s.strip()]
                                matched  = [s for s in required if s.lower() in user_skills]
                                gaps     = [s for s in required if s.lower() not in user_skills]
                            if not rec.get("matched_skills"):
                                rec["matched_skills"] = matched
                            if not rec.get("skill_gaps"):
                                rec["skill_gaps"] = gaps
                            break

                if not recs:
                    st.info("ℹ️ Showing top 5 TF-IDF matches (LLM unavailable).")
                    recs           = _candidates_to_recs(candidates[:5])
                    inference_time = 0
                    hall_check     = {"passed": True, "hallucinations": 0}
            else:
                recs           = _candidates_to_recs(candidates[:5])
                inference_time = 0
                hall_check     = {"passed": True, "hallucinations": 0}

            st.session_state.recommendations = recs

            try:
                save_user(
                    profile.get("name"), profile.get("email"),
                    profile.get("education"), profile.get("field"),
                    profile.get("university"), profile.get("cgpa"),
                    profile.get("skills"), profile.get("experience"),
                    profile.get("riasec_type"),
                )
                from src.database import get_connection
                conn   = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE email = ?", (profile.get("email"),))
                row     = cursor.fetchone()
                user_id = row[0] if row else None
                conn.close()
                if user_id:
                    save_recommendations(user_id, [
                        {
                            "job_id":        r.get("job_id", 0),
                            "tfidf_score":   r.get("tfidf_score", 0),
                            "llm_score":     r.get("llm_score", 0),
                            "final_score":   r.get("match_percent", 0),
                            "rank_position": i + 1,
                            "explanation":   r.get("why_match", ""),
                            "matched_skills":r.get("matched_skills", []),
                            "skill_gaps":    r.get("skill_gaps", []),
                        }
                        for i, r in enumerate(recs)
                    ])
            except Exception as e:
                print(f"DB save error: {e}")

            st.session_state.inference_time = inference_time
            st.session_state.hall_check     = hall_check

        # ── Metrics ───────────────────────────────────────────────────────
        recs       = st.session_state.recommendations
        inf_time   = st.session_state.get("inference_time", 0)
        hall_check = st.session_state.get("hall_check", {})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-box"><div class="value">{len(recs)}</div>
            <div class="label">Top Matches</div></div>""", unsafe_allow_html=True)
        with m2:
            avg_match = sum(r.get("match_percent", 0) for r in recs) / max(len(recs), 1)
            st.markdown(f"""<div class="metric-box"><div class="value">{avg_match:.0f}%</div>
            <div class="label">Avg Match</div></div>""", unsafe_allow_html=True)
        with m3:
            time_str = f"{inf_time:.1f}s" if inf_time > 0 else "TF-IDF"
            st.markdown(f"""<div class="metric-box"><div class="value">{time_str}</div>
            <div class="label">Inference Time</div></div>""", unsafe_allow_html=True)
        with m4:
            hall_status = "✅ 0" if hall_check.get("passed") else f"❌ {hall_check.get('hallucinations',0)}"
            st.markdown(f"""<div class="metric-box"><div class="value">{hall_status}</div>
            <div class="label">Hallucinations</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Download PDF ───────────────────────────────────────────────────
        if recs:
            try:
                pdf_bytes = _build_recommendations_pdf(profile, recs)
                st.download_button(
                    label="⬇️ Download Recommendations as PDF",
                    data=pdf_bytes,
                    file_name=f"career_recommendations_{profile.get('name','candidate').replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=False,
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}. Run: pip install reportlab")

        st.subheader("🏆 Your Top Job Matches")

        for i, rec in enumerate(recs, 1):
            confidence = rec.get("confidence", "Medium")
            conf_class = f"confidence-{confidence.lower()}"
            match_pct  = rec.get("match_percent", 0)

            with st.expander(
                f"#{i} 🎯 {rec.get('job_title','Job')} "
                f"@ {rec.get('company','Company')} "
                f"[{confidence} match]",
                expanded=(i == 1),
            ):
                col_left, col_right = st.columns([2, 1])
                with col_left:
                    if rec.get("why_match"):
                        st.info(f"🤖 **AI Insight:** {rec['why_match']}")
                    st.markdown("**Skills Breakdown:**")
                    have_html = "".join(
                        f'<span class="skill-badge-have">✅ {s}</span>'
                        for s in rec.get("matched_skills", [])
                    ) or "<i>—</i>"
                    gap_html = "".join(
                        f'<span class="skill-badge-gap">📚 {s}</span>'
                        for s in rec.get("skill_gaps", [])
                    ) or "<i>None! You have all required skills.</i>"
                    st.markdown(f"You Have: {have_html}", unsafe_allow_html=True)
                    st.markdown(f"To Learn: {gap_html}", unsafe_allow_html=True)
                    if rec.get("learning_path") and rec["learning_path"].lower() not in ("n/a","none",""):
                        st.markdown("**📚 Learning Roadmap:**")
                        st.markdown(rec["learning_path"])
                with col_right:
                    st.markdown("**Match Score:**")
                    st.progress(min(match_pct / 100, 1.0))
                    st.markdown(f"**{match_pct:.1f}%** compatibility")
                    st.markdown(f"**📍 Location:** {rec.get('location','Malaysia')}")
                    st.markdown(
                        f"""**Confidence:** <span class="{conf_class}">{confidence}</span>""",
                        unsafe_allow_html=True,
                    )
                    if rec.get("confidence_reason"):
                        st.caption(rec["confidence_reason"])
                    if rec.get("tfidf_score"):
                        st.markdown(f"**TF-IDF Score:** `{rec['tfidf_score']:.4f}`")

        # ── Skill Gap Analysis ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📉 Your Skill Gap Analysis")
        all_gaps = []
        for rec in recs:
            all_gaps.extend(rec.get("skill_gaps", []))
        if all_gaps:
            from collections import Counter
            import pandas as pd
            gap_counts = Counter(all_gaps).most_common(10)
            gap_df = pd.DataFrame({
                "Skill":     [g[0] for g in gap_counts],
                "Frequency": [g[1] for g in gap_counts],
            }).set_index("Skill")
            st.bar_chart(gap_df)
        else:
            st.success("🎉 No skill gaps identified across your recommendations!")

        # ── Prompt to generate ATS resume ─────────────────────────────────
        st.markdown("---")
        st.info("📄 **Want an ATS-optimised resume?** Head to the **My ATS Resume** page "
                "to generate a tailored resume based on your top job match.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📄 Generate My ATS Resume →", type="primary"):
                st.session_state.page = "ats_resume"
                st.rerun()
        with col_b:
            if st.button("💬 Leave Feedback →", type="secondary"):
                st.session_state.page = "feedback"
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ATS RESUME GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def show_ats_resume_page():
    profile = st.session_state.user_profile
    recs    = st.session_state.recommendations

    st.markdown("""
    <div class="main-header">
        <h1>📄 ATS-Friendly Resume Generator</h1>
        <p>Generate an optimised resume tailored to your top job match using Gemini 2.5 Flash AI</p>
    </div>
    """, unsafe_allow_html=True)

    if not profile:
        st.warning("Please complete your profile first.")
        if st.button("Go to Profile"):
            st.session_state.page = "profile"
            st.rerun()
        return

    if not recs:
        st.warning("Please generate recommendations first so we can tailor your resume.")
        if st.button("Go to Recommendations"):
            st.session_state.page = "recommendations"
            st.rerun()
        return

    # Show target job
    top_job = recs[0]
    st.markdown(f"""
    <div style="background:#0a2a20;border:1px solid #0B6E4F;border-radius:10px;padding:1rem;margin-bottom:1rem;">
        <div style="color:#21D375;font-weight:700;font-size:1.1rem;">🎯 Tailoring resume for:</div>
        <div style="color:#c8ecd4;margin-top:4px;">
            <b>{top_job.get('job_title','')}</b> @ {top_job.get('company','')}
            &nbsp;|&nbsp; Match: {top_job.get('match_percent',0):.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **What this does:**
    - Reads your profile, skills, work history and projects
    - Incorporates keywords from your top matched job
    - Writes a complete resume in ATS-parseable format
    - Includes action verbs, quantified achievements, and proper sections
    """)

    col_gen, col_info = st.columns([2, 1])
    with col_info:
        st.info("💡 ATS (Applicant Tracking System) scans resumes for keywords "
                "before a human reads them. This resume is optimised to pass that scan.")

    with col_gen:
        generate_btn = st.button(
            "✨ Generate My ATS Resume", use_container_width=True, type="primary"
        )

    if generate_btn:
        with st.spinner("🤖 Gemini is writing your ATS-optimised resume..."):
            resume_text = generate_ats_resume(profile, recs, st.session_state.llm)
            st.session_state.generated_resume = resume_text
        st.success("✅ Resume generated!")

    # Display generated resume
    if st.session_state.generated_resume:
        resume_text = st.session_state.generated_resume

        st.markdown("---")
        st.subheader("📄 Your Generated ATS Resume")

        # Allow editing
        edited_resume = st.text_area(
            "Review and edit your resume below:",
            value=resume_text,
            height=600,
            help="You can edit the generated resume before downloading.",
        )
        st.session_state.generated_resume = edited_resume

        st.markdown("---")
        col_dl1, col_dl2, col_regen = st.columns(3)

        with col_dl1:
            # Download as PDF
            try:
                resume_pdf = _build_ats_resume_pdf(
                    edited_resume, profile.get("name", "candidate")
                )
                st.download_button(
                    label="⬇️ Download as PDF",
                    data=resume_pdf,
                    file_name=f"ATS_Resume_{profile.get('name','candidate').replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF failed: {e}. Run: pip install reportlab")

        with col_dl2:
            # Download as plain text (for copy-paste into job portals)
            st.download_button(
                label="⬇️ Download as .txt",
                data=edited_resume.encode("utf-8"),
                file_name=f"ATS_Resume_{profile.get('name','candidate').replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_regen:
            if st.button("🔄 Regenerate", use_container_width=True, type="secondary"):
                with st.spinner("Regenerating..."):
                    new_resume = generate_ats_resume(profile, recs, st.session_state.llm)
                    st.session_state.generated_resume = new_resume
                st.rerun()

        st.markdown("---")
        st.markdown("**💡 Tips for using this resume:**")
        st.markdown(
            "- Copy the plain text version when applying on job portals (JobStreet, LinkedIn)\n"
            "- Use the PDF version when emailing directly to HR\n"
            "- Always review and personalise before submitting\n"
            "- Add your actual project links, GitHub, and LinkedIn URL manually"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

def show_feedback_page():
    st.markdown("""
    <div class="main-header">
        <h1>💬 Your Feedback</h1>
        <p>Help us improve MyCareerGPT — your opinion matters!</p>
    </div>
    """, unsafe_allow_html=True)

    profile = st.session_state.user_profile
    prefill_email = profile.get("email", "") if profile else ""
    prefill_name  = profile.get("name",  "") if profile else ""

    st.markdown("""
    <div class="feedback-card">
        <p>This feedback is used for UAT (User Acceptance Testing) evaluation.
           It takes about 2 minutes. Thank you for your time! 🙏</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("feedback_form"):
        col1, col2 = st.columns(2)
        with col1:
            fb_name  = st.text_input("Your Name",  value=prefill_name,  placeholder="e.g. Ahmad Rizal")
        with col2:
            fb_email = st.text_input("Your Email", value=prefill_email, placeholder="e.g. ahmad@example.com")

        st.markdown("---")
        st.subheader("⭐ Rate Your Experience")
        st.caption("1 = Very Poor, 5 = Excellent")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            overall_rating = st.slider("Overall Satisfaction",      min_value=1, max_value=5, value=4)
        with col_b:
            rec_quality    = st.slider("Recommendation Quality",    min_value=1, max_value=5, value=4)
        with col_c:
            ease_of_use    = st.slider("Ease of Use",               min_value=1, max_value=5, value=4)

        st.markdown("---")
        st.subheader("📝 Your Thoughts")
        comments = st.text_area(
            "Comments & Suggestions",
            placeholder="What did you like? What can be improved? "
                        "Were the recommendations relevant to your profile?",
            height=120,
        )
        would_recommend = st.checkbox(
            "✅ I would recommend MyCareerGPT to other students / job seekers",
            value=True,
        )

        st.markdown("---")
        st.subheader("🔬 Quick Usability Questions (SUS)")
        st.caption("Answer based on your overall experience.")

        sus_q1 = st.radio("1. I think I would like to use this system frequently.",
                           ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],
                           index=3, horizontal=True)
        sus_q2 = st.radio("2. I found the system unnecessarily complex.",
                           ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],
                           index=1, horizontal=True)
        sus_q3 = st.radio("3. I thought the system was easy to use.",
                           ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],
                           index=3, horizontal=True)
        sus_q4 = st.radio("4. I felt the AI recommendations were trustworthy.",
                           ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],
                           index=3, horizontal=True)
        sus_q5 = st.radio("5. I would need technical support to use this system.",
                           ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],
                           index=1, horizontal=True)

        st.markdown("---")
        submit_fb = st.form_submit_button(
            "✅ Submit Feedback", use_container_width=True, type="primary"
        )

    if submit_fb:
        if not fb_email:
            st.error("Please enter your email so we can track responses.")
            return

        sus_map = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
                   "Agree": 4, "Strongly Agree": 5}
        sus_score = (
            (sus_map[sus_q1] - 1) + (5 - sus_map[sus_q2]) +
            (sus_map[sus_q3] - 1) + (sus_map[sus_q4] - 1) +
            (5 - sus_map[sus_q5])
        ) * 2.5

        full_comments = (
            f"{comments}\n\n"
            f"[SUS] Q1:{sus_q1}|Q2:{sus_q2}|Q3:{sus_q3}|Q4:{sus_q4}|Q5:{sus_q5} "
            f"[SUS Score: {sus_score:.1f}]"
        )

        try:
            save_feedback(
                user_email=fb_email,
                rating=overall_rating,
                rec_quality=rec_quality,
                ease_of_use=ease_of_use,
                comments=full_comments,
                would_recommend=would_recommend,
            )
            st.success(
                f"🎉 Thank you, {fb_name or 'participant'}! Your feedback has been recorded.\n\n"
                f"**Your SUS Score contribution: {sus_score:.1f}/100** (Target: ≥ 80)"
            )
            st.balloons()
        except Exception as e:
            st.error(f"Could not save feedback: {e}")


# ── Helper ────────────────────────────────────────────────────────────────────
def _candidates_to_recs(candidates):
    """Convert TF-IDF candidates to recommendation format (no LLM)."""
    profile     = st.session_state.user_profile
    user_skills = [s.strip().lower() for s in profile.get("skills", "").split(",")]
    results     = []
    for c in candidates:
        matched = c.get("matched_skills", [])
        gaps    = c.get("skill_gaps", [])
        if not matched:
            required = [s.strip() for s in (c.get("skills_required","") or "").split(",") if s.strip()]
            matched  = [s for s in required if s.lower() in user_skills]
            gaps     = [s for s in required if s.lower() not in user_skills]
        results.append({
            "job_title":      c.get("title", ""),
            "company":        c.get("company", ""),
            "location":       c.get("location", "Malaysia"),
            "why_match":      "Matched based on TF-IDF skill similarity (LLM reranking disabled).",
            "matched_skills": matched,
            "skill_gaps":     gaps,
            "learning_path":  "Enable LLM reranking for personalised learning paths.",
            "confidence":     "Medium",
            "tfidf_score":    c.get("cv_tfidf_score", 0) or c.get("tfidf_score", 0),
            "match_percent":  c.get("match_percent", 0),
            "job_id":         c.get("id", 0),
        })
    return results


# ── Router ────────────────────────────────────────────────────────────────────
def main():
    page = st.session_state.page
    if page == "profile":
        show_profile_page()
    elif page == "recommendations":
        show_recommendations_page()
    elif page == "ats_resume":
        show_ats_resume_page()
    elif page == "feedback":
        show_feedback_page()


if __name__ == "__main__":
    main()