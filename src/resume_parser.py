"""
resume_parser.py - Resume/CV Parser

Extracts structured information from uploaded PDF or DOCX resumes:
  - Skills (matched against known skill vocabulary)
  - Education level & field of study
  - Years of experience
  - University name
  - CGPA

Usage:
    from src.resume_parser import parse_resume
    result = parse_resume(uploaded_file_bytes, filename="resume.pdf")
"""

import re
import io


# ── Known vocabularies for extraction ─────────────────────────────────────────

KNOWN_SKILLS = [
    # Programming languages
    "Python", "R", "Java", "JavaScript", "TypeScript", "C++", "C#", "C",
    "PHP", "Swift", "Kotlin", "Go", "Rust", "MATLAB", "Scala", "Ruby",
    # Data & ML
    "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
    "Computer Vision", "Data Analysis", "Data Science", "Data Mining",
    "Statistics", "Statistical Analysis", "Data Visualization",
    "Feature Engineering", "Model Deployment", "MLOps",
    # ML Libraries
    "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
    "NLTK", "SpaCy", "Hugging Face", "OpenCV",
    # Databases
    "SQL", "MySQL", "PostgreSQL", "SQLite", "MongoDB", "Redis",
    "Oracle", "MS SQL", "NoSQL", "Firebase",
    # BI & Reporting
    "Tableau", "Power BI", "Excel", "Google Sheets", "Looker",
    "SPSS", "SAS",
    # Web
    "HTML", "CSS", "React", "Vue", "Angular", "Node.js",
    "Django", "Flask", "FastAPI", "Laravel", "Spring Boot",
    # DevOps & Cloud
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git", "GitHub",
    "CI/CD", "Jenkins", "Terraform", "Linux", "Bash",
    # Project & Soft
    "Project Management", "Agile", "Scrum", "JIRA", "Confluence",
    "Leadership", "Communication", "Teamwork", "Problem Solving",
    "Critical Thinking", "Presentation", "Time Management",
    "Negotiation", "Customer Service",
    # Finance
    "Financial Modeling", "Accounting", "Bloomberg", "VBA",
    "Financial Analysis", "Risk Management",
    # Networking
    "Networking", "Cisco", "TCP/IP", "Cybersecurity", "Firewall",
]

EDUCATION_KEYWORDS = {
    "PhD":              ["phd", "ph.d", "doctor of philosophy", "doctorate"],
    "Master's Degree":  ["master", "msc", "m.sc", "mba", "m.eng", "master of"],
    "Bachelor's Degree":["bachelor", "bsc", "b.sc", "b.eng", "degree", "b.cs",
                         "bachelor of", "sarjana muda"],
    "Diploma":          ["diploma", "dip."],
    "Foundation":       ["foundation", "a-level", "a level", "stpm"],
    "SPM":              ["spm", "sijil pelajaran"],
}

FIELD_KEYWORDS = {
    "Computer Science":       ["computer science", "cs", "computing"],
    "Data Science":           ["data science", "data analytics", "data engineering"],
    "Software Engineering":   ["software engineering", "software development"],
    "Information Technology": ["information technology", "it", "information system",
                               "information systems"],
    "Electrical Engineering": ["electrical engineering", "electronics engineering",
                               "electronic engineering"],
    "Mechanical Engineering": ["mechanical engineering"],
    "Civil Engineering":      ["civil engineering"],
    "Business Administration":["business administration", "business management",
                               "management", "mba"],
    "Finance & Accounting":   ["finance", "accounting", "accountancy",
                               "financial", "actuarial"],
    "Mathematics":            ["mathematics", "math", "applied mathematics"],
    "Statistics":             ["statistics", "actuarial science"],
    "Psychology":             ["psychology"],
}

MALAYSIAN_UNIVERSITIES = {
    "Multimedia University (MMU)":            ["multimedia university", "mmu"],
    "University of Malaya (UM)":             ["university of malaya", "universiti malaya", " um "],
    "Universiti Teknologi Malaysia (UTM)":   ["universiti teknologi malaysia", "utm"],
    "Universiti Putra Malaysia (UPM)":       ["universiti putra malaysia", "upm"],
    "Universiti Kebangsaan Malaysia (UKM)":  ["universiti kebangsaan", "ukm"],
    "Universiti Teknologi MARA (UiTM)":      ["universiti teknologi mara", "uitm"],
    "Universiti Utara Malaysia (UUM)":       ["universiti utara malaysia", "uum"],
    "Universiti Sains Malaysia (USM)":       ["universiti sains malaysia", "usm"],
    "UTAR":                                  ["utar", "universiti tunku abdul rahman"],
    "Taylor's University":                   ["taylor's university", "taylors university"],
    "Sunway University":                     ["sunway university"],
    "APU":                                   ["asia pacific university", " apu "],
    "HELP University":                       ["help university"],
    "INTI International":                    ["inti international", "inti college"],
    "Monash University Malaysia":            ["monash university"],
    "Universiti Islam Antarabangsa (UIAM)":  ["universiti islam antarabangsa", "uiam", "iium"],
}


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_resume(file_bytes: bytes, filename: str) -> dict:
    """
    Parse a resume file and extract structured information.

    Args:
        file_bytes: Raw bytes of the uploaded file
        filename:   Original filename (used to detect PDF vs DOCX)

    Returns:
        dict with keys:
            skills        (list)  : extracted skill names
            education     (str)   : e.g. "Bachelor's Degree"
            field         (str)   : e.g. "Computer Science"
            university    (str)   : e.g. "Multimedia University (MMU)"
            cgpa          (float) : e.g. 3.67
            experience    (int)   : estimated years of experience
            raw_text      (str)   : full extracted text (for debugging)
            confidence    (str)   : "high" / "medium" / "low"
            warnings      (list)  : any extraction warnings
    """
    ext = filename.lower().split(".")[-1]

    # Extract raw text
    raw_text = ""
    warnings = []

    if ext == "pdf":
        raw_text, warn = _extract_pdf(file_bytes)
        warnings.extend(warn)
    elif ext in ("docx", "doc"):
        raw_text, warn = _extract_docx(file_bytes)
        warnings.extend(warn)
    elif ext == "txt":
        raw_text = file_bytes.decode("utf-8", errors="ignore")
    else:
        return {
            "skills": [], "education": "", "field": "",
            "university": "", "cgpa": None, "experience": 0,
            "raw_text": "", "confidence": "low",
            "warnings": [f"Unsupported file type: .{ext}. Use PDF or DOCX."],
        }

    if not raw_text.strip():
        return {
            "skills": [], "education": "", "field": "",
            "university": "", "cgpa": None, "experience": 0,
            "raw_text": "", "confidence": "low",
            "warnings": ["Could not extract text. Try a text-based PDF (not scanned)."],
        }

    text_lower = raw_text.lower()

    # Extract each field
    skills     = _extract_skills(text_lower)
    education  = _extract_education(text_lower)
    field      = _extract_field(text_lower)
    university = _extract_university(text_lower)
    cgpa       = _extract_cgpa(raw_text)
    experience = _extract_experience(text_lower)

    # Confidence score based on how much was extracted
    filled  = sum([bool(skills), bool(education), bool(field),
                   bool(university), cgpa is not None])
    confidence = "high" if filled >= 4 else "medium" if filled >= 2 else "low"

    if not skills:
        warnings.append("No skills detected — please add them manually below.")
    if not education:
        warnings.append("Education level not detected — please select manually.")

    return {
        "skills":     skills,
        "education":  education,
        "field":      field,
        "university": university,
        "cgpa":       cgpa,
        "experience": experience,
        "raw_text":   raw_text[:3000],  # Truncate for display
        "confidence": confidence,
        "warnings":   warnings,
    }


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_skills(text: str) -> list:
    """Match known skills against resume text."""
    found = []
    for skill in KNOWN_SKILLS:
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text):
            found.append(skill)
    return found


def _extract_education(text: str) -> str:
    """Detect highest education level."""
    # Check in order from highest to lowest
    for level, keywords in EDUCATION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return level
    return ""


def _extract_field(text: str) -> str:
    """Detect field of study."""
    for field, keywords in FIELD_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return field
    return ""


def _extract_university(text: str) -> str:
    """Detect Malaysian university name."""
    for uni_name, keywords in MALAYSIAN_UNIVERSITIES.items():
        for kw in keywords:
            if kw in text:
                return uni_name
    return ""


def _extract_cgpa(text: str) -> float:
    """Extract CGPA from resume text."""
    patterns = [
        r'cgpa[:\s]+([0-9]\.[0-9]{1,2})',
        r'gpa[:\s]+([0-9]\.[0-9]{1,2})',
        r'cumulative\s+gpa[:\s]+([0-9]\.[0-9]{1,2})',
        r'([0-9]\.[0-9]{1,2})\s*/\s*4\.0',
        r'([0-9]\.[0-9]{1,2})\s*/\s*4',
        r'pointer[:\s]+([0-9]\.[0-9]{1,2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = float(match.group(1))
            if 0.0 <= val <= 4.0:
                return round(val, 2)
    return None


def _extract_experience(text: str) -> int:
    """Estimate years of experience from resume text."""
    patterns = [
        r'(\d+)\s*\+?\s*years?\s+of\s+experience',
        r'(\d+)\s*\+?\s*years?\s+experience',
        r'experience[:\s]+(\d+)\s*years?',
        r'(\d+)\s*years?\s+working',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            years = int(match.group(1))
            if 0 <= years <= 40:
                return years

    # Estimate from work history: count distinct year mentions
    years_found = re.findall(r'\b(20[0-9]{2}|19[0-9]{2})\b', text)
    if years_found:
        years_int = [int(y) for y in years_found]
        span = max(years_int) - min(years_int)
        return min(span, 20)  # Cap at 20

    return 0


# ── File extractors ───────────────────────────────────────────────────────────

def _extract_pdf(file_bytes: bytes) -> tuple:
    """Extract text from PDF bytes. Returns (text, warnings)."""
    warnings = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text  = "\n".join(pages)
        if not text.strip():
            warnings.append("PDF appears to be scanned/image-based. Text extraction limited.")
        return text, warnings
    except ImportError:
        pass

    # Fallback: pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages)
        return text, warnings
    except ImportError:
        pass

    # Fallback: PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages)
        return text, warnings
    except ImportError:
        warnings.append("PDF library not found. Install: pip install pdfplumber")
        return "", warnings
    except Exception as e:
        warnings.append(f"PDF extraction error: {e}")
        return "", warnings


def _extract_docx(file_bytes: bytes) -> tuple:
    """Extract text from DOCX bytes. Returns (text, warnings)."""
    warnings = []
    try:
        import docx
        doc  = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text, warnings
    except ImportError:
        warnings.append("python-docx not found. Install: pip install python-docx")
        return "", warnings
    except Exception as e:
        warnings.append(f"DOCX extraction error: {e}")
        return "", warnings


# ── Skill matching helpers (used by app.py) ───────────────────────────────────

ALL_TECH_SKILLS = [
    "Python", "R", "SQL", "Java", "JavaScript", "C++", "C#",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Analysis", "Data Visualization", "Statistics",
    "Tableau", "Power BI", "Excel",
    "TensorFlow", "PyTorch", "Scikit-learn",
    "Django", "Flask", "FastAPI", "React", "Node.js",
    "Docker", "AWS", "Azure", "Git",
    "MySQL", "PostgreSQL", "MongoDB", "SQLite",
]

ALL_SOFT_SKILLS = [
    "Leadership", "Communication", "Teamwork", "Problem Solving",
    "Critical Thinking", "Project Management", "Time Management",
    "Presentation", "Negotiation", "Customer Service",
]

def split_skills(skills_list: list) -> tuple:
    """Split extracted skills into tech and soft skill lists."""
    soft_lower = {s.lower() for s in ALL_SOFT_SKILLS}
    tech, soft = [], []
    for s in skills_list:
        if s.lower() in soft_lower:
            soft.append(s)
        else:
            tech.append(s)
    return tech, soft
