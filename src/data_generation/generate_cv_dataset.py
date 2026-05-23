"""
generate_cv_dataset.py - Synthetic Malaysian CV Generator (Expanded)
MyCareerGPT | CV Integration Step 1 (Option B) — Lecturer Revision

Usage:
    python src/data_generation/generate_cv_dataset.py
    → Outputs: data/raw/cv_dataset.csv (1000 CVs by default)
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# MALAYSIAN CONTEXT DATA
# ══════════════════════════════════════════════════════════════════════════════

MALAYSIAN_UNIVERSITIES = [
    # ── Public Universities (Research) ───────────────────────────────────────
    "University of Malaya (UM)",
    "Universiti Teknologi Malaysia (UTM)",
    "Universiti Putra Malaysia (UPM)",
    "Universiti Kebangsaan Malaysia (UKM)",
    "Universiti Sains Malaysia (USM)",
    "Universiti Teknologi MARA (UiTM)",
    "Universiti Utara Malaysia (UUM)",
    "Universiti Malaysia Sabah (UMS)",
    "Universiti Malaysia Sarawak (UNIMAS)",
    "Universiti Malaysia Pahang (UMP)",
    "Universiti Sultan Zainal Abidin (UniSZA)",
    "Universiti Malaysia Perlis (UniMAP)",
    "Universiti Malaysia Terengganu (UMT)",
    "Universiti Pendidikan Sultan Idris (UPSI)",
    "Universiti Islam Antarabangsa Malaysia (UIAM/IIUM)",
    "Universiti Tun Hussein Onn Malaysia (UTHM)",
    "Universiti Teknikal Malaysia Melaka (UTeM)",
    "Universiti Malaysia Kelantan (UMK)",
    # ── Private Universities ──────────────────────────────────────────────────
    "Multimedia University (MMU)",
    "Taylor's University",
    "Sunway University",
    "Asia Pacific University (APU)",
    "HELP University",
    "INTI International University",
    "Universiti Tunku Abdul Rahman (UTAR)",
    "Heriot-Watt University Malaysia",
    "University of Nottingham Malaysia",
    "Monash University Malaysia",
    "UCSI University",
    "SEGi University",
    "Lincoln University College",
    "Management & Science University (MSU)",
    "Binary University",
    "Infrastructure University Kuala Lumpur (IUKL)",
    "University of Cyberjaya",
    "Nilai University",
    "Quest International University (QIU)",
    "KDU University College",
    "MAHSA University",
    "Xiamen University Malaysia",
    "Newcastle University Medicine Malaysia (NUMed)",
    "IMU University (International Medical University)",
]

FIELDS_OF_STUDY = [
    # Technology & Engineering
    "Computer Science",
    "Software Engineering",
    "Information Technology",
    "Data Science",
    "Artificial Intelligence",
    "Cybersecurity",
    "Computer Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Biomedical Engineering",
    "Telecommunications Engineering",
    "Electronic Engineering",
    # Business & Finance
    "Business Administration",
    "Finance",
    "Accounting",
    "Economics",
    "Marketing",
    "Human Resource Management",
    "Entrepreneurship",
    "Supply Chain Management",
    "International Business",
    # Science & Mathematics
    "Mathematics",
    "Statistics",
    "Physics",
    "Chemistry",
    "Biotechnology",
    "Microbiology",
    "Environmental Science",
    # Health & Social Sciences
    "Psychology",
    "Nursing",
    "Public Health",
    "Social Work",
    "Education",
    "Communication Studies",
    # Creative & Design
    "Graphic Design",
    "Multimedia Design",
    "Architecture",
    "Industrial Design",
    "Mass Communication",
]

MALAYSIAN_LOCATIONS = [
    # Klang Valley / Central
    "Kuala Lumpur",
    "Petaling Jaya",
    "Subang Jaya",
    "Shah Alam",
    "Klang",
    "Cyberjaya",
    "Putrajaya",
    "Sepang",
    # Selangor
    "Rawang",
    "Kajang",
    "Puchong",
    "Ampang",
    "Cheras",
    # Northern Region
    "Penang",
    "Alor Setar",
    "Sungai Petani",
    "Ipoh",
    "Taiping",
    # Southern Region
    "Johor Bahru",
    "Iskandar Puteri",
    "Muar",
    "Batu Pahat",
    "Melaka",
    "Seremban",
    # East Coast
    "Kuantan",
    "Kuala Terengganu",
    "Kota Bharu",
    # East Malaysia
    "Kota Kinabalu",
    "Sandakan",
    "Tawau",
    "Kuching",
    "Miri",
    "Sibu",
]

SOFT_SKILLS = [
    "Communication",
    "Teamwork",
    "Problem Solving",
    "Critical Thinking",
    "Time Management",
    "Leadership",
    "Adaptability",
    "Creativity",
    "Attention to Detail",
    "Presentation",
    "Negotiation",
    "Customer Service",
    "Stakeholder Management",
    "Conflict Resolution",
    "Emotional Intelligence",
    "Decision Making",
    "Work Ethic",
    "Analytical Thinking",
]

MALAYSIAN_NAMES = [
    # Malay names
    "Ahmad Faiz", "Nur Aisyah", "Muhammad Haziq", "Siti Zulaikha",
    "Mohd Hafiz", "Nurul Hidayah", "Amirul Haikal", "Fatimah Zahrah",
    "Rashdan Rosli", "Noor Farhana", "Khairul Anwar", "Siti Norbahyah",
    "Azrul Hafifi", "Nurul Ain", "Izzuddin Zulkifli", "Hana Sofea",
    "Faizal Harun", "Zulaikha Zainal", "Hafizuddin Malik", "Liyana Hashim",
    "Amzar Roslan", "Syafiqah Ismail", "Ridhwan Borhan", "Irdina Razali",
    "Arif Hakimi", "Qistina Shafiq", "Hazwan Azmi", "Nabilah Zaidin",
    "Farhan Zulkifle", "Nur Syafawati",
    # Chinese names
    "Lim Wei Jie", "Tan Chee Keong", "Chong Kai Zhen", "Wong Li Ying",
    "Lee Kok Wai", "Melissa Ong", "Yap Jia Hui", "Ng Boon Keat",
    "Chin Xin Yi", "Chua Wei Lin", "Teh Zhi Xiang", "Koh Mei Ling",
    "Goh Jian Wei", "Pua Yee Ling", "Ooi Chun Keat", "Ho Shu Fen",
    "Low Wai Khim", "Sim Yin Yin", "Yeoh Kah Mun", "Leong Chee Wai",
    # Indian names
    "Priya Nair", "Kavitha Raj", "Surinder Singh", "Anitha Pillai",
    "Rajan Muthu", "Deepa Krishnan", "Vignesh Kumar", "Sharmila Devi",
    "Arjun Subramaniam", "Thivya Nadarajan", "Mohan Pillai", "Rekha Gopal",
    "Kartik Ramachandran", "Lavanya Chandran", "Vikram Selvam", "Geetha Munusamy",
    # Other / Mixed
    "James Ambrose", "Rachel Tan", "David Raj", "Sharon Lim",
]

# ══════════════════════════════════════════════════════════════════════════════
# JOB PROFILES — 20 categories (was 10)
# ══════════════════════════════════════════════════════════════════════════════

JOB_PROFILES = {
    # ── Technology ──────────────────────────────────────────────────────────
    "Data Analyst": {
        "core_skills":       ["Python", "SQL", "Excel", "Data Analysis", "Statistics"],
        "bonus_skills":      ["Tableau", "Power BI", "R", "Pandas", "NumPy", "Looker"],
        "companies":         ["Maybank", "CIMB", "RHB Bank", "Petronas", "Axiata",
                              "Maxis", "AirAsia", "Lazada", "Shopee", "Digi", "TIME dotCom"],
        "salary_range":      (3500, 6500),
        "preferred_fields":  ["Data Science", "Statistics", "Mathematics", "Computer Science"],
    },
    "Software Engineer": {
        "core_skills":       ["Python", "Java", "Git", "SQL", "OOP"],
        "bonus_skills":      ["JavaScript", "React", "Docker", "AWS", "Microservices", "Kotlin"],
        "companies":         ["Grab", "Shopee", "Sea Group", "Fusionex", "Revenue Monster",
                              "Soft Space", "iPay88", "MOL", "Agmo Studio", "Netccentric"],
        "salary_range":      (4000, 9000),
        "preferred_fields":  ["Software Engineering", "Computer Science", "Information Technology"],
    },
    "Machine Learning Engineer": {
        "core_skills":       ["Python", "Machine Learning", "Scikit-learn", "SQL", "Statistics"],
        "bonus_skills":      ["TensorFlow", "PyTorch", "Deep Learning", "NLP", "MLOps", "Spark"],
        "companies":         ["Grab", "Axiata", "Telekom Malaysia", "Intel", "Fusionex",
                              "DataSpark", "Neuron", "Sedania", "MDEC"],
        "salary_range":      (5500, 12000),
        "preferred_fields":  ["Data Science", "Computer Science", "Mathematics", "Artificial Intelligence"],
    },
    "Data Scientist": {
        "core_skills":       ["Python", "R", "Machine Learning", "Statistics", "SQL"],
        "bonus_skills":      ["Deep Learning", "NLP", "TensorFlow", "Data Visualization", "Spark", "Hadoop"],
        "companies":         ["Axiata", "Grab", "Petronas", "Telekom Malaysia", "MDEC",
                              "DataSpark", "Fusionex", "Setel", "Carsome"],
        "salary_range":      (5500, 13000),
        "preferred_fields":  ["Data Science", "Statistics", "Computer Science", "Mathematics", "Artificial Intelligence"],
    },
    "Web Developer": {
        "core_skills":       ["HTML", "CSS", "JavaScript", "Git", "SQL"],
        "bonus_skills":      ["React", "Vue", "Node.js", "PHP", "Laravel", "Django", "TypeScript"],
        "companies":         ["Exabytes", "WebFX", "Naga DDB", "Geometry Global",
                              "Astro", "Media Prima", "REV Asia", "iPrice Group", "PropertyGuru"],
        "salary_range":      (3000, 7000),
        "preferred_fields":  ["Software Engineering", "Information Technology", "Computer Science", "Multimedia Design"],
    },
    "DevOps Engineer": {
        "core_skills":       ["Linux", "Docker", "Git", "Python", "CI/CD"],
        "bonus_skills":      ["Kubernetes", "AWS", "Azure", "Terraform", "Jenkins", "Ansible"],
        "companies":         ["Grab", "Shopee", "Telekom Malaysia", "Maxis", "Celcom",
                              "Oracle Malaysia", "IBM Malaysia", "Accenture"],
        "salary_range":      (5500, 12000),
        "preferred_fields":  ["Computer Science", "Information Technology", "Software Engineering"],
    },
    "Cybersecurity Analyst": {
        "core_skills":       ["Network Security", "Linux", "Python", "SIEM", "Vulnerability Assessment"],
        "bonus_skills":      ["Penetration Testing", "Forensics", "CISSP", "CEH", "Splunk", "Firewall"],
        "companies":         ["CyberSecurity Malaysia", "Telekom Malaysia", "Maxis", "Bank Negara Malaysia",
                              "PETRONAS", "Palo Alto Networks MY", "IBM Security"],
        "salary_range":      (4500, 11000),
        "preferred_fields":  ["Cybersecurity", "Computer Science", "Information Technology", "Computer Engineering"],
    },
    "Mobile Developer": {
        "core_skills":       ["Java", "Kotlin", "Swift", "Git", "REST APIs"],
        "bonus_skills":      ["Flutter", "React Native", "Firebase", "Android Studio", "Xcode"],
        "companies":         ["Grab", "AirAsia", "Setel", "BigPay", "Carsome",
                              "PropertyGuru", "iMoney", "Agmo Studio"],
        "salary_range":      (4000, 9000),
        "preferred_fields":  ["Software Engineering", "Computer Science", "Information Technology"],
    },
    "Cloud Engineer": {
        "core_skills":       ["AWS", "Azure", "GCP", "Linux", "Python"],
        "bonus_skills":      ["Terraform", "Kubernetes", "Docker", "Serverless", "CloudFormation"],
        "companies":         ["Telekom Malaysia", "Maxis", "IBM Malaysia", "Accenture",
                              "TM ONE", "DXC Technology", "NTT Malaysia"],
        "salary_range":      (5000, 13000),
        "preferred_fields":  ["Computer Science", "Information Technology", "Software Engineering", "Computer Engineering"],
    },
    "UI/UX Designer": {
        "core_skills":       ["Figma", "User Research", "Wireframing", "Prototyping", "Adobe XD"],
        "bonus_skills":      ["HTML", "CSS", "Sketch", "InVision", "Usability Testing", "Photoshop"],
        "companies":         ["Grab", "Lazada", "AirAsia", "iPrice Group", "PropertyGuru",
                              "Fave", "Naga DDB", "Wunderman Thompson"],
        "salary_range":      (3500, 8000),
        "preferred_fields":  ["Multimedia Design", "Graphic Design", "Computer Science", "Communication Studies"],
    },
    # ── Business & Finance ───────────────────────────────────────────────────
    "Business Analyst": {
        "core_skills":       ["Excel", "SQL", "Communication", "Problem Solving", "Presentation"],
        "bonus_skills":      ["Power BI", "Tableau", "Project Management", "JIRA", "Agile", "Visio"],
        "companies":         ["CIMB", "Maybank", "Deloitte", "KPMG", "PwC",
                              "Accenture", "IBM Malaysia", "Sapura", "Maxis"],
        "salary_range":      (3500, 8000),
        "preferred_fields":  ["Business Administration", "Finance", "Computer Science", "Economics"],
    },
    "Financial Analyst": {
        "core_skills":       ["Excel", "Financial Modeling", "Accounting", "Statistics", "SQL"],
        "bonus_skills":      ["Bloomberg", "Power BI", "Python", "VBA", "CFA", "SAP"],
        "companies":         ["Maybank Investment", "CIMB Investment", "AmBank",
                              "Hong Leong Bank", "Public Bank", "RHB Capital", "Bank Islam",
                              "Affin Bank", "Alliance Bank"],
        "salary_range":      (4000, 9000),
        "preferred_fields":  ["Finance", "Accounting", "Mathematics", "Economics"],
    },
    "Accountant": {
        "core_skills":       ["Accounting", "Excel", "Financial Reporting", "Tax", "Audit"],
        "bonus_skills":      ["SAP", "MYOB", "AutoCount", "SQL Accounting", "ACCA", "ICMA"],
        "companies":         ["PwC Malaysia", "Deloitte Malaysia", "EY Malaysia", "KPMG Malaysia",
                              "BDO Malaysia", "Grant Thornton", "Crowe Malaysia"],
        "salary_range":      (3000, 7500),
        "preferred_fields":  ["Accounting", "Finance", "Business Administration"],
    },
    "Human Resource Executive": {
        "core_skills":       ["Recruitment", "Payroll", "Employment Law", "Communication", "Excel"],
        "bonus_skills":      ["HRMS", "Training & Development", "Performance Management", "SAP HR", "SHRM"],
        "companies":         ["Maybank", "CIMB", "Petronas", "Maxis", "AirAsia",
                              "Top Glove", "IHH Healthcare", "Sime Darby", "Public Bank"],
        "salary_range":      (2800, 6000),
        "preferred_fields":  ["Human Resource Management", "Business Administration", "Psychology"],
    },
    "Marketing Executive": {
        "core_skills":       ["Digital Marketing", "Social Media", "Content Creation", "SEO", "Google Analytics"],
        "bonus_skills":      ["Facebook Ads", "Google Ads", "Copywriting", "Canva", "Email Marketing", "CRM"],
        "companies":         ["Lazada", "Shopee", "AirAsia", "Astro", "Media Prima",
                              "Naga DDB", "Leo Burnett", "IPG Mediabrands", "Mindshare"],
        "salary_range":      (2800, 6500),
        "preferred_fields":  ["Marketing", "Mass Communication", "Business Administration", "Communication Studies"],
    },
    "Project Manager": {
        "core_skills":       ["Project Management", "Communication", "Leadership", "Risk Management", "Agile"],
        "bonus_skills":      ["PMP", "JIRA", "MS Project", "Scrum", "Stakeholder Management", "PRINCE2"],
        "companies":         ["Petronas", "Sapura Energy", "IJM Corporation", "Gamuda",
                              "KLCC", "Accenture", "IBM Malaysia", "Deloitte", "YTL Corporation"],
        "salary_range":      (5000, 14000),
        "preferred_fields":  ["Business Administration", "Engineering", "Information Technology", "Computer Science"],
    },
    # ── Engineering ──────────────────────────────────────────────────────────
    "Network Engineer": {
        "core_skills":       ["Networking", "Linux", "Cisco", "TCP/IP", "Security"],
        "bonus_skills":      ["AWS", "Azure", "Python", "CCNA", "Firewall", "BGP", "SD-WAN"],
        "companies":         ["Telekom Malaysia", "Maxis", "Celcom", "Digi",
                              "TIME dotCom", "U Mobile", "Ericsson Malaysia", "Huawei Malaysia"],
        "salary_range":      (3500, 8000),
        "preferred_fields":  ["Information Technology", "Computer Science", "Electrical Engineering", "Telecommunications Engineering"],
    },
    "Electrical Engineer": {
        "core_skills":       ["AutoCAD", "Circuit Design", "PLC", "Electrical Standards", "Project Management"],
        "bonus_skills":      ["SCADA", "HMI", "Power Systems", "Solar", "BIM", "Revit"],
        "companies":         ["Tenaga Nasional (TNB)", "Petronas", "Sapura Energy", "Gamuda",
                              "IJM Engineering", "SESCO", "Sarawak Energy", "Alstom Malaysia"],
        "salary_range":      (3500, 8500),
        "preferred_fields":  ["Electrical Engineering", "Electronic Engineering", "Mechatronics"],
    },
    "Mechanical Engineer": {
        "core_skills":       ["AutoCAD", "SolidWorks", "Thermodynamics", "Fluid Mechanics", "Manufacturing"],
        "bonus_skills":      ["ANSYS", "CATIA", "Lean Manufacturing", "Six Sigma", "ISO 9001", "CAM"],
        "companies":         ["Proton", "Perodua", "Petronas", "Sapura Energy", "Sime Darby Industrial",
                              "Panasonic Malaysia", "Sharp Malaysia", "Intel Malaysia"],
        "salary_range":      (3200, 8000),
        "preferred_fields":  ["Mechanical Engineering", "Biomedical Engineering", "Chemical Engineering"],
    },
    # ── Healthcare & Science ─────────────────────────────────────────────────
    "Data Engineer": {
        "core_skills":       ["Python", "SQL", "ETL", "Data Warehousing", "Apache Spark"],
        "bonus_skills":      ["Kafka", "Airflow", "dbt", "Redshift", "Snowflake", "BigQuery"],
        "companies":         ["Grab", "Shopee", "Axiata", "Telekom Malaysia", "MDEC",
                              "Setel", "Carsome", "PropertyGuru", "iPrice Group"],
        "salary_range":      (4500, 11000),
        "preferred_fields":  ["Computer Science", "Data Science", "Information Technology", "Statistics"],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_cv_dataset(
    n: int = 1500,
    output_path: str = "data/raw/cv_dataset.csv"
) -> pd.DataFrame:
    """
    Generate n synthetic Malaysian CVs with verified job placements.

    Each CV has a ground truth label (actual_job_obtained) used for training.
    Covers 20 job categories across all major Malaysian states.

    Args:
        n:           Number of CVs to generate (lecturer requires > 500; default 1500)
        output_path: Where to save the CSV

    Returns:
        DataFrame with CV data
    """
    job_titles = list(JOB_PROFILES.keys())
    rows = []

    for i in range(n):
        # Pick a job this person obtained (ground truth label)
        job_title = random.choice(job_titles)
        job_info  = JOB_PROFILES[job_title]

        # ── Education ────────────────────────────────────────────────────────
        if random.random() < 0.72:   # 72 % have a matching field
            field = random.choice(job_info["preferred_fields"])
        else:
            field = random.choice(FIELDS_OF_STUDY)

        university = random.choice(MALAYSIAN_UNIVERSITIES)

        # Weight: most CVs are Bachelor's; ~15 % Master's; ~5 % Diploma
        education = random.choices(
            ["Bachelor's Degree", "Master's Degree", "Diploma"],
            weights=[0.80, 0.15, 0.05],
            k=1
        )[0]

        # ── CGPA ─────────────────────────────────────────────────────────────
        # Higher-bar roles skew CGPA up
        if job_title in ["Machine Learning Engineer", "Data Scientist", "Cloud Engineer"]:
            cgpa = round(random.uniform(3.3, 4.0), 2)
        elif job_title in ["Accountant", "Financial Analyst", "Cybersecurity Analyst"]:
            cgpa = round(random.uniform(3.0, 4.0), 2)
        else:
            cgpa = round(random.uniform(2.5, 4.0), 2)

        # ── Experience ───────────────────────────────────────────────────────
        if education == "Master's Degree":
            experience_years = random.randint(2, 10)
        elif education == "Diploma":
            experience_years = random.randint(0, 5)
        else:
            experience_years = random.randint(0, 8)

        # ── Skills ───────────────────────────────────────────────────────────
        n_core  = random.randint(3, len(job_info["core_skills"]))
        n_bonus = random.randint(0, min(4, len(job_info["bonus_skills"])))
        n_soft  = random.randint(2, 5)

        core_picked  = random.sample(job_info["core_skills"],  n_core)
        bonus_picked = random.sample(job_info["bonus_skills"], n_bonus)
        soft_picked  = random.sample(SOFT_SKILLS, n_soft)

        all_skills = core_picked + bonus_picked + soft_picked
        random.shuffle(all_skills)

        # ── Company, Salary, Location ─────────────────────────────────────────
        company = random.choice(job_info["companies"])
        salary  = random.randint(*job_info["salary_range"])

        # Location weighted: ~55% Klang Valley, rest spread across Malaysia
        kv = ["Kuala Lumpur", "Petaling Jaya", "Subang Jaya", "Shah Alam",
              "Cyberjaya", "Klang", "Puchong", "Cheras", "Ampang"]
        non_kv = [loc for loc in MALAYSIAN_LOCATIONS if loc not in kv]
        location = random.choices(
            [random.choice(kv), random.choice(non_kv)],
            weights=[0.55, 0.45],
            k=1
        )[0]

        # ── Name ─────────────────────────────────────────────────────────────
        name = random.choice(MALAYSIAN_NAMES) + f" {chr(65 + (i % 26))}"

        rows.append({
            "name":               name,
            "education":          education,
            "field":              field,
            "university":         university,
            "cgpa":               cgpa,
            "skills":             ", ".join(all_skills),
            "experience_years":   experience_years,
            "actual_job_obtained": job_title,   # ← GROUND TRUTH LABEL
            "company":            company,
            "salary":             salary,
            "location":           location,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n✅  Generated {n} Malaysian CVs → {output_path}")
    print(f"\n📊  Job distribution ({len(job_titles)} categories):")
    for job, count in df["actual_job_obtained"].value_counts().items():
        print(f"    {job:40s}: {count:4d}")
    print(f"\n🏙️   Location distribution (top 10):")
    for loc, count in df["location"].value_counts().head(10).items():
        print(f"    {loc:30s}: {count:4d}")
    print(f"\n🎓  University distribution (top 10):")
    for uni, count in df["university"].value_counts().head(10).items():
        print(f"    {uni:45s}: {count:4d}")
    print(f"\n📁  Columns: {list(df.columns)}")
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Malaysian CV dataset")
    parser.add_argument("--n",    type=int, default=1500,
                        help="Number of CVs to generate (default: 1500)")
    parser.add_argument("--out",  type=str, default="data/raw/cv_dataset.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    df = generate_cv_dataset(args.n, args.out)
    print(f"\n✅  CV dataset ready at: {args.out}")
    print(f"    Use this to train CVEnhancedMatcher and SkillGapPredictor")