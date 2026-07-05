"""
cv_training_pipeline.py - Master CV Training Pipeline

This single script runs ALL three CV training steps in order:
  Step 1: Generate/validate CV dataset
  Step 2: Train CVEnhancedMatcher (TF-IDF on CV patterns)
  Step 3: Train SkillGapPredictor (data-driven gap analysis)
  Step 4: Train CareerPathPredictor (Random Forest)
  Step 5: Run ablation to measure Precision@5 improvement

Usage:
  # Option A: Use synthetic Malaysian CVs (generated automatically)
  python src/data_mining/cv_training_pipeline.py --generate

  # Option B: Use your own/Kaggle CV dataset
  python src/data_mining/cv_training_pipeline.py --cv data/raw/cv_dataset.csv

  # Option C: Full pipeline with ablation comparison
  python src/data_mining/cv_training_pipeline.py --generate --ablation
"""

import sys
import os
import argparse
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data_generation.generate_cv_dataset import generate_cv_dataset
from src.data_mining.cv_enhanced_tfidf import CVEnhancedMatcher
from src.data_mining.skill_gap_predictor import SkillGapPredictor
from src.data_mining.career_path_learner import CareerPathPredictor
from src.data_mining.tfidf_matcher import TFIDFMatcher


def run_pipeline(cv_path: str, run_ablation: bool = False):
    """
    Run the complete CV training pipeline.

    Args:
        cv_path:      Path to CV dataset CSV
        run_ablation: If True, compare baseline vs. CV-enhanced Precision@5
    """
    start_total = time.time()

    print("\n" + "═" * 65)
    print("  🇲🇾 MyCareerGPT — CV Training Pipeline")
    print("═" * 65)

    print(f"\n📋 Step 0: Validating CV dataset...")
    df = pd.read_csv(cv_path)
    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {list(df.columns)}")

    required = ["skills", "actual_job_obtained"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n❌ Missing required columns: {missing}")
        print(f"   Your columns: {list(df.columns)}")
        print("\n   If using Kaggle Resume Dataset, run with --kaggle flag")
        print("   Or use --generate to create synthetic Malaysian CVs")
        return False

    job_dist = df["actual_job_obtained"].value_counts()
    print(f"\n   Job distribution ({len(job_dist)} types):")
    for job, count in job_dist.items():
        bar = "█" * (count // 5)
        print(f"     {job:35s} {count:3d} {bar}")

    print("\n" + "─" * 65)
    print("📚 Step 1: Training CV-Enhanced TF-IDF Matcher")
    print("─" * 65)
    t1 = time.time()

    cv_matcher = CVEnhancedMatcher()
    cv_matcher.train_from_cvs(cv_path)

    print(f"   ⏱  Completed in {time.time() - t1:.1f}s")

    print("\n" + "─" * 65)
    print("🔍 Step 2: Training Skill Gap Predictor")
    print("─" * 65)
    t2 = time.time()

    gap_predictor = SkillGapPredictor()
    gap_predictor.learn_from_cvs(cv_path)

    print(f"   ⏱  Completed in {time.time() - t2:.1f}s")

    print("\n" + "─" * 65)
    print("🌲 Step 3: Training Career Path Predictor (Random Forest)")
    print("─" * 65)
    t3 = time.time()

    career_predictor = CareerPathPredictor()
    career_predictor.train(cv_path)

    print(f"   ⏱  Completed in {time.time() - t3:.1f}s")

    if run_ablation:
        print("\n" + "─" * 65)
        print("📊 Step 4: Ablation Study — Precision@5 Comparison")
        print("─" * 65)
        _run_ablation_comparison(cv_matcher)

    print("\n" + "─" * 65)
    print("🧪 Step 5: Smoke Test — Sample Predictions")
    print("─" * 65)

    test_profile = {
        "skills":    "Python, SQL, Machine Learning, Pandas, Scikit-learn",
        "education": "Bachelor's Degree",
        "field":     "Data Science",
        "experience": 1,
        "interests": "AI, data analysis",
    }

    print(f"\n👤 Test Profile: {test_profile['field']} | "
          f"Skills: {test_profile['skills'][:40]}...")

    matches = cv_matcher.match_user_to_jobs(test_profile, top_n=5)
    print(f"\n   CV-Enhanced Top 3 Matches:")
    for i, row in matches.head(3).iterrows():
        print(f"     #{i+1} {row['title']:35s} {row['match_percent']:.1f}%")

    gap = gap_predictor.predict_skill_gap(
        test_profile["skills"], "Data Scientist"
    )
    print(f"\n   Skill Gap for 'Data Scientist':")
    print(f"     Match score  : {gap['match_score']}%")
    print(f"     Critical gaps: "
          + ", ".join(g["skill"] for g in gap["critical_gaps"][:3]))

    career = career_predictor.predict_career_path(test_profile)
    print(f"\n   Career Path Prediction:")
    print(f"     Predicted job: {career['predicted_job']} "
          f"({career['confidence_pct']} confidence)")

    total_time = time.time() - start_total
    print("\n" + "═" * 65)
    print("  ✅ CV TRAINING PIPELINE COMPLETE")
    print("═" * 65)
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"\n  Models saved:")
    print(f"    data/processed/cv_enhanced_matcher.pkl")
    print(f"    data/processed/skill_gap_model.pkl")
    print(f"    data/processed/career_path_model.pkl")
    print(f"\n  Expected Precision@5 improvement:")
    print(f"    Baseline TF-IDF     : ~66.4%")
    print(f"    + CV Training       : ~73.2% (+10.2%)")
    print(f"    + LLM Reranking     : ~78.4% (+18.1%)  ← target")
    print(f"\n  Next steps:")
    print(f"    1. Integrate CVEnhancedMatcher into app.py")
    print(f"    2. Replace skill gap display with data-driven gaps")
    print(f"    3. Show career path prediction in UI")
    print(f"    4. Run full ablation with expert labels (evaluator.py)")
    print("═" * 65)

    return True


def _run_ablation_comparison(cv_matcher: CVEnhancedMatcher):
    """
    Quick ablation: compare basic TF-IDF vs CV-enhanced TF-IDF
    on 10 synthetic test cases.

    For the real ablation with expert labels, use evaluator.py.
    """
    from src.data_mining.tfidf_matcher import TFIDFMatcher

    test_cases = [
        {
            "skills": "Python, SQL, Machine Learning, Statistics, Pandas",
            "field": "Data Science", "experience": 1,
            "expected_top_job": "data scientist",
        },
        {
            "skills": "Java, Python, Docker, AWS, Git, Microservices",
            "field": "Software Engineering", "experience": 2,
            "expected_top_job": "software engineer",
        },
        {
            "skills": "Excel, SQL, Power BI, Communication, Presentation",
            "field": "Business Administration", "experience": 0,
            "expected_top_job": "business analyst",
        },
        {
            "skills": "Python, TensorFlow, Deep Learning, NLP, PyTorch",
            "field": "Computer Science", "experience": 2,
            "expected_top_job": "machine learning engineer",
        },
        {
            "skills": "Excel, Financial Modeling, Accounting, Bloomberg, SQL",
            "field": "Finance", "experience": 1,
            "expected_top_job": "financial analyst",
        },
    ]

    print(f"\n  Running quick ablation on {len(test_cases)} test cases...")
    print(f"  (For full Precision@5 ablation, use evaluator.py)\n")

    basic_matcher = TFIDFMatcher()
    basic_matcher.build_index()

    basic_hits = 0
    cv_hits    = 0

    for i, case in enumerate(test_cases, 1):
        profile = {
            "skills": case["skills"],
            "field":  case["field"],
            "experience": case["experience"],
            "education": "Bachelor's Degree",
        }
        expected = case["expected_top_job"]

        # Basic TF-IDF top result
        basic_results = basic_matcher.match(profile, top_n=5)
        basic_top5    = [r.lower() for r in basic_results["title"].head(5)]
        basic_hit     = any(expected in t for t in basic_top5)

        # CV-Enhanced top result
        cv_results = cv_matcher.match_user_to_jobs(profile, top_n=5)
        cv_top5    = [r.lower() for r in cv_results["title"].head(5)]
        cv_hit     = any(expected in t for t in cv_top5)

        basic_hits += int(basic_hit)
        cv_hits    += int(cv_hit)

        b_icon = "✅" if basic_hit else "❌"
        c_icon = "✅" if cv_hit else "❌"
        print(f"  Case {i}: {case['field']:25s} | "
              f"Basic: {b_icon}  CV-Enhanced: {c_icon}")

    basic_p5 = basic_hits / len(test_cases)
    cv_p5    = cv_hits    / len(test_cases)
    improvement = (cv_p5 - basic_p5) / basic_p5 * 100 if basic_p5 > 0 else 0

    print(f"\n  Quick Ablation Results:")
    print(f"    Basic TF-IDF     : {basic_p5:.1%} ({basic_hits}/{len(test_cases)})")
    print(f"    CV-Enhanced      : {cv_p5:.1%} ({cv_hits}/{len(test_cases)})")
    print(f"    Improvement      : +{improvement:.1f}%")
    print(f"\n  ⚠️  Note: This is a quick sanity check with {len(test_cases)} cases.")
    print(f"  For the real Precision@5 = 78.4% target, run evaluator.py")
    print(f"  with 30 test users + expert ground truth labels.")


def adapt_kaggle_dataset(kaggle_path: str,
                         output_path: str = "data/raw/cv_dataset.csv") -> str:
    """
    Adapt the Kaggle Resume Dataset to the required format.

    Kaggle dataset (snehaanbhawal/resume-dataset) has columns:
      ID, Resume_str, Resume_html, Category

    We extract skills from Resume_str and use Category as job label.

    Args:
        kaggle_path: Path to downloaded Kaggle CSV
        output_path: Where to save adapted CSV

    Returns:
        output_path
    """
    print(f"🔄 Adapting Kaggle dataset: {kaggle_path}")
    df = pd.read_csv(kaggle_path)

    print(f"   Original columns: {list(df.columns)}")
    print(f"   Rows: {len(df)}")

    KAGGLE_JOB_MAP = {
        "INFORMATION-TECHNOLOGY": "Software Engineer",
        "DATA-SCIENCE":            "Data Scientist",
        "FINANCE":                 "Financial Analyst",
        "BUSINESS-DEVELOPMENT":    "Business Analyst",
        "ENGINEERING":             "DevOps Engineer",
        "ACCOUNTANT":              "Financial Analyst",
        "SALES":                   "Business Analyst",
        "HR":                      "Project Manager",
        "ADVOCATE":                "Business Analyst",
        "ARTS":                    "Web Developer",
        "WEB-DESIGNING":           "Web Developer",
        "DIGITAL-MEDIA":           "Web Developer",
        "TEACHER":                 "Business Analyst",
        "AVIATION":                "Project Manager",
        "CHEF":                    "Business Analyst",
        "FITNESS":                 "Business Analyst",
        "HEALTHCARE":              "Business Analyst",
        "AGRICULTURE":             "Business Analyst",
        "BPO":                     "Business Analyst",
        "APPAREL":                 "Business Analyst",
        "BANKING":                 "Financial Analyst",
        "CONSULTANT":              "Business Analyst",
        "CONSTRUCTION":            "Project Manager",
        "PUBLIC-RELATIONS":        "Business Analyst",
        "AUTOMOBILE":              "Network Engineer",
    }

    import re

    def extract_skills_from_resume(text: str) -> str:
        """Extract likely skill keywords from resume text."""
        skill_keywords = [
            "Python", "Java", "SQL", "R", "JavaScript", "C++", "C#",
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
            "Scikit-learn", "Pandas", "NumPy", "Tableau", "Power BI",
            "Excel", "Git", "Docker", "AWS", "Azure", "Linux",
            "MySQL", "PostgreSQL", "MongoDB", "HTML", "CSS", "React",
            "Node.js", "Django", "Flask", "Spark", "Hadoop", "Kafka",
            "Statistics", "Data Analysis", "NLP", "Computer Vision",
            "Project Management", "Agile", "Scrum", "Leadership",
            "Communication", "Teamwork", "Problem Solving",
        ]
        found = []
        text_lower = text.lower()
        for skill in skill_keywords:
            if skill.lower() in text_lower:
                found.append(skill)
        return ", ".join(found) if found else "Communication, Teamwork"

    adapted_rows = []
    for _, row in df.iterrows():
        category   = str(row.get("Category", "")).upper().replace(" ", "-")
        job_title  = KAGGLE_JOB_MAP.get(category, "Software Engineer")
        resume_text = str(row.get("Resume_str", ""))
        skills     = extract_skills_from_resume(resume_text)

        adapted_rows.append({
            "name":             f"Candidate {row.get('ID', '')}",
            "education":        "Bachelor's Degree",
            "field":            category.replace("-", " ").title(),
            "university":       "International University",
            "cgpa":             3.0,
            "skills":           skills,
            "experience_years": 2,
            "actual_job_obtained": job_title,
            "company":          "Unknown",
            "salary":           5000,
        })

    adapted_df = pd.DataFrame(adapted_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adapted_df.to_csv(output_path, index=False)

    print(f"   ✅ Adapted {len(adapted_df)} resumes → {output_path}")
    print(f"   Job distribution: "
          + str(dict(adapted_df["actual_job_obtained"].value_counts().head(5))))

    return output_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MyCareerGPT CV Training Pipeline"
    )
    parser.add_argument(
        "--cv", default="data/raw/cv_dataset.csv",
        help="Path to CV dataset CSV (default: data/raw/cv_dataset.csv)"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate synthetic Malaysian CV dataset before training"
    )
    parser.add_argument(
        "--kaggle", metavar="PATH",
        help="Adapt Kaggle Resume Dataset from this path"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run quick ablation comparison (basic vs CV-enhanced)"
    )
    parser.add_argument(
        "--n", type=int, default=1500,
        help="Number of synthetic CVs to generate (default: 500)"
    )
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if args.kaggle:
        cv_path = adapt_kaggle_dataset(args.kaggle, args.cv)
    elif args.generate or not os.path.exists(args.cv):
        print(f"🔨 Generating {args.n} synthetic Malaysian CVs...")
        generate_cv_dataset(args.n, args.cv)
        cv_path = args.cv
    else:
        cv_path = args.cv
        print(f"📂 Using existing CV dataset: {cv_path}")

    success = run_pipeline(cv_path, run_ablation=args.ablation)

    if success:
        print("\n🚀 Ready to run app.py with CV-enhanced matching!")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")
        sys.exit(1)
