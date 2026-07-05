"""
tfidf_matcher.py - TF-IDF Job Matching Engine

Usage:
  from src.data_mining.tfidf_matcher import TFIDFMatcher
  matcher = TFIDFMatcher()
  matcher.build_index()
  candidates = matcher.match(user_profile, top_n=20)
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.database import get_all_jobs, get_connection

INDEX_PATH = "data/processed/tfidf_index.pkl"


def _preprocess_text(text: str) -> str:
    """
    Module-level preprocessor so pickle can always resolve it,
    regardless of whether the module is run as __main__ or imported.
    """
    text = text.lower()
    text = text.replace("ml",  "machine learning")
    text = text.replace("nlp", "natural language processing")
    text = text.replace("dl",  "deep learning")
    text = text.replace("bi",  "business intelligence")
    text = text.replace("oop", "object oriented programming")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TFIDFMatcher:
    """
    TF-IDF job matching engine.

    Key design decisions (from your FYP doc):
      - max_features=500  : balances vocabulary coverage vs. noise
      - ngram_range=(1,2) : catches "machine learning", "data science" phrases
      - min_df=2          : removes very rare/misspelled terms
      - Similarity threshold: 0.15 (returns ~20 candidates on avg)
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.job_matrix  = None        
        self.jobs_df: Optional[pd.DataFrame] = None
        self.is_built = False


    def build_index(self, force_rebuild: bool = False) -> "TFIDFMatcher":
        """
        Build (or load cached) TF-IDF index from the jobs database.

        Args:
            force_rebuild: If True, ignore cache and rebuild from DB.

        Returns:
            self (for method chaining)
        """
        if not force_rebuild and os.path.exists(INDEX_PATH):
            print("📦 Loading cached TF-IDF index...")
            self._load_index()
            print(f"   ✅ Loaded index: {len(self.jobs_df):,} jobs, "
                  f"{len(self.vectorizer.vocabulary_):,} features")
            return self

        print("🔨 Building TF-IDF index from database...")
        self.jobs_df = get_all_jobs()

        if len(self.jobs_df) == 0:
            raise ValueError(
                "❌ No jobs in database!\n"
                "   Run: python src/database.py --load data/raw/your_jobs.csv"
            )

        corpus = self.jobs_df.apply(self._build_job_text, axis=1).tolist()

        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),     # unigrams + bigrams
            min_df=2,               # ignore terms in < 2 documents
            stop_words="english",
            preprocessor=self._preprocess,
        )
        self.job_matrix = self.vectorizer.fit_transform(corpus)

        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        self._save_index()

        try:
            conn = get_connection()
            conn.execute("DELETE FROM tfidf_cache")
            conn.execute(
                "INSERT INTO tfidf_cache (job_count, feature_count) VALUES (?, ?)",
                (len(self.jobs_df), len(self.vectorizer.vocabulary_))
            )
            conn.commit()
            conn.close()
        except Exception:
            pass 

        print(f"   ✅ Index built: {len(self.jobs_df):,} jobs × "
              f"{len(self.vectorizer.vocabulary_):,} features")
        self.is_built = True
        return self

    def match(self, user_profile: dict, top_n: int = 20,
              min_similarity: float = 0.15) -> pd.DataFrame:
        """
        Match a user profile against all jobs.

        Args:
            user_profile: dict with keys:
                - skills        (str): "Python, SQL, Machine Learning"
                - education     (str): "Bachelor of Computer Science"
                - field         (str): "Data Science"
                - experience    (int): years of experience
                - riasec_type   (str): Holland code e.g. "RIA"
                - interests     (str): free text career interests (optional)
            top_n:          Max candidates to return (default 20)
            min_similarity: Minimum cosine similarity (default 0.15)

        Returns:
            DataFrame with columns:
                job_id, title, company, location, skills_required,
                description, tfidf_score, matched_skills, skill_gaps
            Sorted by tfidf_score descending.
        """
        if not self.is_built:
            self.build_index()

        user_text = self._build_user_text(user_profile)
        user_vector = self.vectorizer.transform([user_text])

        scores = cosine_similarity(user_vector, self.job_matrix).flatten()

        valid_mask   = scores >= min_similarity
        valid_scores = scores[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            print(f"   ⚠️  No jobs above threshold {min_similarity}. "
                  f"Returning top {top_n} regardless.")
            top_idx   = np.argsort(scores)[::-1][:top_n]
            top_scores = scores[top_idx]
        else:
            sorted_order = np.argsort(valid_scores)[::-1][:top_n]
            top_idx      = valid_indices[sorted_order]
            top_scores   = valid_scores[sorted_order]

        results = self.jobs_df.iloc[top_idx].copy()
        results["tfidf_score"] = top_scores
        results["job_id"]      = results["id"]

        user_skills = _parse_skills(user_profile.get("skills", ""))
        results["matched_skills"] = results["skills_required"].apply(
            lambda s: _find_matched_skills(user_skills, s)
        )
        results["skill_gaps"] = results["skills_required"].apply(
            lambda s: _find_skill_gaps(user_skills, s)
        )
        results["match_percent"] = results["tfidf_score"].apply(
            lambda s: min(round(s * 100, 1), 99.9)
        )

        return results.reset_index(drop=True)

    def _build_job_text(self, row: pd.Series) -> str:
        """
        Combine job fields into a single text for vectorization.
        Title is repeated 3x to give it higher weight in TF-IDF.
        """
        title  = str(row.get("title", ""))
        skills = str(row.get("skills_required", ""))
        desc   = str(row.get("description", ""))[:500]
        industry = str(row.get("industry", ""))

        return f"{title} {title} {title} {skills} {industry} {desc}"

    def _build_user_text(self, profile: dict) -> str:
        """
        Convert user profile dict to query text.
        Skills are repeated 2x for emphasis.
        """
        skills     = str(profile.get("skills", ""))
        education  = str(profile.get("education", ""))
        field      = str(profile.get("field", ""))
        interests  = str(profile.get("interests", ""))
        riasec     = str(profile.get("riasec_type", ""))

        riasec_keywords = _riasec_to_keywords(riasec)

        return (f"{skills} {skills} "
                f"{education} {field} "
                f"{interests} {riasec_keywords}")

    @staticmethod
    def _preprocess(text: str) -> str:
        """Normalize text before vectorization."""
        return _preprocess_text(text)

    def _save_index(self):
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "job_matrix": self.job_matrix,
                "jobs_df":    self.jobs_df,
            }, f)
        print(f"   💾 Index saved to {INDEX_PATH}")

    def _load_index(self):
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.job_matrix = data["job_matrix"]
        self.jobs_df    = data["jobs_df"]
        self.is_built   = True

    def reindex(self):
        """Force a full rebuild — call this after loading new job data."""
        print("🔄 Forcing TF-IDF reindex...")
        self.build_index(force_rebuild=True)


def _parse_skills(skills_str: str) -> set:
    """Parse comma-separated skills into a lowercase set."""
    if not skills_str:
        return set()
    return {s.strip().lower() for s in skills_str.split(",") if s.strip()}


def _find_matched_skills(user_skills: set, job_skills_str: str) -> list:
    job_skills = _parse_skills(job_skills_str)
    # Case-insensitive comparison
    user_lower = {s.lower() for s in user_skills}
    return sorted([s for s in job_skills if s.lower() in user_lower])

def _find_skill_gaps(user_skills: set, job_skills_str: str) -> list:
    job_skills = _parse_skills(job_skills_str)
    user_lower = {s.lower() for s in user_skills}
    return sorted([s for s in job_skills if s.lower() not in user_lower])


def _riasec_to_keywords(riasec: str) -> str:
    """Map Holland RIASEC codes to career-relevant keywords."""
    mapping = {
        "R": "technical hands-on mechanical engineering hardware",
        "I": "research analytical data science investigative problem solving",
        "A": "creative design arts communication media",
        "S": "social helping education counseling service",
        "E": "leadership management business entrepreneurial sales",
        "C": "detail-oriented organized administrative finance accounting",
    }
    keywords = []
    for code in riasec.upper():
        if code in mapping:
            keywords.append(mapping[code])
    return " ".join(keywords)

if __name__ == "__main__":
    print("=" * 60)
    print("  TF-IDF Matcher — Quick Test")
    print("=" * 60)

    matcher = TFIDFMatcher()
    matcher.build_index()

    test_profile = {
        "skills":       "Python, SQL, Machine Learning, Pandas, Scikit-learn",
        "education":    "Bachelor of Computer Science",
        "field":        "Data Science",
        "experience":   1,
        "riasec_type":  "IRA",
        "interests":    "AI, data analysis, software development",
    }

    print(f"\n👤 Test Profile:")
    for k, v in test_profile.items():
        print(f"   {k:15s}: {v}")

    print("\n🔍 Matching jobs...")
    results = matcher.match(test_profile, top_n=20)

    print(f"\n📋 Top 10 Candidates (from {len(results)} found):\n")
    for i, row in results.head(10).iterrows():
        print(f"  #{i+1:2d} [{row['match_percent']:5.1f}%] "
              f"{row['title'][:45]:45s} | {str(row.get('company',''))[:20]:20s}")
        if row["matched_skills"]:
            print(f"       ✅ Have: {', '.join(row['matched_skills'][:3])}")
        if row["skill_gaps"]:
            print(f"       📚 Need: {', '.join(row['skill_gaps'][:3])}")
        print()

    print(f"✅ TF-IDF matcher working! Found {len(results)} candidates.")
