"""
skill_gap_predictor.py - Data-Driven Skill Gap Analysis

Usage:
    from src.data_mining.skill_gap_predictor import SkillGapPredictor
    predictor = SkillGapPredictor()
    predictor.learn_from_cvs('data/raw/cv_dataset.csv')
    gaps = predictor.predict_skill_gap('Python, SQL', 'Data Analyst')
"""

import os
import sys
import pickle
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

SKILL_GAP_MODEL_PATH = "data/processed/skill_gap_model.pkl"


class SkillGapPredictor:
    """
    Learns which skills successful candidates had for each job type.

    For each job title in the CV dataset, tracks:
      - Which skills appeared most frequently
      - What % of successful candidates had each skill
      - Which skills are "critical" (>50% of placed candidates had them)

    This gives data-driven skill gaps instead of rule-based guessing.
    """

    def __init__(self):
        self.skill_patterns: dict = {}
        self.skill_frequencies: dict = {}
        self.job_cv_counts: dict = {}
        self.is_trained = False


    def learn_from_cvs(self, cv_dataset_path: str) -> "SkillGapPredictor":
        """
        Learn skill patterns for each job type from the CV dataset.

        Args:
            cv_dataset_path: Path to CV CSV with columns:
                skills, actual_job_obtained

        Returns:
            self
        """
        print(f"📚 Learning skill patterns from: {cv_dataset_path}")
        cvs = pd.read_csv(cv_dataset_path)

        if "actual_job_obtained" not in cvs.columns:
            raise ValueError(
                "❌ CV dataset needs 'actual_job_obtained' column.\n"
                "   This is the ground truth label (which job they got)."
            )

        job_types = cvs["actual_job_obtained"].unique()

        for job_type in job_types:
            job_cvs = cvs[cvs["actual_job_obtained"] == job_type]
            total   = len(job_cvs)

            all_skills = []
            for skills_str in job_cvs["skills"]:
                skill_list = [s.strip().lower()
                              for s in str(skills_str).split(",") if s.strip()]
                all_skills.extend(skill_list)

            skill_freq  = Counter(all_skills)
            self.skill_patterns[job_type]    = skill_freq
            self.job_cv_counts[job_type]     = total

            self.skill_frequencies[job_type] = {
                skill: count / total
                for skill, count in skill_freq.items()
            }

        self.is_trained = True

        os.makedirs(os.path.dirname(SKILL_GAP_MODEL_PATH), exist_ok=True)
        with open(SKILL_GAP_MODEL_PATH, "wb") as f:
            pickle.dump({
                "skill_patterns":    self.skill_patterns,
                "skill_frequencies": self.skill_frequencies,
                "job_cv_counts":     self.job_cv_counts,
            }, f)

        print(f"   ✅ Learned skill patterns for {len(job_types)} job types")
        for job in job_types:
            top3 = self.skill_patterns[job].most_common(3)
            top3_str = ", ".join(f"{s}({c})" for s, c in top3)
            print(f"     {job:35s}: top skills → {top3_str}")

        return self


    def predict_skill_gap(self, user_skills_str: str,
                          target_job: str,
                          threshold: float = 0.50) -> dict:
        """
        Predict skill gaps for a user targeting a specific job.

        Args:
            user_skills_str: Comma-separated user skills e.g. "Python, SQL"
            target_job:      Job title to target e.g. "Data Analyst"
            threshold:       Skills needed by > this % of placed candidates
                             are flagged as critical gaps (default 50%)

        Returns:
            dict with:
              - critical_gaps:   Skills >50% of placed candidates had (user lacks)
              - nice_to_have:    Skills 20-50% of placed candidates had
              - user_has:        Skills user already has that match the job
              - match_score:     % of critical skills the user already has
              - data_source:     How many CVs this is based on
        """
        if not self.is_trained:
            self._load()

        matched_job = self._match_job_title(target_job)
        if not matched_job:
            return {
                "error": f"No data for job: {target_job}",
                "available_jobs": list(self.skill_patterns.keys()),
            }

        user_skills  = {s.strip().lower()
                        for s in user_skills_str.split(",") if s.strip()}
        freq_map     = self.skill_frequencies[matched_job]
        n_cvs        = self.job_cv_counts[matched_job]

        critical_gaps  = []  # User lacks, >50% of placed candidates had
        nice_to_have   = []  # User lacks, 20-50% of placed candidates had
        user_has       = []  # User has, appears in placed candidates

        for skill, freq in sorted(freq_map.items(),
                                  key=lambda x: x[1], reverse=True):
            has_skill = skill in user_skills
            if has_skill:
                user_has.append({
                    "skill": skill,
                    "frequency": freq,
                    "pct": f"{freq*100:.0f}%",
                })
            elif freq >= threshold:
                critical_gaps.append({
                    "skill": skill,
                    "frequency": freq,
                    "pct": f"{freq*100:.0f}% of placed candidates had this",
                })
            elif freq >= 0.20:
                nice_to_have.append({
                    "skill": skill,
                    "frequency": freq,
                    "pct": f"{freq*100:.0f}% of placed candidates had this",
                })

        critical_skills = {s for s, f in freq_map.items() if f >= threshold}
        user_critical   = user_skills & critical_skills
        match_score     = (len(user_critical) / len(critical_skills) * 100
                           if critical_skills else 100)

        return {
            "target_job":    matched_job,
            "critical_gaps": critical_gaps,
            "nice_to_have":  nice_to_have,
            "user_has":      user_has,
            "match_score":   round(match_score, 1),
            "data_source":   f"Based on {n_cvs} real CVs",
        }

    def get_top_skills_for_job(self, job_title: str, top_n: int = 10) -> list:
        """Return the top N most common skills for a job type."""
        if not self.is_trained:
            self._load()
        matched = self._match_job_title(job_title)
        if not matched:
            return []
        return self.skill_patterns[matched].most_common(top_n)

    def get_all_job_types(self) -> list:
        """Return all job types in the training data."""
        if not self.is_trained:
            self._load()
        return sorted(self.skill_patterns.keys())

    def _match_job_title(self, title: str) -> str:
        """Fuzzy match job title against known job types."""
        title_lower = title.lower()
        for job in self.skill_patterns:
            if job.lower() == title_lower:
                return job
        for job in self.skill_patterns:
            if title_lower in job.lower() or job.lower() in title_lower:
                return job
        return None

    def _load(self):
        if not os.path.exists(SKILL_GAP_MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Skill gap model not found.\n"
                "   Run: predictor.learn_from_cvs('data/raw/cv_dataset.csv')"
            )
        with open(SKILL_GAP_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.skill_patterns    = data["skill_patterns"]
        self.skill_frequencies = data["skill_frequencies"]
        self.job_cv_counts     = data["job_cv_counts"]
        self.is_trained        = True
        print(f"✅ Loaded skill gap model "
              f"({len(self.skill_patterns)} job types)")



if __name__ == "__main__":
    predictor = SkillGapPredictor()
    predictor.learn_from_cvs("data/raw/cv_dataset.csv")

    print("\n" + "=" * 55)
    print("  SKILL GAP TEST")
    print("=" * 55)

    result = predictor.predict_skill_gap(
        user_skills_str="Python, SQL, Excel",
        target_job="Data Scientist",
    )

    print(f"\n🎯 Target: {result['target_job']}")
    print(f"   {result['data_source']}")
    print(f"   Match Score: {result['match_score']}%")

    print(f"\n❌ Critical Gaps ({len(result['critical_gaps'])}):")
    for g in result["critical_gaps"][:5]:
        print(f"   • {g['skill']:25s} — {g['pct']}")

    print(f"\n💡 Nice to Have ({len(result['nice_to_have'])}):")
    for g in result["nice_to_have"][:3]:
        print(f"   • {g['skill']:25s} — {g['pct']}")

    print(f"\n✅ You Already Have ({len(result['user_has'])}):")
    for g in result["user_has"][:5]:
        print(f"   • {g['skill']:25s} — {g['pct']}")
