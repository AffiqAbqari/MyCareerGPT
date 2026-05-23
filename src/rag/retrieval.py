"""
retrieval.py - RAG Pipeline with CV-Enhanced Matching

Pipeline:
    1. CVEnhancedMatcher  -> retrieves top-20 using CV-trained TF-IDF
    2. SkillGapPredictor  -> adds data-driven skill gaps
    3. CareerPathPredictor -> adds career path prediction context
    4. Formats everything into a rich prompt for Gemini API
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.data_mining.tfidf_matcher import TFIDFMatcher

CV_MATCHER_PATH  = "data/processed/cv_enhanced_matcher.pkl"
SKILL_GAP_PATH   = "data/processed/skill_gap_model.pkl"
CAREER_MODEL_PATH = "data/processed/career_path_model.pkl"


class RAGRetriever:
    """
    Retrieval component of the RAG pipeline.
    Auto-uses best available matcher: CV-Enhanced > Basic TF-IDF.
    """

    def __init__(self):
        self.matcher          = None
        self.gap_predictor    = None
        self.career_predictor = None
        self.using_cv_enhanced = False
        self._init_matcher()
        self._init_skill_gap()
        self._init_career_path()

    def _init_matcher(self):
        if os.path.exists(CV_MATCHER_PATH):
            try:
                from src.data_mining.cv_enhanced_tfidf import CVEnhancedMatcher
                print("✅ Using CV-Enhanced TF-IDF matcher")
                self.matcher = CVEnhancedMatcher()
                self.matcher._load()
                self.using_cv_enhanced = True
                return
            except Exception as e:
                print(f"⚠️ CV-Enhanced load failed ({e}), falling back to basic")
        print("ℹ️  Using basic TF-IDF matcher")
        print("   Run cv_training_pipeline.py --generate to enable CV-enhanced")
        self.matcher = TFIDFMatcher()
        self.matcher.build_index()

    def _init_skill_gap(self):
        if os.path.exists(SKILL_GAP_PATH):
            try:
                from src.data_mining.skill_gap_predictor import SkillGapPredictor
                self.gap_predictor = SkillGapPredictor()
                self.gap_predictor._load()
                print("✅ Skill gap predictor loaded (data-driven)")
            except Exception as e:
                print(f"ℹ️  Skill gap predictor not available ({e})")
        else:
            print("ℹ️  Skill gap: rule-based (run cv_training_pipeline.py to upgrade)")

    def _init_career_path(self):
        if os.path.exists(CAREER_MODEL_PATH):
            try:
                from src.data_mining.career_path_learner import CareerPathPredictor
                self.career_predictor = CareerPathPredictor()
                self.career_predictor._load()
                print("✅ Career path predictor loaded (Random Forest)")
            except Exception as e:
                print(f"ℹ️  Career path predictor not available ({e})")
        else:
            print("ℹ️  Career path predictor not trained yet")

    # ── Core Methods ──────────────────────────────────────────────────────────
    def retrieve(self, user_profile: dict, top_n: int = 20) -> list:
        """Retrieve top-N candidate jobs enriched with skill gap data."""
        if self.using_cv_enhanced:
            results_df = self.matcher.match_user_to_jobs(user_profile, top_n=top_n)
        else:
            results_df = self.matcher.match(user_profile, top_n=top_n)

        candidates = results_df.to_dict(orient="records")

        # ── FIX 2: Realistic match_percent ────────────────────────────────────
        # Raw TF-IDF cosine similarity clusters near 95-100% for similar docs,
        # which is misleading. We rescale it to a more meaningful 0-100% range
        # using skill overlap as the primary signal.
        user_skills = [
            s.strip().lower()
            for s in user_profile.get("skills", "").split(",")
            if s.strip()
        ]

        for candidate in candidates:
            required_str = candidate.get("skills_required", "") or ""
            required     = [s.strip().lower() for s in required_str.split(",") if s.strip()]

            if required and user_skills:
                # Skill overlap ratio (Jaccard-style)
                matched_count  = sum(1 for r in required if any(r in u or u in r for u in user_skills))
                skill_overlap  = matched_count / len(required)

                # Raw TF-IDF score (already normalised 0-1)
                raw_tfidf = candidate.get("cv_tfidf_score") or candidate.get("tfidf_score") or 0

                # ── Blended score: 60% skill overlap + 40% TF-IDF ─────────────
                # This gives a much more realistic percentage:
                # - Perfect skill match + high TF-IDF = ~85-95%
                # - Partial skill match                = ~50-75%
                # - Low match                          = ~20-40%
                blended = (0.60 * skill_overlap + 0.40 * float(raw_tfidf)) * 100

                # Apply a confidence penalty based on LLM later —
                # for now cap at 95% (100% should never happen in practice)
                candidate["match_percent"] = min(round(blended, 1), 95.0)
            else:
                # No skills data — use a reduced TF-IDF score
                raw_tfidf = candidate.get("cv_tfidf_score") or candidate.get("tfidf_score") or 0
                candidate["match_percent"] = min(round(float(raw_tfidf) * 70, 1), 70.0)

            # Populate matched/gap skills
            if required and user_skills:
                matched = [
                    r for r in (candidate.get("skills_required", "") or "").split(",")
                    if r.strip() and any(
                        r.strip().lower() in u or u in r.strip().lower()
                        for u in user_skills
                    )
                ]
                gaps = [
                    r for r in (candidate.get("skills_required", "") or "").split(",")
                    if r.strip() and not any(
                        r.strip().lower() in u or u in r.strip().lower()
                        for u in user_skills
                    )
                ]
                candidate["matched_skills"] = [m.strip() for m in matched[:8]]
                candidate["skill_gaps"]     = [g.strip() for g in gaps[:6]]

        # Enrich with data-driven skill gap predictor if available
        if self.gap_predictor:
            for candidate in candidates:
                gap_result = self.gap_predictor.predict_skill_gap(
                    user_profile.get("skills", ""),
                    candidate.get("title", "")
                )
                if "error" not in gap_result:
                    candidate["data_driven_gaps"] = [
                        g["skill"] for g in gap_result.get("critical_gaps", [])
                    ]
                    candidate["nice_to_have"]  = [
                        g["skill"] for g in gap_result.get("nice_to_have", [])
                    ]
                    candidate["gap_match_score"] = gap_result.get("match_score", 0)
                    candidate["data_source"]     = gap_result.get("data_source", "")

        return candidates

    def get_career_prediction(self, user_profile: dict) -> dict:
        """Get Random Forest career path prediction."""
        if self.career_predictor:
            return self.career_predictor.predict_career_path(user_profile)
        return {"predicted_job": "N/A", "confidence_pct": "N/A", "top3": []}

    def build_prompt(self, user_profile: dict, candidates: list) -> str:
        """
        Build LLM prompt with CV-enhanced context.

        FIX 3: Prompt is much stricter about:
          - Using EXACT job titles and company names from the list
          - NOT inventing or modifying any information
          - NOT mixing candidate's work history with job listings
        """
        user_skills   = user_profile.get("skills", "")
        user_edu      = user_profile.get("education", "")
        user_field    = user_profile.get("field", "")
        user_exp      = user_profile.get("experience", 0)
        user_riasec   = user_profile.get("riasec_type", "")
        user_interests = user_profile.get("interests", "")

        career_pred    = self.get_career_prediction(user_profile)
        career_context = ""
        if career_pred.get("predicted_job") not in ("N/A", None):
            career_context = (
                f"Career path prediction: {career_pred['predicted_job']} "
                f"({career_pred.get('confidence_pct','N/A')} confidence)"
            )

        # Build numbered job list — makes it easier for LLM to reference
        jobs_context = ""
        for i, job in enumerate(candidates[:10], 1):
            gaps     = (job.get("data_driven_gaps") or job.get("skill_gaps", []))[:5]
            matched  = job.get("matched_skills", [])[:5]
            nice     = job.get("nice_to_have", [])[:3]
            gap_score = job.get("gap_match_score", "")
            match_pct = job.get("match_percent", 0)

            jobs_context += (
                f"\nJob {i}:\n"
                f"  Title   : {job.get('title','')}\n"
                f"  Company : {job.get('company','Unknown')}\n"
                f"  Location: {job.get('location','Malaysia')}\n"
                f"  Required Skills: {job.get('skills_required','Not specified')}\n"
                f"  Match Score    : {match_pct:.1f}%\n"
                f"  Skills You Have: {', '.join(matched) or 'None matched'}\n"
                f"  Critical Gaps  : {', '.join(gaps) or 'None'}"
                f"{f' | Skill match: {gap_score}%' if gap_score else ''}\n"
                f"  Nice to Have   : {', '.join(nice) or 'N/A'}\n"
            )

        matcher_note = (
            "CV-Enhanced TF-IDF (trained on 1500 real Malaysian CV placements)"
            if self.using_cv_enhanced else "Basic TF-IDF keyword matching"
        )

        prompt = f"""CRITICAL RULES — READ BEFORE RESPONDING:
1. You MUST use EXACT job titles and company names from the numbered list below.
2. Do NOT modify, shorten, or paraphrase the job title or company name.
3. Do NOT mix the candidate's work history with the job listings.
4. Your response MUST start immediately with "RECOMMENDATION [1]:" — nothing before it.
5. Only recommend jobs from the list below — NEVER invent new jobs.
6. You MUST provide at least 3 recommendations, and up to 5 if there are good matches. If only 3 are strong matches, only provide 3. Do NOT pad with weak matches just to reach 5.
7. Each recommendation must be on separate lines with no blank lines within a block.

CANDIDATE PROFILE:
- Education : {user_edu} in {user_field}
- Skills    : {user_skills}
- Experience: {user_exp} years
- RIASEC    : {user_riasec}
- Interests : {user_interests}
{f'- {career_context}' if career_context else ''}

RETRIEVAL METHOD: {matcher_note}

AVAILABLE JOBS FROM MALAYSIAN DATABASE (choose only from these):
{jobs_context}

TASK: Select TOP 5 best matches from the jobs listed above.

MANDATORY FORMAT — copy the title and company EXACTLY as shown in the list:

RECOMMENDATION [1]:
Job: [copy EXACT title from list above — do not change it]
Company: [copy EXACT company from list above — do not change it]
Why Strong Match: [1-2 sentences explaining why the candidate fits this role]
Skills You Have: [comma separated matching skills from candidate profile]
Skills to Learn: [comma separated skill gaps for this role]
Learning Path: [specific course or certification with realistic timeline]
Confidence: [High/Medium/Low] - [one sentence reason based on skill match %]
---
RECOMMENDATION [2]:
Job: [copy EXACT title from list above]
Company: [copy EXACT company from list above]
Why Strong Match: [1-2 sentences]
Skills You Have: [comma separated]
Skills to Learn: [comma separated]
Learning Path: [specific course or cert]
Confidence: [High/Medium/Low] - [one sentence]
---
RECOMMENDATION [3]:
Job: [copy EXACT title from list above]
Company: [copy EXACT company from list above]
Why Strong Match: [1-2 sentences]
Skills You Have: [comma separated]
Skills to Learn: [comma separated]
Learning Path: [specific course or cert]
Confidence: [High/Medium/Low] - [one sentence]
---
RECOMMENDATION [4]:
Job: [copy EXACT title from list above]
Company: [copy EXACT company from list above]
Why Strong Match: [1-2 sentences]
Skills You Have: [comma separated]
Skills to Learn: [comma separated]
Learning Path: [specific course or cert]
Confidence: [High/Medium/Low] - [one sentence]
---
RECOMMENDATION [5]:
Job: [copy EXACT title from list above]
Company: [copy EXACT company from list above]
Why Strong Match: [1-2 sentences]
Skills You Have: [comma separated]
Skills to Learn: [comma separated]
Learning Path: [specific course or cert]
Confidence: [High/Medium/Low] - [one sentence]
---"""

        return prompt

    def get_top5_tfidf_only(self, candidates: list) -> list:
        """Return top 5 by TF-IDF only (ablation baseline)."""
        key = "cv_tfidf_score" if self.using_cv_enhanced else "tfidf_score"
        return sorted(candidates, key=lambda x: x.get(key, 0), reverse=True)[:5]

    @property
    def status(self) -> dict:
        return {
            "matcher":     "CV-Enhanced TF-IDF" if self.using_cv_enhanced else "Basic TF-IDF",
            "skill_gap":   "Data-driven" if self.gap_predictor else "Rule-based",
            "career_path": "Random Forest" if self.career_predictor else "Not available",
        }


if __name__ == "__main__":
    retriever = RAGRetriever()
    print(f"\n📊 Status: {retriever.status}")

    test_profile = {
        "skills":     "Python, SQL, Machine Learning, Pandas",
        "education":  "Bachelor of Computer Science",
        "field":      "Data Science",
        "experience": 1,
        "riasec_type": "IRA",
        "interests":  "AI, fintech",
    }
    candidates = retriever.retrieve(test_profile)
    print(f"✅ Retrieved {len(candidates)} candidates")
    print(f"   Sample match %: {[c.get('match_percent') for c in candidates[:5]]}")

    career = retriever.get_career_prediction(test_profile)
    print(f"🎯 Career Prediction: {career.get('predicted_job')} ({career.get('confidence_pct')})")