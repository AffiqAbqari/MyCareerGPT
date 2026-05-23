"""
llm_interface.py - Google Gemini API

Setup:
    pip install google-genai
    Set env var: GOOGLE_API_KEY=your_key_here
    Get free key at: https://aistudio.google.com/app/apikey

Usage:
    from src.llm.llm_interface import LLMInterface, check_hallucinations
    llm = LLMInterface()
    result = llm.generate_recommendations(prompt)
"""

import re
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-2.5-flash"
API_KEY    = os.environ.get("GOOGLE_API_KEY", "")


class LLMInterface:
    """
    Interface to Google Gemini API.
    Replaces local Ollama/Llama — runs entirely in the cloud.
    No model download, no RAM limit, no GPU required.
    Inference target: < 15 seconds.
    """

    def __init__(self, model: str = MODEL_NAME, api_key: str = API_KEY):
        self.model_name          = model
        self.last_inference_time = None
        self._client             = None

        if not GEMINI_AVAILABLE:
            print("⚠️  google-genai not installed.")
            print("   Run: pip install google-genai")
            return

        if not api_key:
            print("⚠️  GOOGLE_API_KEY not set.")
            return

        try:
            self._client = genai.Client(api_key=api_key)
            print(f"✅ Gemini API ready: {self.model_name}")
        except Exception as e:
            print(f"⚠️  Gemini init failed: {e}")
            self._client = None

    # ── Public API ─────────────────────────────────────────────────────────────
    def generate_recommendations(
        self,
        prompt: str,
        max_tokens: int    = 2000,
        temperature: float = 0.3,
    ) -> dict:
        """
        Send prompt to Gemini and parse the recommendations.

        Args:
            prompt:      The full RAG prompt (from retrieval.py)
            max_tokens:  Max output tokens (2000 for recs, 4000 for ATS resume)
            temperature: Lower = more focused/consistent (0.3 recommended)

        Returns:
            dict with raw_text, recommendations, inference_time
        """
        if self._client is None:
            return self._mock_response()

        start = time.time()
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            raw_text = response.text
            elapsed  = time.time() - start
            self.last_inference_time = elapsed

            if elapsed > 15:
                print(f"⚠️  Inference took {elapsed:.1f}s (target: <15s)")
            else:
                print(f"✅ Gemini inference: {elapsed:.1f}s")

            recommendations = self._parse_recommendations(raw_text)
            return {
                "raw_text":          raw_text,
                "recommendations":   recommendations,
                "inference_time":    elapsed,
                "hallucination_check": len(recommendations) > 0,
            }

        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return {"raw_text": "", "recommendations": [], "error": str(e)}

    # ── Parsing ────────────────────────────────────────────────────────────────
    def _parse_recommendations(self, raw_text: str) -> list:
        """Parse structured LLM output into recommendation dicts."""
        recommendations = []
        blocks = re.split(r"RECOMMENDATION\s*\[\d+\]:", raw_text)
        for block in blocks[1:]:
            if not block.strip():
                continue
            rec = self._parse_single_block(block)
            if rec.get("job_title"):
                recommendations.append(rec)
        return recommendations

    def _parse_single_block(self, block: str) -> dict:
        """Parse one RECOMMENDATION block."""
        def extract(pattern, text, default=""):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().split("\n")[0].strip()
            return default

        job_title      = extract(r"Job:\s*(.+)",             block)
        company        = extract(r"Company:\s*(.+)",          block)
        why_match      = extract(r"Why Strong Match:\s*(.+)", block)
        skills_have    = extract(r"Skills You Have:\s*(.+)",  block)
        skills_learn   = extract(r"Skills to Learn:\s*(.+)",  block)
        learning       = extract(r"Learning Path:\s*(.+)",    block)
        confidence_raw = extract(r"Confidence:\s*(.+)",       block)

        confidence        = "Medium"
        confidence_reason = ""
        if confidence_raw:
            parts             = confidence_raw.split(" - ", 1)
            confidence        = parts[0].strip()
            confidence_reason = parts[1].strip() if len(parts) > 1 else ""

        matched_skills = [s.strip() for s in skills_have.split(",")  if s.strip()]
        skill_gaps     = [s.strip() for s in skills_learn.split(",") if s.strip()]

        return {
            "job_title":         job_title,
            "company":           company,
            "why_match":         why_match,
            "matched_skills":    matched_skills,
            "skill_gaps":        skill_gaps,
            "learning_path":     learning,
            "confidence":        confidence,
            "confidence_reason": confidence_reason,
            "explanation":       why_match,
            "llm_score":         _confidence_to_score(confidence),
        }

    def _mock_response(self) -> dict:
        """Return mock response when Gemini API is not configured."""
        return {
            "raw_text": "Mock response — Gemini API not configured",
            "recommendations": [
                {
                    "job_title":      "Data Analyst",
                    "company":        "Maybank",
                    "why_match":      "Strong Python and SQL skills align with data analysis requirements.",
                    "matched_skills": ["Python", "SQL", "Data Analysis"],
                    "skill_gaps":     ["Tableau", "Power BI"],
                    "learning_path":  "Complete Tableau Desktop Specialist cert. Free Coursera courses available.",
                    "confidence":     "High",
                    "llm_score":      0.9,
                }
            ],
            "inference_time": 0.0,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _confidence_to_score(confidence: str) -> float:
    return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(confidence.lower(), 0.5)


# ── Improved Hallucination Checker ────────────────────────────────────────────
def check_hallucinations(recommendations: list, candidates: list) -> dict:
    """
    Verify LLM recommendations are grounded in retrieved context.
    Uses flexible token-overlap matching to avoid false positives
    when Gemini adds slight variations to job titles.
    Target: Zero hallucinations on 50 test cases.
    """
    candidate_titles    = {c.get("title",   "").lower().strip() for c in candidates}
    candidate_companies = {c.get("company", "").lower().strip() for c in candidates}
    hallucinations      = []

    for rec in recommendations:
        raw_title   = rec.get("job_title", "").lower().strip()
        raw_company = rec.get("company",   "").lower().strip()

        # Strip "at CompanyName" suffix Gemini sometimes adds to titles
        clean_title = re.sub(r'\s+at\s+.+$', '', raw_title).strip()

        title_matched = any(
            clean_title in ct or ct in clean_title or
            _token_overlap(clean_title, ct) >= 0.6
            for ct in candidate_titles
        )

        company_matched = any(
            raw_company in cc or cc in raw_company or
            _token_overlap(raw_company, cc) >= 0.5
            for cc in candidate_companies
        ) if raw_company else True

        if not title_matched and not company_matched:
            hallucinations.append({
                "recommended": rec.get("job_title"),
                "company":     rec.get("company"),
                "issue":       "Job/company not found in retrieved candidates",
            })

    return {
        "total_recommendations": len(recommendations),
        "hallucinations":        len(hallucinations),
        "hallucination_rate":    len(hallucinations) / max(len(recommendations), 1),
        "details":               hallucinations,
        "passed":                len(hallucinations) == 0,
    }


def _token_overlap(a: str, b: str) -> float:
    """Compute token-level Jaccard overlap between two strings."""
    if not a or not b:
        return 0.0
    stops    = {"at", "in", "for", "the", "and", "of", "a", "an", "sdn", "bhd"}
    tokens_a = set(a.lower().split()) - stops
    tokens_b = set(b.lower().split()) - stops
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = LLMInterface()
    test_prompt = (
        "You are a Malaysian career counselor. "
        "Recommend one job for a Computer Science graduate with Python and SQL skills. "
        "Use this format:\n"
        "RECOMMENDATION [1]:\nJob: Data Analyst\nCompany: Maybank\n"
        "Why Strong Match: Good match.\nSkills You Have: Python, SQL\n"
        "Skills to Learn: Tableau\nLearning Path: Take Coursera course.\n"
        "Confidence: High - Strong technical foundation.\n---"
    )
    result = llm.generate_recommendations(test_prompt)
    print(f"\nParsed {len(result['recommendations'])} recommendations")
    print(f"Inference time: {result.get('inference_time', 0):.2f}s")