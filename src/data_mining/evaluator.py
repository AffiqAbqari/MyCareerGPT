"""
evaluator.py - Precision@5 & Ablation Study

Usage:
  python src/data_mining/evaluator.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def precision_at_k(recommendations: List[int], ground_truth: List[int], k: int = 5) -> float:
    """
    Calculate Precision@K.

    Args:
        recommendations: Ranked list of job IDs from your system
        ground_truth:    List of job IDs labelled as relevant by experts
        k:               K value (default 5 per your FYP)

    Returns:
        Precision@K score between 0.0 and 1.0

    Example:
        recommendations = [101, 205, 303, 410, 512]
        ground_truth    = [101, 303, 512, 620, 781]
        precision_at_5  = 3/5 = 0.60
    """
    top_k = recommendations[:k]
    relevant = [job for job in top_k if job in ground_truth]
    return len(relevant) / k


def ndcg_at_k(recommendations: List[int], ground_truth: List[int], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain @K.
    Target from FYP doc: NDCG@10 >= 0.827
    """
    def dcg(ranked, relevant, k):
        score = 0.0
        for i, job in enumerate(ranked[:k]):
            if job in relevant:
                score += 1.0 / np.log2(i + 2) 
        return score

    actual_dcg  = dcg(recommendations, ground_truth, k)
    ideal_ranked = [j for j in recommendations if j in ground_truth][:k]
    ideal_dcg   = dcg(ideal_ranked, ground_truth, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def f1_score_at_k(recommendations: List[int], ground_truth: List[int], k: int = 5) -> float:
    """F1 score combining precision and recall. Target: >= 0.75"""
    top_k     = set(recommendations[:k])
    relevant  = top_k & set(ground_truth)
    precision = len(relevant) / k if k > 0 else 0
    recall    = len(relevant) / len(ground_truth) if ground_truth else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class AblationStudy:
    """
    Runs the ablation study required by your FYP doc:
      Configuration 1: TF-IDF only          → expect ~66.4%
      Configuration 2: TF-IDF + LLM rerank  → expect ~78.4%

    How to use:
      1. Prepare ground_truth: get 3 expert career counselors to label
         which of your system's top-20 candidates are "relevant" for
         each test user. Your FYP doc mentions Fleiss' Kappa = 0.72.

      2. Run this ablation with both recommendation lists.
    """

    def __init__(self):
        self.results = {}

    def run(self,
            test_users: List[Dict],
            tfidf_recs: Dict[int, List[int]],   # user_id → ranked job_ids (TF-IDF only)
            full_recs:  Dict[int, List[int]],   # user_id → ranked job_ids (TF-IDF + LLM)
            ground_truth: Dict[int, List[int]], # user_id → relevant job_ids (expert labels)
            k: int = 5) -> Dict:
        """
        Run complete ablation study.

        Args:
            test_users:   List of user profile dicts with 'id' key
            tfidf_recs:   Recommendations from TF-IDF only
            full_recs:    Recommendations from full system (TF-IDF + LLM)
            ground_truth: Expert-labelled relevant jobs per user
            k:            K for Precision@K (default 5)

        Returns:
            Dict with all metrics
        """
        tfidf_p5  = []
        full_p5   = []
        tfidf_ndcg = []
        full_ndcg  = []
        tfidf_f1   = []
        full_f1    = []

        for user in test_users:
            uid = user["id"]
            if uid not in ground_truth:
                continue

            gt = ground_truth[uid]

            if uid in tfidf_recs:
                tfidf_p5.append(precision_at_k(tfidf_recs[uid], gt, k))
                tfidf_ndcg.append(ndcg_at_k(tfidf_recs[uid], gt, 10))
                tfidf_f1.append(f1_score_at_k(tfidf_recs[uid], gt, k))

            if uid in full_recs:
                full_p5.append(precision_at_k(full_recs[uid], gt, k))
                full_ndcg.append(ndcg_at_k(full_recs[uid], gt, 10))
                full_f1.append(f1_score_at_k(full_recs[uid], gt, k))

        self.results = {
            "tfidf_only": {
                f"precision@{k}": np.mean(tfidf_p5),
                "ndcg@10":        np.mean(tfidf_ndcg),
                f"f1@{k}":        np.mean(tfidf_f1),
                "n_users":        len(tfidf_p5),
            },
            "full_system": {
                f"precision@{k}": np.mean(full_p5),
                "ndcg@10":        np.mean(full_ndcg),
                f"f1@{k}":        np.mean(full_f1),
                "n_users":        len(full_p5),
            },
        }

        tfidf_prec = self.results["tfidf_only"][f"precision@{k}"]
        full_prec  = self.results["full_system"][f"precision@{k}"]
        improvement = ((full_prec - tfidf_prec) / tfidf_prec * 100
                       if tfidf_prec > 0 else 0)
        self.results["llm_improvement_pct"] = round(improvement, 1)

        self._print_results(k)
        return self.results

    def _print_results(self, k):
        print("\n" + "=" * 55)
        print("  ABLATION STUDY RESULTS")
        print("=" * 55)

        configs = [
            ("TF-IDF Only (baseline)",   "tfidf_only"),
            ("TF-IDF + LLM (full system)", "full_system"),
        ]

        for label, key in configs:
            r = self.results[key]
            print(f"\n  {label}")
            for metric, val in r.items():
                if isinstance(val, float):
                    target = _get_target(metric)
                    status = "✅" if val >= target else "❌"
                    print(f"    {metric:20s}: {val:.3f}  {status} (target {target})")

        print(f"\n  LLM improvement  : +{self.results['llm_improvement_pct']}%")
        print("=" * 55)


def _get_target(metric: str) -> float:
    targets = {
        "precision@5": 0.784,
        "ndcg@10":     0.827,
        "f1@5":        0.75,
    }
    return targets.get(metric, 0.0)


def generate_expert_labelling_sheet(test_users: List[Dict],
                                    candidate_jobs: Dict[int, pd.DataFrame],
                                    output_path: str = "data/expert_labels_template.csv"):
    """
    Generate a CSV for your 3 expert career counselors to fill in.

    They label each candidate job as:
      1 = Relevant, 0 = Not Relevant

    After collection, calculate Fleiss' Kappa for inter-rater agreement.
    Target: Kappa >= 0.72 (as stated in your FYP doc).
    """
    rows = []
    for user in test_users:
        uid = user["id"]
        if uid not in candidate_jobs:
            continue
        jobs = candidate_jobs[uid].head(20)  
        for _, job in jobs.iterrows():
            rows.append({
                "user_id":        uid,
                "user_name":      user.get("name", ""),
                "user_skills":    user.get("skills", ""),
                "job_id":         job["id"],
                "job_title":      job["title"],
                "job_skills":     job.get("skills_required", ""),
                "expert1_label":  "",  
                "expert2_label":  "", 
                "expert3_label":  "", 
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Expert labelling sheet saved: {output_path}")
    print(f"   {len(rows)} rows for {len(test_users)} users × 20 candidates each")
    return df


def calculate_fleiss_kappa(labels_df: pd.DataFrame) -> float:
    """
    Calculate Fleiss' Kappa for inter-rater agreement among 3 experts.
    Target: >= 0.72

    Args:
        labels_df: DataFrame with columns expert1_label, expert2_label, expert3_label
                   Values: 1 (relevant) or 0 (not relevant)
    """
    for col in ["expert1_label", "expert2_label", "expert3_label"]:
        labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce")

    labels_df = labels_df.dropna(subset=["expert1_label", "expert2_label", "expert3_label"])
    n = len(labels_df)
    k_raters = 3
    k_categories = 2 

    P_i = []
    for _, row in labels_df.iterrows():
        labels = [row["expert1_label"], row["expert2_label"], row["expert3_label"]]
        n_agree = sum(labels.count(c) * (labels.count(c) - 1) for c in [0, 1])
        P_i.append(n_agree / (k_raters * (k_raters - 1)))

    P_bar = np.mean(P_i)

    p_j = []
    all_labels = (labels_df[["expert1_label", "expert2_label", "expert3_label"]]
                  .values.flatten())
    for cat in [0, 1]:
        p_j.append(np.sum(all_labels == cat) / (n * k_raters))

    P_e = sum(p ** 2 for p in p_j)

    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0
    print(f"\n📊 Fleiss' Kappa: {kappa:.3f} ({'✅' if kappa >= 0.72 else '❌'} target: 0.72)")
    return kappa


#  Quick Demo

if __name__ == "__main__":
    print("📊 Evaluator loaded. Functions available:")
    print("   precision_at_k(recs, ground_truth, k=5)")
    print("   ndcg_at_k(recs, ground_truth, k=10)")
    print("   f1_score_at_k(recs, ground_truth, k=5)")
    print("   AblationStudy().run(users, tfidf_recs, full_recs, ground_truth)")
    print("   generate_expert_labelling_sheet(users, candidates)")
    print("   calculate_fleiss_kappa(labels_df)")

    # Demo with synthetic data
    print("\n--- Demo with synthetic data ---")
    recs_tfidf = [101, 205, 303, 410, 512, 620, 781, 830]
    recs_full  = [101, 303, 512, 205, 620, 410, 781, 830]
    ground_truth = [101, 303, 512, 620, 781]

    print(f"TF-IDF Precision@5 : {precision_at_k(recs_tfidf, ground_truth):.3f}")
    print(f"Full   Precision@5 : {precision_at_k(recs_full, ground_truth):.3f}")
    print(f"NDCG@10            : {ndcg_at_k(recs_full, ground_truth):.3f}")
    print(f"F1@5               : {f1_score_at_k(recs_full, ground_truth):.3f}")
