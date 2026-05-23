"""
career_path_learner.py - Random Forest Career Path Predictor
MyCareerGPT | CV Integration

Trains a Random Forest classifier on CV features to predict
which job a candidate is most likely to get.

Usage:
    from src.data_mining.career_path_learner import CareerPathPredictor
    predictor = CareerPathPredictor()
    predictor.train('data/raw/cv_dataset.csv')
    result = predictor.predict_career_path(user_profile)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

CAREER_MODEL_PATH = "data/processed/career_path_model.pkl"


class CareerPathPredictor:
    """
    Random Forest classifier: CV features → Predicted job title.

    Features used:
      - Education level (one-hot)
      - Field of study (one-hot)
      - CGPA (normalized)
      - Years of experience
      - Skills (binary presence flags for top skills)

    Target: actual_job_obtained (from CV ground truth)
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_leaf=2,
        )
        self.label_encoder    = LabelEncoder()
        self.feature_columns  = []
        self.all_skills_vocab = []
        self.is_trained       = False
        self.train_accuracy   = None
        self.cv_accuracy      = None
        self.classes_         = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, cv_dataset_path: str) -> "CareerPathPredictor":
        """
        Train Random Forest on CV dataset.

        Args:
            cv_dataset_path: Path to CV CSV with columns:
                education, field, cgpa, experience_years, skills,
                actual_job_obtained

        Returns:
            self
        """
        print(f"🌲 Training Random Forest on: {cv_dataset_path}")
        cvs = pd.read_csv(cv_dataset_path)

        # Build all_skills vocabulary from training data
        all_skills_flat = []
        for s in cvs["skills"]:
            all_skills_flat.extend(
                [x.strip().lower() for x in str(s).split(",") if x.strip()]
            )
        from collections import Counter
        skill_counts = Counter(all_skills_flat)
        # Only use skills that appear in at least 5 CVs (avoid noise)
        self.all_skills_vocab = [s for s, c in skill_counts.items() if c >= 5]

        # Build feature matrix
        X = self._extract_features(cvs)
        self.feature_columns = list(X.columns)

        # Encode target labels
        y_raw = cvs["actual_job_obtained"].fillna("Unknown")
        y     = self.label_encoder.fit_transform(y_raw)
        self.classes_ = list(self.label_encoder.classes_)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred  = self.model.predict(X_test)
        self.train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy       = accuracy_score(y_test, test_pred)

        # 5-fold cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")
        self.cv_accuracy = cv_scores.mean()

        print(f"\n   ✅ Model trained on {len(X_train)} CVs")
        print(f"   ✅ Test accuracy   : {test_accuracy:.3f}")
        print(f"   ✅ CV accuracy     : {self.cv_accuracy:.3f} ± {cv_scores.std():.3f}")
        print(f"   ✅ Classes         : {len(self.classes_)} job types")
        print(f"   ✅ Features        : {len(self.feature_columns)}")

        # Feature importance (top 10)
        importances = self.model.feature_importances_
        feat_imp    = sorted(zip(self.feature_columns, importances),
                             key=lambda x: x[1], reverse=True)
        print(f"\n   📊 Top 10 important features:")
        for feat, imp in feat_imp[:10]:
            bar = "█" * int(imp * 100)
            print(f"     {feat:30s} {imp:.3f} {bar}")

        # Classification report
        print(f"\n   📊 Per-class performance:")
        report = classification_report(
            y_test, test_pred,
            target_names=self.classes_,
            zero_division=0,
        )
        print(report)

        # Save
        self._save()
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_career_path(self, user_profile: dict) -> dict:
        """
        Predict the most likely job for a user profile.

        Args:
            user_profile: dict with education, field, cgpa,
                          experience (years), skills (comma-separated)

        Returns:
            dict with:
              - predicted_job:   Most likely job title
              - confidence:      Probability (0-1)
              - top3:            Top 3 predictions with probabilities
              - feature_importance: Which profile aspects drove the prediction
        """
        if not self.is_trained:
            self._load()

        row = pd.DataFrame([{
            "education":        user_profile.get("education", "Bachelor's Degree"),
            "field":            user_profile.get("field", ""),
            "cgpa":             user_profile.get("cgpa", 3.0),
            "experience_years": user_profile.get("experience", 0),
            "skills":           user_profile.get("skills", ""),
        }])

        X = self._extract_features(row)

        # Align columns with training (add missing, drop extra)
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        # Predict with probability
        proba         = self.model.predict_proba(X)[0]
        predicted_idx = np.argmax(proba)
        predicted_job = self.classes_[predicted_idx]
        confidence    = proba[predicted_idx]

        # Top 3 predictions
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {
                "job":        self.classes_[i],
                "probability": round(float(proba[i]), 3),
                "pct":        f"{proba[i]*100:.1f}%",
            }
            for i in top3_idx
        ]

        return {
            "predicted_job": predicted_job,
            "confidence":    round(float(confidence), 3),
            "confidence_pct": f"{confidence*100:.1f}%",
            "top3":          top3,
            "model_accuracy": f"{self.cv_accuracy:.1%}" if self.cv_accuracy else "N/A",
        }

    def validate_kmeans_clusters(self, users_df: pd.DataFrame,
                                 cluster_labels: list) -> dict:
        """
        Validate K-Means clusters using career path predictions.

        For each cluster, predict the career paths of users in it.
        A good cluster should have >60% users predicting the same job category.

        Args:
            users_df:       DataFrame of user profiles
            cluster_labels: K-Means cluster assignment per user

        Returns:
            dict: cluster_id → dominant job type and alignment score
        """
        if not self.is_trained:
            self._load()

        users_df = users_df.copy()
        users_df["cluster"]      = cluster_labels
        users_df["predicted_job"] = users_df.apply(
            lambda row: self.predict_career_path({
                "education": row.get("education", ""),
                "field":     row.get("field_of_study", ""),
                "cgpa":      row.get("cgpa", 3.0),
                "experience": row.get("experience_years", 0),
                "skills":    row.get("skills", ""),
            })["predicted_job"],
            axis=1,
        )

        results = {}
        for cluster_id in sorted(users_df["cluster"].unique()):
            cluster_users = users_df[users_df["cluster"] == cluster_id]
            from collections import Counter
            job_counts    = Counter(cluster_users["predicted_job"])
            dominant_job  = job_counts.most_common(1)[0][0]
            dominant_pct  = job_counts.most_common(1)[0][1] / len(cluster_users)

            results[int(cluster_id)] = {
                "dominant_job":    dominant_job,
                "alignment_score": round(dominant_pct, 3),
                "alignment_pct":   f"{dominant_pct*100:.1f}%",
                "user_count":      len(cluster_users),
                "job_distribution": dict(job_counts.most_common(3)),
                "validated":       dominant_pct >= 0.60,  # 60% threshold
            }

        print("\n📊 K-Means Cluster Validation:")
        for cid, info in results.items():
            status = "✅" if info["validated"] else "⚠️"
            print(f"  Cluster {cid}: {info['dominant_job']:30s} "
                  f"alignment={info['alignment_pct']} "
                  f"n={info['user_count']} {status}")

        return results

    # ── Feature Engineering ───────────────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert user/CV profiles to numeric feature matrix."""
        features = pd.DataFrame()

        # CGPA
        features["cgpa"] = pd.to_numeric(
            df.get("cgpa", pd.Series([3.0] * len(df))), errors="coerce"
        ).fillna(3.0)

        # Experience years
        features["experience_years"] = pd.to_numeric(
            df.get("experience_years", pd.Series([0] * len(df))), errors="coerce"
        ).fillna(0)

        # Education level (ordinal encoding)
        edu_map = {
            "spm": 1, "foundation": 2, "diploma": 3,
            "bachelor's degree": 4, "bachelor": 4,
            "master's degree": 5, "master": 5, "phd": 6,
        }
        edu_col = df.get("education", pd.Series(["Bachelor's Degree"] * len(df)))
        features["education_level"] = edu_col.str.lower().map(
            lambda x: next((v for k, v in edu_map.items() if k in str(x)), 4)
        )

        # Field of study (one-hot)
        fields = [
            "computer science", "data science", "software engineering",
            "information technology", "business", "finance",
            "engineering", "mathematics", "statistics", "accounting",
        ]
        field_col = df.get("field", pd.Series([""] * len(df)))
        for field in fields:
            features[f"field_{field.replace(' ', '_')}"] = (
                field_col.str.lower().str.contains(field, na=False).astype(int)
            )

        # Skills (binary presence) using vocabulary from training
        skills_col = df.get("skills", pd.Series([""] * len(df)))
        vocab = self.all_skills_vocab if self.all_skills_vocab else [
            "python", "sql", "java", "javascript", "machine learning",
            "deep learning", "data analysis", "excel", "r", "tableau",
            "power bi", "communication", "leadership", "project management",
            "docker", "aws", "git", "linux", "statistics", "accounting",
            "financial modeling", "networking", "cisco",
        ]
        for skill in vocab:
            safe_name = skill.replace(" ", "_").replace("-", "_")
            features[f"skill_{safe_name}"] = (
                skills_col.str.lower().str.contains(skill, na=False).astype(int)
            )

        return features

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(CAREER_MODEL_PATH), exist_ok=True)
        with open(CAREER_MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":            self.model,
                "label_encoder":    self.label_encoder,
                "feature_columns":  self.feature_columns,
                "all_skills_vocab": self.all_skills_vocab,
                "classes_":         self.classes_,
                "train_accuracy":   self.train_accuracy,
                "cv_accuracy":      self.cv_accuracy,
            }, f)
        print(f"\n   💾 Model saved to {CAREER_MODEL_PATH}")

    def _load(self):
        if not os.path.exists(CAREER_MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Career path model not found.\n"
                "   Run: predictor.train('data/raw/cv_dataset.csv')"
            )
        with open(CAREER_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.model            = data["model"]
        self.label_encoder    = data["label_encoder"]
        self.feature_columns  = data["feature_columns"]
        self.all_skills_vocab = data["all_skills_vocab"]
        self.classes_         = data["classes_"]
        self.train_accuracy   = data["train_accuracy"]
        self.cv_accuracy      = data["cv_accuracy"]
        self.is_trained       = True
        print(f"✅ Loaded career path model "
              f"({len(self.classes_)} classes, "
              f"CV accuracy={self.cv_accuracy:.1%})")


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Career Path Predictor — Train & Test")
    print("=" * 60)

    predictor = CareerPathPredictor()
    predictor.train("data/raw/cv_dataset.csv")

    print("\n" + "=" * 60)
    print("  PREDICTION TEST")
    print("=" * 60)

    test_profiles = [
        {
            "name":       "CS Graduate (Data Focus)",
            "education":  "Bachelor's Degree",
            "field":      "Computer Science",
            "cgpa":       3.7,
            "experience": 1,
            "skills":     "Python, SQL, Machine Learning, Pandas, Statistics",
        },
        {
            "name":       "Business Graduate",
            "education":  "Bachelor's Degree",
            "field":      "Business Administration",
            "cgpa":       3.2,
            "experience": 0,
            "skills":     "Excel, Communication, Leadership, Presentation, SQL",
        },
        {
            "name":       "Engineering Graduate",
            "education":  "Bachelor's Degree",
            "field":      "Electrical Engineering",
            "cgpa":       3.5,
            "experience": 2,
            "skills":     "Python, MATLAB, Linux, Networking, Cisco",
        },
    ]

    for profile in test_profiles:
        print(f"\n👤 {profile['name']}")
        print(f"   Skills: {profile['skills'][:50]}...")
        result = predictor.predict_career_path(profile)
        print(f"   🎯 Predicted: {result['predicted_job']} "
              f"({result['confidence_pct']} confidence)")
        print(f"   Top 3:")
        for r in result["top3"]:
            print(f"     • {r['job']:35s} {r['pct']}")
