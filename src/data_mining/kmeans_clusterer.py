"""
kmeans_clusterer.py - K-Means User Clustering

Groups users into career clusters based on their profile features.
Target: 5 interpretable clusters, Silhouette score >= 0.461

Usage:
  from src.data_mining.kmeans_clusterer import KMeansClusterer
  clusterer = KMeansClusterer(k=5)
  clusterer.fit(users_df)
  cluster_id = clusterer.predict(user_profile)
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CLUSTER_MODEL_PATH = "data/processed/kmeans_model.pkl"

# Human-readable cluster labels (update after inspecting your actual clusters)
CLUSTER_LABELS = {
    0: "Tech & Software",
    1: "Data & Analytics",
    2: "Business & Management",
    3: "Engineering & Manufacturing",
    4: "Finance & Administration",
}


class KMeansClusterer:
    """
    Groups users into 5 career personality clusters.

    Features used:
      - Field of study (one-hot encoded)
      - CGPA (normalized)
      - Years of experience
      - RIASEC type (encoded)
      - Top skills (binary presence)
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.model: KMeans = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.silhouette = None
        self.is_fitted = False

    def fit(self, users_df: pd.DataFrame) -> "KMeansClusterer":
        """
        Fit K-Means on a DataFrame of user profiles.

        Args:
            users_df: DataFrame with columns from the users table.

        Returns:
            self
        """
        X, self.feature_columns = self._extract_features(users_df)
        X_scaled = self.scaler.fit_transform(X)

        # Find best K using silhouette score
        best_k, best_score, best_model = self._find_best_k(X_scaled)
        self.k = best_k
        self.model = best_model
        self.silhouette = best_score
        self.is_fitted = True

        # Save model
        os.makedirs(os.path.dirname(CLUSTER_MODEL_PATH), exist_ok=True)
        with open(CLUSTER_MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "k": self.k,
                "silhouette": self.silhouette,
            }, f)

        print(f"✅ K-Means fitted: k={best_k}, silhouette={best_score:.3f}")
        return self

    def predict(self, user_profile: dict) -> dict:
        """
        Predict cluster for a single user.

        Returns:
            dict with cluster_id, label, description, similar_users_count
        """
        if not self.is_fitted:
            self._load()

        row = pd.DataFrame([user_profile])
        X, _ = self._extract_features(row)

        # Handle missing features vs. training set
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        X_scaled = self.scaler.transform(X)
        cluster_id = int(self.model.predict(X_scaled)[0])
        label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")

        return {
            "cluster_id": cluster_id,
            "label": label,
            "description": _cluster_description(cluster_id),
        }

    def get_cluster_summary(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Return per-cluster statistics for the admin dashboard."""
        if not self.is_fitted:
            self._load()
        X, _ = self._extract_features(users_df)
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        X_scaled = self.scaler.transform(X)
        users_df = users_df.copy()
        users_df["cluster"] = self.model.predict(X_scaled)
        users_df["cluster_label"] = users_df["cluster"].map(CLUSTER_LABELS)

        summary = users_df.groupby("cluster_label").agg(
            count=("cluster", "count"),
            avg_cgpa=("cgpa", "mean"),
            avg_experience=("experience_years", "mean"),
        ).round(2)
        return summary

    # ── Feature Engineering ────────────────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame):
        """Convert user profiles to numeric feature matrix."""
        features = pd.DataFrame()

        # CGPA (normalized 0-4)
        features["cgpa"] = pd.to_numeric(df.get("cgpa", 0), errors="coerce").fillna(0)

        # Experience years
        features["experience"] = pd.to_numeric(
            df.get("experience_years", 0), errors="coerce"
        ).fillna(0)

        # Field of study — one-hot encoding
        fields = ["computer science", "data science", "software engineering",
                  "information technology", "business", "engineering",
                  "mathematics", "finance"]
        field_col = df.get("field_of_study", pd.Series([""] * len(df)))
        for field in fields:
            features[f"field_{field.replace(' ', '_')}"] = (
                field_col.str.lower().str.contains(field, na=False).astype(int)
            )

        # RIASEC type — one-hot per letter
        riasec_col = df.get("riasec_type", pd.Series([""] * len(df)))
        for code in "RIASEC":
            features[f"riasec_{code}"] = (
                riasec_col.str.upper().str.contains(code, na=False).astype(int)
            )

        # Top skills — binary presence
        top_skills = [
            "python", "sql", "excel", "java", "javascript", "r",
            "machine learning", "deep learning", "data analysis",
            "project management", "leadership", "communication",
        ]
        skills_col = df.get("skills", pd.Series([""] * len(df)))
        for skill in top_skills:
            features[f"skill_{skill.replace(' ', '_')}"] = (
                skills_col.str.lower().str.contains(skill, na=False).astype(int)
            )

        feature_columns = list(features.columns)
        return features, feature_columns

    def _find_best_k(self, X_scaled):
        """Test K=3 to K=8, pick the K with best silhouette score."""
        print("   Testing K values 3–8 for best silhouette score...")
        best_k, best_score, best_model = 5, -1, None

        for k in range(3, 9):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            print(f"     K={k} → silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_model = model

        return best_k, best_score, best_model

    def _load(self):
        if not os.path.exists(CLUSTER_MODEL_PATH):
            raise FileNotFoundError(
                "❌ K-Means model not found. Run clusterer.fit(users_df) first."
            )
        with open(CLUSTER_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.k = data["k"]
        self.silhouette = data["silhouette"]
        self.is_fitted = True


def _cluster_description(cluster_id: int) -> str:
    descriptions = {
        0: "Strong fit for software development, IT, and tech roles. Typically CS/SE graduates with programming skills.",
        1: "Best matched to data analyst, data scientist, and ML engineer roles. Strong in Python and statistics.",
        2: "Suited to management, consulting, and business analyst roles. Mix of technical and soft skills.",
        3: "Engineering-focused roles in manufacturing, electronics, and infrastructure sectors.",
        4: "Finance, accounting, and administrative roles. Strong in Excel, analytical thinking.",
    }
    return descriptions.get(cluster_id, "General professional profile.")
