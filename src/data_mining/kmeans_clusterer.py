"""
kmeans_clusterer.py - K-Means User Clustering

Groups users into career clusters based on their profile features.
Target: 5 interpretable clusters, Silhouette score >= 0.461

"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CLUSTER_MODEL_PATH = "data/processed/kmeans_model.pkl"

CLUSTER_LABELS = {
    0: "Tech & Software",
    1: "Data & Analytics",
    2: "Business & Management",
    3: "Engineering & Manufacturing",
    4: "Finance & Administration",
}


class KMeansClusterer:
    """
    Groups users into career personality clusters.

    Features used:
      - Field of study (one-hot encoded)
      - Education level (ordinal encoded)
      - CGPA (normalized)
      - Years of experience
      - RIASEC type (encoded)
      - Top skills (binary presence)
      - PCA dimensionality reduction (improves silhouette)
    """

    def __init__(self, k: int = 5):
        self.k       = k
        self.model   = None
        self.scaler  = StandardScaler()
        self.pca     = None
        self.feature_columns = []
        self.silhouette      = None
        self.is_fitted       = False

    def fit(self, users_df: pd.DataFrame) -> "KMeansClusterer":
        X, self.feature_columns = self._extract_features(users_df)
        X_scaled = self.scaler.fit_transform(X)

        #  PCA: reduce to meaningful components 
        n_components = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
        self.pca     = PCA(n_components=n_components, random_state=42)
        X_pca        = self.pca.fit_transform(X_scaled)
        explained    = sum(self.pca.explained_variance_ratio_) * 100
        print(f"   PCA: {n_components} components → {explained:.1f}% variance explained")

        best_k, best_score, best_model = self._find_best_k(X_pca)
        self.k         = best_k
        self.model     = best_model
        self.silhouette = best_score
        self.is_fitted  = True

        os.makedirs(os.path.dirname(CLUSTER_MODEL_PATH), exist_ok=True)
        with open(CLUSTER_MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":           self.model,
                "scaler":          self.scaler,
                "pca":             self.pca,
                "feature_columns": self.feature_columns,
                "k":               self.k,
                "silhouette":      self.silhouette,
            }, f)

        print(f"✅ K-Means fitted: k={best_k}, silhouette={best_score:.3f}")
        return self

    def predict(self, user_profile: dict) -> dict:
        if not self.is_fitted:
            self._load()

        row = pd.DataFrame([user_profile])
        X, _ = self._extract_features(row)

        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        X_scaled   = self.scaler.transform(X)
        X_pca      = self.pca.transform(X_scaled)
        cluster_id = int(self.model.predict(X_pca)[0])
        label      = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")

        return {
            "cluster_id":  cluster_id,
            "label":       label,
            "description": _cluster_description(cluster_id),
        }

    def get_cluster_summary(self, users_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            self._load()
        X, _ = self._extract_features(users_df)
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        X_scaled   = self.scaler.transform(X)
        X_pca      = self.pca.transform(X_scaled)
        users_df   = users_df.copy()
        users_df["cluster"]       = self.model.predict(X_pca)
        users_df["cluster_label"] = users_df["cluster"].map(CLUSTER_LABELS)

        summary = users_df.groupby("cluster_label").agg(
            count=("cluster", "count"),
            avg_cgpa=("cgpa", "mean"),
            avg_experience=("experience_years", "mean"),
        ).round(2)
        return summary

    #  Feature Engineering 
    def _extract_features(self, df: pd.DataFrame):
        features = pd.DataFrame(index=df.index)

        #  CGPA (continuous, normalized 0–4) 
        features["cgpa"] = pd.to_numeric(
            df.get("cgpa", 0), errors="coerce"
        ).fillna(0)

        # Experience years (continuous) 
        features["experience"] = pd.to_numeric(
            df.get("experience_years", 0), errors="coerce"
        ).fillna(0)

        # Education level (ordinal: Diploma=1, Bachelor=2, Master=3, PhD=4)
        edu_map = {
            "spm": 0, "diploma": 1, "foundation": 1,
            "bachelor": 2, "degree": 2,
            "master": 3, "msc": 3,
            "phd": 4, "doctor": 4,
        }
        edu_col = df.get("education", pd.Series([""] * len(df)))
        def encode_edu(val):
            v = str(val).lower()
            for k, score in edu_map.items():
                if k in v:
                    return score
            return 2  
        features["education_level"] = edu_col.apply(encode_edu)

        #  Field of study (one-hot) 
        fields = [
            "computer science", "data science", "software engineering",
            "information technology", "business", "engineering",
            "mathematics", "statistics", "finance",
        ]
        field_col = df.get("field_of_study", pd.Series([""] * len(df)))
        for field in fields:
            features[f"field_{field.replace(' ', '_')}"] = (
                field_col.str.lower().str.contains(field, na=False).astype(int)
            )

        # RIASEC type (one-hot per letter) 
        riasec_col = df.get("riasec_type", pd.Series([""] * len(df)))
        for code in "RIASEC":
            features[f"riasec_{code}"] = (
                riasec_col.str.upper().str.contains(code, na=False).astype(int)
            )

        # Skills (binary presence — expanded list) 
        top_skills = [
            "python", "sql", "excel", "java", "javascript", "r",
            "machine learning", "deep learning", "data analysis",
            "project management", "leadership", "communication",
            "power bi", "tableau", "statistics", "docker", "aws",
            "accounting", "finance", "marketing", "autocad", "matlab",
        ]
        skills_col = df.get("skills", pd.Series([""] * len(df)))
        for skill in top_skills:
            features[f"skill_{skill.replace(' ', '_')}"] = (
                skills_col.str.lower().str.contains(skill, na=False).astype(int)
            )

        feature_columns = list(features.columns)
        return features.fillna(0), feature_columns

    def _find_best_k(self, X_pca):
        """Test K=3 to K=8, pick best silhouette score."""
        print("   Testing K values 3–8 for best silhouette score...")
        best_k, best_score, best_model = 5, -1, None

        for k in range(3, 9):
            if k >= len(X_pca):
                continue
            model  = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            labels = model.fit_predict(X_pca)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_pca, labels)
            print(f"     K={k} → silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k     = k
                best_model = model

        return best_k, best_score, best_model

    def _load(self):
        if not os.path.exists(CLUSTER_MODEL_PATH):
            raise FileNotFoundError(
                "❌ K-Means model not found. Run clusterer.fit(users_df) first."
            )
        with open(CLUSTER_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.model           = data["model"]
        self.scaler          = data["scaler"]
        self.pca             = data.get("pca")
        self.feature_columns = data["feature_columns"]
        self.k               = data["k"]
        self.silhouette      = data["silhouette"]
        self.is_fitted       = True


def _cluster_description(cluster_id: int) -> str:
    descriptions = {
        0: "Strong fit for software development, IT, and tech roles. Typically CS/SE graduates with programming skills.",
        1: "Best matched to data analyst, data scientist, and ML engineer roles. Strong in Python and statistics.",
        2: "Suited to management, consulting, and business analyst roles. Mix of technical and soft skills.",
        3: "Engineering-focused roles in manufacturing, electronics, and infrastructure sectors.",
        4: "Finance, accounting, and administrative roles. Strong in Excel, analytical thinking.",
    }
    return descriptions.get(cluster_id, "General professional profile.")
