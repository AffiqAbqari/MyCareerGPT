"""
database.py - Database Manager (OpenGauss / SQLite fallback)

Supports:
  - OpenGauss (Docker, port 5432) — set OPENGAUSS_* in .env
  - SQLite    (local fallback)    — used when OPENGAUSS_USER is empty

Run this FIRST before anything else.
Usage: python src/database.py --load data/raw/your_jobs.csv

Note: OpenGauss uses different upsert syntax than SQLite.
      SQLite: ON CONFLICT(email)
      OpenGauss: ON CONFLICT ON CONSTRAINT users_email_key
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load .env BEFORE reading environment variables
load_dotenv()

# ── Backend Selection ──────────────────────────────────────────────────────────
OPENGAUSS_HOST     = os.environ.get("OPENGAUSS_HOST",     "localhost")
OPENGAUSS_PORT     = os.environ.get("OPENGAUSS_PORT",     "5432")
OPENGAUSS_DB       = os.environ.get("OPENGAUSS_DB",       "malaysia_careers")
OPENGAUSS_USER     = os.environ.get("OPENGAUSS_USER",     "")
OPENGAUSS_PASSWORD = os.environ.get("OPENGAUSS_PASSWORD", "")

DB_PATH       = "data/malaysia_careers.db"
USE_OPENGAUSS = bool(OPENGAUSS_USER and OPENGAUSS_PASSWORD)


def get_connection():
    """
    Get a database connection.
    Returns OpenGauss (psycopg2) if OPENGAUSS_USER + OPENGAUSS_PASSWORD
    are set in .env, otherwise falls back to SQLite.
    """
    if USE_OPENGAUSS:
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=OPENGAUSS_HOST,
                port=int(OPENGAUSS_PORT),
                dbname=OPENGAUSS_DB,
                user=OPENGAUSS_USER,
                password=OPENGAUSS_PASSWORD,
                connect_timeout=10,
            )
            conn.autocommit = False
            return conn
        except ImportError:
            print("⚠️  psycopg2 not installed. Run: pip install psycopg2-binary")
            print("   Falling back to SQLite...")
        except Exception as e:
            print(f"⚠️  OpenGauss connection failed: {e}")
            print("   Falling back to SQLite...")

    import sqlite3
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _is_opengauss(conn) -> bool:
    """Check if this is a psycopg2 (OpenGauss) connection."""
    return "psycopg2" in type(conn).__module__


def _ph(conn) -> str:
    """SQL placeholder: %s for OpenGauss, ? for SQLite."""
    return "%s" if _is_opengauss(conn) else "?"


def create_schema():
    """Create all tables if they don't exist. Works for both backends."""
    conn  = get_connection()
    is_og = _is_opengauss(conn)

    if is_og:
        cursor = conn.cursor()
        statements = [
            """CREATE TABLE IF NOT EXISTS jobs (
                id               SERIAL PRIMARY KEY,
                title            TEXT NOT NULL,
                company          TEXT,
                location         TEXT,
                skills_required  TEXT,
                description      TEXT,
                salary_min       REAL,
                salary_max       REAL,
                industry         TEXT,
                job_type         TEXT,
                created_at       TIMESTAMP DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS users (
                id               SERIAL PRIMARY KEY,
                name             TEXT NOT NULL,
                email            TEXT UNIQUE,
                education        TEXT,
                field_of_study   TEXT,
                university       TEXT,
                cgpa             REAL,
                skills           TEXT,
                experience_years INTEGER DEFAULT 0,
                riasec_type      TEXT,
                created_at       TIMESTAMP DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS recommendations (
                id               SERIAL PRIMARY KEY,
                user_id          INTEGER NOT NULL REFERENCES users(id),
                job_id           INTEGER NOT NULL REFERENCES jobs(id),
                tfidf_score      REAL,
                llm_score        REAL,
                final_score      REAL,
                rank_position    INTEGER,
                explanation      TEXT,
                matched_skills   TEXT,
                skill_gaps       TEXT,
                created_at       TIMESTAMP DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS user_skills (
                id           SERIAL PRIMARY KEY,
                user_id      INTEGER NOT NULL REFERENCES users(id),
                skill_name   TEXT NOT NULL,
                proficiency  TEXT DEFAULT 'beginner',
                source       TEXT DEFAULT 'self-reported'
            )""",
            """CREATE TABLE IF NOT EXISTS tfidf_cache (
                id            SERIAL PRIMARY KEY,
                indexed_at    TIMESTAMP DEFAULT NOW(),
                job_count     INTEGER,
                feature_count INTEGER,
                status        TEXT DEFAULT 'active'
            )""",
            """CREATE TABLE IF NOT EXISTS feedback (
                id                      SERIAL PRIMARY KEY,
                user_id                 INTEGER REFERENCES users(id),
                user_email              TEXT,
                rating                  INTEGER CHECK (rating BETWEEN 1 AND 5),
                recommendation_quality  INTEGER CHECK (recommendation_quality BETWEEN 1 AND 5),
                ease_of_use             INTEGER CHECK (ease_of_use BETWEEN 1 AND 5),
                comments                TEXT,
                would_recommend         BOOLEAN DEFAULT TRUE,
                submitted_at            TIMESTAMP DEFAULT NOW()
            )""",
        ]
        for stmt in statements:
            cursor.execute(stmt)
        conn.commit()
    else:
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                title            TEXT NOT NULL,
                company          TEXT,
                location         TEXT,
                skills_required  TEXT,
                description      TEXT,
                salary_min       REAL,
                salary_max       REAL,
                industry         TEXT,
                job_type         TEXT,
                created_at       TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS users (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT NOT NULL,
                email            TEXT UNIQUE,
                education        TEXT,
                field_of_study   TEXT,
                university       TEXT,
                cgpa             REAL,
                skills           TEXT,
                experience_years INTEGER DEFAULT 0,
                riasec_type      TEXT,
                created_at       TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS recommendations (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          INTEGER NOT NULL,
                job_id           INTEGER NOT NULL,
                tfidf_score      REAL,
                llm_score        REAL,
                final_score      REAL,
                rank_position    INTEGER,
                explanation      TEXT,
                matched_skills   TEXT,
                skill_gaps       TEXT,
                created_at       TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (job_id)  REFERENCES jobs(id)
            );
            CREATE TABLE IF NOT EXISTS user_skills (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                skill_name   TEXT NOT NULL,
                proficiency  TEXT DEFAULT 'beginner',
                source       TEXT DEFAULT 'self-reported',
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS tfidf_cache (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                indexed_at    TEXT DEFAULT (datetime('now')),
                job_count     INTEGER,
                feature_count INTEGER,
                status        TEXT DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                 INTEGER REFERENCES users(id),
                user_email              TEXT,
                rating                  INTEGER CHECK (rating BETWEEN 1 AND 5),
                recommendation_quality  INTEGER CHECK (recommendation_quality BETWEEN 1 AND 5),
                ease_of_use             INTEGER CHECK (ease_of_use BETWEEN 1 AND 5),
                comments                TEXT,
                would_recommend         INTEGER DEFAULT 1,
                submitted_at            TEXT DEFAULT (datetime('now'))
            );
        """)

    conn.close()
    backend = (
        f"OpenGauss @ {OPENGAUSS_HOST}:{OPENGAUSS_PORT}/{OPENGAUSS_DB}"
        if USE_OPENGAUSS else f"SQLite @ {DB_PATH}"
    )
    print(f"✅ Schema created — Backend: {backend}")


def load_jobs_from_csv(csv_path: str):
    """Load your jobs CSV into the database."""
    COLUMN_MAP = {
        "job_title":       "title",     "job title":    "title",
        "position":        "title",     "employer":     "company",
        "organization":    "company",   "city":         "location",
        "state":           "location",  "skills":       "skills_required",
        "required_skills": "skills_required",           "requirements": "skills_required",
        "job_description": "description", "overview":   "description",
        "min_salary":      "salary_min", "max_salary":  "salary_max",
        "sector":          "industry",  "category":     "industry",
        "employment_type": "job_type",  "type":         "job_type",
    }

    print(f"📂 Loading jobs from: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns=COLUMN_MAP, inplace=True)

    if "title" not in df.columns:
        raise ValueError(
            f"❌ Column 'title' not found.\n"
            f"   Columns: {list(df.columns)}\n"
            f"   Add to COLUMN_MAP in database.py"
        )

    for col in ["company", "location", "skills_required", "description",
                "salary_min", "salary_max", "industry", "job_type"]:
        if col not in df.columns:
            df[col] = None

    df["title"]       = df["title"].fillna("Unknown Role").str.strip()
    df["description"] = df["description"].fillna("").str.strip()

    import re
    SKILL_KEYWORDS = [
        "Python", "SQL", "Excel", "Java", "JavaScript", "TypeScript", "R",
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Keras",
        "Power BI", "Tableau", "Looker", "Data Analysis", "Data Science",
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git", "Linux",
        "React", "Node.js", "Django", "Flask", "FastAPI", "Laravel", "PHP",
        "Communication", "Leadership", "Teamwork", "Problem Solving",
        "Project Management", "Agile", "Scrum", "SAP", "AutoCAD",
        "Accounting", "Finance", "Marketing", "Sales", "Negotiation",
        "Statistics", "Pandas", "NumPy", "Scikit-learn", "Spark", "Hadoop",
        "MongoDB", "PostgreSQL", "MySQL", "Oracle", "Redis",
        "Figma", "Adobe XD", "Photoshop", "UI/UX", "SEO", "Google Ads",
        "C++", "C#", ".NET", "Swift", "Kotlin", "Flutter", "Android",
        "Network Security", "Penetration Testing", "Cybersecurity", "Firewall",
        "SolidWorks", "ANSYS", "PLC", "SCADA",
    ]

    def extract_skills(text):
        if not isinstance(text, str) or not text:
            return ""
        return ", ".join(
            s for s in SKILL_KEYWORDS
            if re.search(r'\b' + re.escape(s) + r'\b', text, re.IGNORECASE)
        )

    desc_col = "descriptions" if "descriptions" in df.columns else "description"
    df["description"]     = df[desc_col].fillna("").str.strip()
    df["skills_required"] = df["skills_required"].fillna("").str.strip()
    mask = df["skills_required"] == ""
    df.loc[mask, "skills_required"] = df.loc[mask, "description"].apply(extract_skills)
    print(f"✅ Skills extracted: {(df['skills_required'] != '').sum():,} jobs have skills")

    conn   = get_connection()
    cursor = conn.cursor()
    ph     = _ph(conn)
    cursor.execute("DELETE FROM jobs")
    inserted = 0
    skipped  = 0

    for _, row in df.iterrows():
        try:
            cursor.execute(
                f"""INSERT INTO jobs (title, company, location, skills_required,
                    description, salary_min, salary_max, industry, job_type)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})""",
                (
                    str(row.get("title", "")),
                    str(row.get("company", "") or ""),
                    str(row.get("location", "") or ""),
                    str(row.get("skills_required", "") or ""),
                    str(row.get("description", "") or ""),
                    _safe_float(row.get("salary_min")),
                    _safe_float(row.get("salary_max")),
                    str(row.get("industry", "") or ""),
                    str(row.get("job_type", "") or ""),
                ),
            )
            inserted += 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"⚠️  Skipped row: {e}")

    conn.commit()
    conn.close()
    print(f"✅ Loaded {inserted} jobs ({skipped} skipped)")
    return inserted


def _safe_float(val):
    try:
        return float(val) if pd.notna(val) else None
    except (ValueError, TypeError):
        return None


def get_all_jobs() -> pd.DataFrame:
    """Fetch all jobs as a DataFrame — used by TF-IDF matcher."""
    conn = get_connection()
    if _is_opengauss(conn):
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs")
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=cols)
    else:
        df = pd.read_sql_query("SELECT * FROM jobs", conn)
        conn.close()
        return df


def get_job_count() -> int:
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jobs")
    count  = cursor.fetchone()[0]
    conn.close()
    return count


def save_user(name, email, education, field, university, cgpa,
              skills, experience, riasec) -> int:
    """Insert or update a user, return their ID."""
    conn   = get_connection()
    cursor = conn.cursor()
    ph     = _ph(conn)

    if _is_opengauss(conn):
        # OpenGauss — check first then insert or update
        cursor.execute(f"SELECT id FROM users WHERE email = {ph}", (email,))
        existing = cursor.fetchone()
        if existing:
            user_id = existing[0]
            cursor.execute(
                f"""UPDATE users SET name={ph}, education={ph},
                    field_of_study={ph}, university={ph}, cgpa={ph},
                    skills={ph}, experience_years={ph}, riasec_type={ph}
                    WHERE email={ph}""",
                (name, education, field, university, cgpa,
                 skills, experience, riasec, email),
            )
        else:
            cursor.execute(
                f"""INSERT INTO users (name, email, education, field_of_study,
                    university, cgpa, skills, experience_years, riasec_type)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
                    RETURNING id""",
                (name, email, education, field, university,
                 cgpa, skills, experience, riasec),
            )
            user_id = cursor.fetchone()[0]
    else:
        # SQLite uses ON CONFLICT(column)
        cursor.execute(
            f"""INSERT INTO users (name, email, education, field_of_study,
                university, cgpa, skills, experience_years, riasec_type)
                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
                ON CONFLICT(email) DO UPDATE SET
                name=excluded.name,
                education=excluded.education,
                field_of_study=excluded.field_of_study,
                university=excluded.university,
                cgpa=excluded.cgpa,
                skills=excluded.skills,
                experience_years=excluded.experience_years,
                riasec_type=excluded.riasec_type""",
            (name, email, education, field, university, cgpa, skills, experience, riasec),
        )
        user_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return user_id


def save_recommendations(user_id: int, recommendations: list):
    """Save a batch of recommendations for a user."""
    conn   = get_connection()
    cursor = conn.cursor()
    ph     = _ph(conn)
    cursor.execute(f"DELETE FROM recommendations WHERE user_id = {ph}", (user_id,))
    for rec in recommendations:
        cursor.execute(
            f"""INSERT INTO recommendations
                (user_id, job_id, tfidf_score, llm_score, final_score,
                 rank_position, explanation, matched_skills, skill_gaps)
                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})""",
            (
                user_id,
                rec["job_id"],
                rec.get("tfidf_score", 0.0),
                rec.get("llm_score", 0.0),
                rec.get("final_score", 0.0),
                rec.get("rank_position", 0),
                rec.get("explanation", ""),
                json.dumps(rec.get("matched_skills", [])),
                json.dumps(rec.get("skill_gaps", [])),
            ),
        )
    conn.commit()
    conn.close()


def save_feedback(user_email: str, rating: int, rec_quality: int,
                  ease_of_use: int, comments: str, would_recommend: bool,
                  user_id: int = None):
    """Save UAT user feedback to the database."""
    conn   = get_connection()
    cursor = conn.cursor()
    ph     = _ph(conn)
    cursor.execute(
        f"""INSERT INTO feedback
            (user_id, user_email, rating, recommendation_quality,
             ease_of_use, comments, would_recommend)
            VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph})""",
        (user_id, user_email, rating, rec_quality, ease_of_use,
         comments, 1 if would_recommend else 0),
    )
    conn.commit()
    conn.close()


def get_feedback_stats() -> dict:
    """Return feedback summary statistics."""
    conn = get_connection()
    if _is_opengauss(conn):
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*), AVG(rating), AVG(recommendation_quality), AVG(ease_of_use) FROM feedback"
        )
        row = cursor.fetchone()
        conn.close()
        return {
            "count":           row[0] or 0,
            "avg_rating":      round(float(row[1] or 0), 2),
            "avg_rec_quality": round(float(row[2] or 0), 2),
            "avg_ease":        round(float(row[3] or 0), 2),
        }
    else:
        df = pd.read_sql_query(
            "SELECT COUNT(*) as count, AVG(rating) as avg_rating, "
            "AVG(recommendation_quality) as avg_rec_quality, "
            "AVG(ease_of_use) as avg_ease FROM feedback",
            conn,
        )
        conn.close()
        return {
            "count":           int(df["count"].iloc[0] or 0),
            "avg_rating":      round(float(df["avg_rating"].iloc[0] or 0), 2),
            "avg_rec_quality": round(float(df["avg_rec_quality"].iloc[0] or 0), 2),
            "avg_ease":        round(float(df["avg_ease"].iloc[0] or 0), 2),
        }


def db_status():
    """Print a quick health check of the database."""
    conn   = get_connection()
    cursor = conn.cursor()

    def _count(t):
        cursor.execute(f"SELECT COUNT(*) FROM {t}")
        return cursor.fetchone()[0]

    jobs  = _count("jobs")
    users = _count("users")
    recs  = _count("recommendations")
    fb    = _count("feedback")
    conn.close()

    backend = (
        f"OpenGauss @ {OPENGAUSS_HOST}:{OPENGAUSS_PORT}/{OPENGAUSS_DB}"
        if USE_OPENGAUSS else f"SQLite @ {DB_PATH}"
    )
    print(f"\n📊 Database Status — {backend}")
    print(f"   Jobs            : {jobs:,}")
    print(f"   Users           : {users:,}")
    print(f"   Recommendations : {recs:,}")
    print(f"   Feedback entries: {fb:,}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyCareerGPT Database Manager")
    parser.add_argument("--load",   metavar="CSV", help="Load jobs from CSV")
    parser.add_argument("--status", action="store_true", help="Show DB status")
    parser.add_argument("--reset",  action="store_true", help="Reset all tables")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    create_schema()

    if args.load:
        load_jobs_from_csv(args.load)
    if args.reset:
        conn   = get_connection()
        cursor = conn.cursor()
        for tbl in ["feedback", "recommendations", "user_skills", "users", "jobs", "tfidf_cache"]:
            cursor.execute(f"DELETE FROM {tbl}")
        conn.commit()
        conn.close()
        print("🗑️  All tables cleared")

    db_status()