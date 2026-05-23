"""
admin_panel.py - MyCareerGPT Admin Dashboard
src/app/admin_panel.py

Run with: streamlit run src/app/admin_panel.py
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.database import get_connection, load_jobs_from_csv, get_job_count, create_schema, db_status

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Admin · MyCareerGPT",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
 
    /* ── Global ─────────────────────────────────────────────────────── */
    .stApp {
        background: #021a14 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
 
    /* ── Sidebar ────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #073B3A 0%, #042820 100%) !important;
        border-right: 1px solid #0B6E4F !important;
    }
    [data-testid="stSidebar"] * { color: #a8d5b5 !important; }
 
    /* ── Admin Header ───────────────────────────────────────────────── */
    .admin-header {
        background: linear-gradient(135deg, #073B3A 0%, #0B6E4F 60%, #08A045 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #08A045;
        box-shadow: 0 0 30px rgba(8,160,69,0.15);
    }
    .admin-header h2 {
        color: #21D375 !important;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    .admin-header p {
        color: #6BBF59 !important;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }
 
    /* ── Metric Cards ───────────────────────────────────────────────── */
    .admin-metric {
        background: linear-gradient(135deg, #0a2a20, #073B3A);
        border: 1px solid #0B6E4F;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .admin-metric .value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #21D375;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    .admin-metric .label {
        color: #6BBF59;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 6px;
    }
    .admin-metric .sublabel { color: #08A045; font-size: 0.75rem; }
 
    /* ── Status Dots ────────────────────────────────────────────────── */
    .status-ready  { color: #21D375; font-weight: 600; }
    .status-warn   { color: #6BBF59; font-weight: 600; }
    .status-error  { color: #08A045; font-weight: 600; }
 
    /* ── Data Tables ────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #0B6E4F !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] th {
        background: #073B3A !important;
        color: #21D375 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stDataFrame"] td {
        background: #0a2a20 !important;
        color: #c8ecd4 !important;
        border-color: #0B6E4F !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background: #0d3526 !important;
    }
 
    /* ── Buttons ────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #08A045, #0B6E4F) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(8,160,69,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #21D375, #08A045) !important;
        box-shadow: 0 4px 25px rgba(33,211,117,0.4) !important;
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #6BBF59 !important;
        border: 1px solid #0B6E4F !important;
        border-radius: 10px !important;
    }
 
    /* ── Expanders ──────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: #0a2a20 !important;
        border: 1px solid #0B6E4F !important;
        border-radius: 10px !important;
    }
 
    /* ── Inputs ─────────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: #0a2a20 !important;
        border-color: #0B6E4F !important;
        color: #d4f5e0 !important;
        border-radius: 8px !important;
    }
 
    /* ── Tabs ───────────────────────────────────────────────────────── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #073B3A !important;
        border-radius: 10px;
        padding: 4px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #6BBF59 !important;
        border-radius: 8px;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #08A045 !important;
        color: white !important;
    }
 
    /* ── Progress ───────────────────────────────────────────────────── */
    .stProgress > div > div > div { background-color: #08A045 !important; }
    .stProgress > div > div { background-color: #073B3A !important; }
 
    /* ── Charts ─────────────────────────────────────────────────────── */
    [data-testid="stVegaLiteChart"] { background: transparent !important; }
 
    /* ── Text ───────────────────────────────────────────────────────── */
    h1, h2, h3 { color: #21D375 !important; }
    h4, h5, h6 { color: #6BBF59 !important; }
    p, li, span, label, td, th { color: #c8ecd4 !important; }
    code {
        background: #073B3A !important;
        color: #21D375 !important;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace !important;
    }
 
    /* ── File Uploader ──────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: #0a2a20 !important;
        border: 2px dashed #0B6E4F !important;
        border-radius: 12px !important;
    }
 
    /* ── Success/Info/Warning ───────────────────────────────────────── */
    [data-testid="stAlert"] { border-radius: 10px !important; }
 
    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #021a14; }
    ::-webkit-scrollbar-thumb { background: #0B6E4F; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #08A045; }
 
    hr { border-color: #0B6E4F !important; opacity: 0.3; }
 
    /* ── System Log ─────────────────────────────────────────────────── */
    .log-entry {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #6BBF59;
        padding: 4px 0;
        border-bottom: 1px solid rgba(11,110,79,0.2);
    }
    
    /* ── Stat Cards (Dashboard) ─────────────────────────────────── */
    .stat-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #0a2a20, #073B3A);
        border: 1px solid #0B6E4F;
        border-radius: 10px;
        padding: 1.25rem;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: #21D375;
    }
    .stat-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 400;
        color: #21D375;
        line-height: 1;
        margin-bottom: 6px;
    }
    .stat-label {
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #6BBF59;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-sub {
        font-size: 11px;
        color: #08A045;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Section Labels ─────────────────────────────────────────── */
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #6BBF59;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #0B6E4F;
    }

    /* ── Admin Title ────────────────────────────────────────────── */
    .admin-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #21D375;
        margin-bottom: 4px;
    }
    .admin-subtitle {
        font-size: 24px;
        font-weight: 300;
        color: #6BBF59;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }

    /* ── Status Dots ────────────────────────────────────────────── */
    .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    .dot-ok   { background: #21D375; }
    .dot-warn { background: #6BBF59; }
    .dot-err  { background: #08A045; }

    /* ── Log Terminal ───────────────────────────────────────────── */
    .log-box {
        background: #042820;
        border: 1px solid #0B6E4F;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #21D375;
        max-height: 260px;
        overflow-y: auto;
        line-height: 1.7;
    }

    /* ── Target Metric Rows ─────────────────────────────────────── */
    .target-row { display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid #0B6E4F; font-size: 13px; }
    .target-name { flex: 2; color: #c8ecd4; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .target-val  { flex: 1; color: #6BBF59; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .target-badge { padding: 2px 10px; border-radius: 3px; font-size: 11px; font-family: 'JetBrains Mono', monospace; font-weight: 500; }
    .badge-pass { background: rgba(33,211,117,0.15); color: #21D375; border: 1px solid rgba(33,211,117,0.3); }
    .badge-pend { background: rgba(107,191,89,0.15); color: #6BBF59; border: 1px solid rgba(107,191,89,0.3); }
    .badge-fail { background: rgba(8,160,69,0.15);   color: #08A045; border: 1px solid rgba(8,160,69,0.3); }
    </style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "admin_page" not in st.session_state:
    st.session_state.admin_page = "dashboard"
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

def log(msg: str, level: str = "info"):
    icons = {"info": "→", "ok": "✓", "warn": "!", "err": "✗"}
    ts    = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_lines.append(f"[{ts}] {icons.get(level,'→')} {msg}")
    if len(st.session_state.log_lines) > 80:
        st.session_state.log_lines = st.session_state.log_lines[-80:]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("```\nADMIN CONSOLE\nMyCareerGPT\n```")
    st.markdown("---")

    nav = {
        "[ 01 ] Dashboard":    "dashboard",
        "[ 02 ] Job database": "jobs",
        "[ 03 ] Users":        "users",
        "[ 04 ] Model status": "models",
        "[ 05 ] Metrics":      "metrics",
        "[ 06 ] System log":   "log",
    }
    for label, page_id in nav.items():
        active = st.session_state.admin_page == page_id
        if st.button(label, key=f"nav_{page_id}"):
            st.session_state.admin_page = page_id
            st.rerun()

    st.markdown("---")
    st.markdown(f"<span style='font-size:10px;color:#333;font-family:monospace'>v1.0 · FYP02-DS-T2610-0382</span>", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_stats():
    try:
        conn = get_connection()
        jobs  = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        recs  = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
        avg_trust = conn.execute(
            "SELECT AVG(llm_score) FROM recommendations WHERE llm_score > 0"
        ).fetchone()[0] or 0
        conn.close()
        return jobs, users, recs, avg_trust
    except Exception:
        return 0, 0, 0, 0

def model_exists(path):
    return os.path.exists(path)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.admin_page == "dashboard":
    st.markdown('<div class="admin-title">MyCareerGPT</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-subtitle">Admin dashboard</div>', unsafe_allow_html=True)

    jobs, users, recs, avg_trust = get_stats()

    # Stat cards
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-num">{jobs:,}</div>
            <div class="stat-label">Job postings</div>
            <div class="stat-sub">Malaysian market</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{users}</div>
            <div class="stat-label">Registered users</div>
            <div class="stat-sub">UAT target: 30</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{recs}</div>
            <div class="stat-label">Recommendations</div>
            <div class="stat-sub">Generated total</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{avg_trust:.2f}</div>
            <div class="stat-label">Avg trust score</div>
            <div class="stat-sub">Target: ≥ 0.9</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # System health
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-label">System health</div>', unsafe_allow_html=True)

        checks = [
            ("Database",           os.path.exists("data/malaysia_careers.db"),      "data/malaysia_careers.db"),
            ("TF-IDF index",       model_exists("data/processed/tfidf_index.pkl"),   "data/processed/tfidf_index.pkl"),
            ("CV-Enhanced TF-IDF", model_exists("data/processed/cv_enhanced_matcher.pkl"), "data/processed/cv_enhanced_matcher.pkl"),
            ("Skill gap model",    model_exists("data/processed/skill_gap_model.pkl"), "data/processed/skill_gap_model.pkl"),
            ("Career path model",  model_exists("data/processed/career_path_model.pkl"), "data/processed/career_path_model.pkl"),
            ("CV dataset",         model_exists("data/raw/cv_dataset.csv"),           "data/raw/cv_dataset.csv"),
        ]

        for name, ok, path in checks:
            dot   = "dot-ok" if ok else "dot-warn"
            label = "ready" if ok else "missing"
            st.markdown(
                f'<div style="padding:8px 0;border-bottom:1px solid #f5f5f5;font-size:13px;">'
                f'<span class="status-dot {dot}"></span>'
                f'<span style="color:#333">{name}</span>'
                f'<span style="float:right;font-family:monospace;font-size:11px;color:{"#22c55e" if ok else "#f0c040"}">{label}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.markdown('<div class="section-label">Quick actions</div>', unsafe_allow_html=True)

        if st.button("↻  Re-index TF-IDF (rebuild from jobs DB)", use_container_width=True):
            with st.spinner("Rebuilding TF-IDF index..."):
                try:
                    from src.data_mining.tfidf_matcher import TFIDFMatcher
                    m = TFIDFMatcher()
                    m.build_index(force_rebuild=True)
                    log(f"TF-IDF reindexed — {jobs} jobs", "ok")
                    st.success(f"✓ Re-indexed {jobs:,} jobs")
                except Exception as e:
                    log(f"Re-index failed: {e}", "err")
                    st.error(str(e))

        if st.button("⚙  Run CV training pipeline (--generate)", use_container_width=True):
            with st.spinner("Generating CVs + training all models..."):
                try:
                    from src.data_generation.generate_cv_dataset import generate_cv_dataset
                    from src.data_mining.cv_training_pipeline import run_pipeline
                    generate_cv_dataset(1500, "data/raw/cv_dataset.csv")
                    run_pipeline("data/raw/cv_dataset.csv", run_ablation=False)
                    log("CV training pipeline completed", "ok")
                    st.success("✓ All CV models trained")
                    st.rerun()
                except Exception as e:
                    log(f"CV pipeline failed: {e}", "err")
                    st.error(str(e))

        if st.button("🗑  Clear all recommendations", use_container_width=True):
            conn = get_connection()
            conn.execute("DELETE FROM recommendations")
            conn.commit()
            conn.close()
            log("All recommendations cleared", "warn")
            st.warning("Recommendations cleared")
            st.rerun()

        if st.button("📊  Export database report (CSV)", use_container_width=True):
            conn = get_connection()
            df   = pd.read_sql_query(
                "SELECT u.name, u.email, u.field_of_study, u.skills, "
                "COUNT(r.id) as rec_count, AVG(r.final_score) as avg_score "
                "FROM users u LEFT JOIN recommendations r ON u.id=r.user_id "
                "GROUP BY u.id", conn
            )
            conn.close()
            csv = df.to_csv(index=False)
            st.download_button("Download report.csv", csv,
                               "report.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: JOB DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.admin_page == "jobs":
    st.markdown('<div class="admin-title">MyCareerGPT</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-subtitle">Job database</div>', unsafe_allow_html=True)

    # Upload new CSV
    st.markdown('<div class="section-label">Upload new jobs CSV</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your jobs CSV here",
        type=["csv"],
        help="Must have a 'title' column. Other columns are auto-mapped."
    )

    if uploaded:
        preview_df = pd.read_csv(uploaded)
        st.markdown(f"**Preview** — {len(preview_df):,} rows, columns: `{list(preview_df.columns)}`")
        st.dataframe(preview_df.head(5), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✓  Load into database", type="primary", use_container_width=True):
                os.makedirs("data/raw", exist_ok=True)
                tmp_path = "data/raw/_upload_temp.csv"
                preview_df.to_csv(tmp_path, index=False)
                with st.spinner("Loading jobs..."):
                    try:
                        create_schema()
                        n = load_jobs_from_csv(tmp_path)
                        log(f"Loaded {n} jobs from upload", "ok")
                        st.success(f"✓ Loaded {n:,} jobs")
                        # Auto reindex
                        from src.data_mining.tfidf_matcher import TFIDFMatcher
                        TFIDFMatcher().build_index(force_rebuild=True)
                        log("TF-IDF auto-reindexed after upload", "ok")
                        st.rerun()
                    except Exception as e:
                        log(f"Upload failed: {e}", "err")
                        st.error(str(e))

    st.markdown("---")

    # Browse jobs
    st.markdown('<div class="section-label">Browse job postings</div>', unsafe_allow_html=True)

    conn = get_connection()
    jobs_df = pd.read_sql_query("SELECT * FROM jobs", conn)
    conn.close()

    if len(jobs_df) == 0:
        st.info("No jobs loaded yet. Upload a CSV above.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            search = st.text_input("Search title / company", placeholder="e.g. Data Analyst")
        with fc2:
            industries = ["All"] + sorted(jobs_df["industry"].dropna().unique().tolist())
            industry   = st.selectbox("Industry", industries)
        with fc3:
            locations = ["All"] + sorted(jobs_df["location"].dropna().unique().tolist())
            location  = st.selectbox("Location", locations)

        filtered = jobs_df.copy()
        if search:
            mask = (filtered["title"].str.contains(search, case=False, na=False) |
                    filtered["company"].str.contains(search, case=False, na=False))
            filtered = filtered[mask]
        if industry != "All":
            filtered = filtered[filtered["industry"] == industry]
        if location != "All":
            filtered = filtered[filtered["location"] == location]

        st.markdown(f'<div style="font-family:monospace;font-size:12px;color:#888;margin-bottom:8px">'
                    f'Showing {len(filtered):,} of {len(jobs_df):,} jobs</div>',
                    unsafe_allow_html=True)

        display_cols = ["title", "company", "location", "industry",
                        "skills_required", "salary_min", "salary_max"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(200), use_container_width=True, height=380)

        # Industry breakdown
        st.markdown('<div class="section-label" style="margin-top:1.5rem">Industry breakdown</div>',
                    unsafe_allow_html=True)
        if "industry" in jobs_df.columns:
            ind_counts = jobs_df["industry"].value_counts().head(10)
            st.bar_chart(ind_counts)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: USERS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.admin_page == "users":
    st.markdown('<div class="admin-title">MyCareerGPT</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-subtitle">Users & recommendations</div>', unsafe_allow_html=True)

    conn = get_connection()
    users_df = pd.read_sql_query("SELECT * FROM users ORDER BY created_at DESC", conn)
    recs_df  = pd.read_sql_query("""
        SELECT r.*, u.name as user_name, j.title as job_title, j.company
        FROM recommendations r
        JOIN users u ON r.user_id = u.id
        JOIN jobs  j ON r.job_id  = j.id
        ORDER BY r.created_at DESC
        LIMIT 200
    """, conn)
    conn.close()

    if len(users_df) == 0:
        st.info("No users registered yet. Launch the main app and create a profile.")
    else:
        # UAT progress
        uat_count = len(users_df)
        uat_target = 30
        uat_pct    = min(uat_count / uat_target, 1.0)

        st.markdown('<div class="section-label">UAT progress</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Participants", f"{uat_count} / {uat_target}")
        with col2:
            st.metric("Still needed", max(0, uat_target - uat_count))
        with col3:
            st.metric("Progress", f"{uat_pct*100:.0f}%")
        st.progress(uat_pct)

        st.markdown("---")
        st.markdown('<div class="section-label">Registered users</div>', unsafe_allow_html=True)

        display_u = ["name", "email", "field_of_study", "education",
                     "cgpa", "experience_years", "riasec_type", "created_at"]
        display_u = [c for c in display_u if c in users_df.columns]
        st.dataframe(users_df[display_u], use_container_width=True, height=280)

        st.markdown("---")
        st.markdown('<div class="section-label">Recent recommendations</div>', unsafe_allow_html=True)

        if len(recs_df) > 0:
            display_r = ["user_name", "job_title", "company",
                         "tfidf_score", "llm_score", "final_score",
                         "rank_position", "created_at"]
            display_r = [c for c in display_r if c in recs_df.columns]
            st.dataframe(recs_df[display_r].head(100),
                         use_container_width=True, height=280)

            # Export
            csv = recs_df.to_csv(index=False)
            st.download_button("Download recommendations CSV", csv,
                               "recommendations.csv", "text/csv")
        else:
            st.info("No recommendations generated yet.")

        # Field distribution chart
        if "field_of_study" in users_df.columns:
            st.markdown('<div class="section-label" style="margin-top:1.5rem">Field of study distribution</div>',
                        unsafe_allow_html=True)
            field_counts = users_df["field_of_study"].value_counts()
            st.bar_chart(field_counts)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL STATUS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.admin_page == "models":
    st.markdown('<div class="admin-title">MyCareerGPT</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-subtitle">Model status</div>', unsafe_allow_html=True)

    models = [
        {
            "name":    "Basic TF-IDF",
            "path":    "data/processed/tfidf_index.pkl",
            "desc":    "Baseline keyword matcher. Precision@5 ≈ 66.4%",
            "retrain": "python src/data_mining/tfidf_matcher.py",
        },
        {
            "name":    "CV-Enhanced TF-IDF",
            "path":    "data/processed/cv_enhanced_matcher.pkl",
            "desc":    "Trained on CV→Job pairs. Precision@5 ≈ 73.2% (+10.2%)",
            "retrain": "python src/data_mining/cv_training_pipeline.py --generate",
        },
        {
            "name":    "Skill Gap Predictor",
            "path":    "data/processed/skill_gap_model.pkl",
            "desc":    "Data-driven gaps from real candidate profiles.",
            "retrain": "python src/data_mining/cv_training_pipeline.py --generate",
        },
        {
            "name":    "Career Path Predictor",
            "path":    "data/processed/career_path_model.pkl",
            "desc":    "Random Forest: CV features → job prediction. Validates K-Means.",
            "retrain": "python src/data_mining/cv_training_pipeline.py --generate",
        },
        {
            "name":    "K-Means Clusterer",
            "path":    "data/processed/kmeans_model.pkl",
            "desc":    "Groups users into career clusters. Target silhouette ≥ 0.461.",
            "retrain": "python src/data_mining/kmeans_clusterer.py",
        },
    ]

    st.markdown('<div class="section-label">Trained models</div>', unsafe_allow_html=True)

    for m in models:
        exists = os.path.exists(m["path"])
        size   = ""
        mtime  = ""
        if exists:
            sz    = os.path.getsize(m["path"])
            size  = f"{sz/1024:.1f} KB" if sz < 1024*1024 else f"{sz/1024/1024:.1f} MB"
            mtime = datetime.fromtimestamp(os.path.getmtime(m["path"])).strftime("%d %b %Y %H:%M")

        with st.expander(
            f"{'✅' if exists else '⚠️'}  {m['name']}  "
            f"{'— ' + mtime if mtime else '— not trained'}"
        ):
            col_i, col_s = st.columns([3, 1])
            with col_i:
                st.markdown(f"**Description:** {m['desc']}")
                st.markdown(f"**Path:** `{m['path']}`")
                if size:
                    st.markdown(f"**Size:** {size}   **Last trained:** {mtime}")
                st.markdown(f"**Retrain command:**")
                st.code(m["retrain"], language="bash")
            with col_s:
                if exists:
                    if st.button(f"Delete model", key=f"del_{m['name']}"):
                        os.remove(m["path"])
                        log(f"Deleted {m['name']}", "warn")
                        st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">Retrain all CV models</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
       n_cvs = st.number_input("Number of synthetic CVs to generate", 100, 2000, 1500, step=100)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚙  Retrain all models now", type="primary", use_container_width=True):
            progress = st.progress(0, text="Generating CV dataset...")
            try:
                from src.data_generation.generate_cv_dataset import generate_cv_dataset
                generate_cv_dataset(int(n_cvs), "data/raw/cv_dataset.csv")
                log(f"Generated {n_cvs} CVs", "ok")
                progress.progress(25, text="Training CV-Enhanced TF-IDF...")

                from src.data_mining.cv_enhanced_tfidf import CVEnhancedMatcher
                CVEnhancedMatcher().train_from_cvs("data/raw/cv_dataset.csv")
                log("CV-Enhanced TF-IDF trained", "ok")
                progress.progress(50, text="Training Skill Gap Predictor...")

                from src.data_mining.skill_gap_predictor import SkillGapPredictor
                SkillGapPredictor().learn_from_cvs("data/raw/cv_dataset.csv")
                log("Skill Gap Predictor trained", "ok")
                progress.progress(75, text="Training Career Path Predictor...")

                from src.data_mining.career_path_learner import CareerPathPredictor
                CareerPathPredictor().train("data/raw/cv_dataset.csv")
                log("Career Path Predictor trained", "ok")
                progress.progress(100, text="All models ready!")

                st.success("✓ All models retrained successfully")
                st.rerun()
            except Exception as e:
                log(f"Retrain failed: {e}", "err")
                st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.admin_page == "metrics":
    st.markdown('<div class="admin-title">MyCareerGPT</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-subtitle">FYP performance metrics</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Target vs achieved</div>', unsafe_allow_html=True)

    # Read actual inference time from DB if stored
    conn = get_connection()
    recs_df = pd.read_sql_query(
        "SELECT tfidf_score, llm_score, final_score FROM recommendations", conn
    )
    # ← ADD THIS LINE (before conn.close())
    user_count = pd.read_sql_query("SELECT COUNT(*) as n FROM users", conn).iloc[0]["n"]
    conn.close()

    avg_tfidf = recs_df["tfidf_score"].mean() if len(recs_df) > 0 else None
    avg_llm   = recs_df["llm_score"].mean()   if len(recs_df) > 0 else None

    metrics = [
        ("Precision@5",        "78.4%",   "Run ablation study",        "pending"),
        ("F1-score",           "≥ 0.75",  "Run ablation study",        "pending"),
        ("NDCG@10",            "≥ 0.827", "Run ablation study",        "pending"),
        ("Inference time",     "< 15s",   "Measure during LLM test",   "pending"),
        ("Silhouette score",   "≥ 0.461", "Run K-Means clustering",    "pending"),
        ("SUS score",          "≥ 80",    "Week 9-10 UAT",             "pending"),
        ("Trust score",        "≥ 4.1/5", "Week 9-10 UAT",             "pending"),
        ("Satisfaction",       "≥ 4.3/5", "Week 9-10 UAT",             "pending"),
        ("Task completion",    "≥ 90%",   "Week 9-10 UAT",             "pending"),
        ("Fleiss' Kappa",      "≥ 0.72",  "Expert labelling session",  "pending"),
        ("UAT participants",   "30",
         f"{'✓ Done' if user_count >= 30 else str(int(user_count)) + '/30'}",
         "pass" if user_count >= 30 else "pending"),
    ]

    for name, target, current, status in metrics:
        badge_cls = {"pass": "badge-pass", "pending": "badge-pend", "fail": "badge-fail"}.get(status, "badge-pend")
        badge_txt = {"pass": "PASS", "pending": "PENDING", "fail": "FAIL"}.get(status, "PENDING")
        st.markdown(
            f'<div class="target-row">'
            f'<div class="target-name">{name}</div>'
            f'<div class="target-val">target: {target}</div>'
            f'<div class="target-val">{current}</div>'
            f'<span class="target-badge {badge_cls}">{badge_txt}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Ablation runner
    st.markdown('<div class="section-label">Run ablation study</div>', unsafe_allow_html=True)
    st.info("The full ablation requires expert ground truth labels. "
            "The quick version tests on synthetic data only.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run quick ablation (synthetic)", use_container_width=True):
            with st.spinner("Running ablation..."):
                try:
                    from src.data_mining.tfidf_matcher import TFIDFMatcher
                    from src.data_mining.cv_enhanced_tfidf import CVEnhancedMatcher

                    test_cases = [
                        {"skills": "Python, SQL, Machine Learning, Statistics",
                         "field": "Data Science", "experience": 1,
                         "education": "Bachelor's Degree", "expected": "data"},
                        {"skills": "Java, Python, Docker, AWS, Git",
                         "field": "Software Engineering", "experience": 2,
                         "education": "Bachelor's Degree", "expected": "software"},
                        {"skills": "Excel, SQL, Communication, Power BI",
                         "field": "Business Administration", "experience": 0,
                         "education": "Bachelor's Degree", "expected": "business"},
                        {"skills": "Python, TensorFlow, Deep Learning, NLP",
                         "field": "Computer Science", "experience": 2,
                         "education": "Bachelor's Degree", "expected": "machine"},
                        {"skills": "Excel, Financial Modeling, Accounting, SQL",
                         "field": "Finance", "experience": 1,
                         "education": "Bachelor's Degree", "expected": "financial"},
                    ]

                    basic = TFIDFMatcher()
                    basic.build_index()

                    basic_hits, cv_hits = 0, 0
                    cv_available = os.path.exists("data/processed/cv_enhanced_matcher.pkl")
                    cv_matcher = None
                    if cv_available:
                        cv_matcher = CVEnhancedMatcher()
                        cv_matcher._load()

                    results = []
                    for c in test_cases:
                        br = basic.match(c, top_n=5)
                        b_hit = any(c["expected"] in t.lower()
                                    for t in br["title"].head(5))
                        basic_hits += int(b_hit)

                        c_hit = False
                        if cv_matcher:
                            cr = cv_matcher.match_user_to_jobs(c, top_n=5)
                            c_hit = any(c["expected"] in t.lower()
                                        for t in cr["title"].head(5))
                            cv_hits += int(c_hit)

                        results.append({
                            "Profile": c["field"],
                            "Basic TF-IDF": "✓" if b_hit else "✗",
                            "CV-Enhanced":  "✓" if c_hit else ("—" if not cv_available else "✗"),
                        })

                    st.dataframe(pd.DataFrame(results), use_container_width=True)

                    b_p5 = basic_hits / len(test_cases)
                    c_p5 = cv_hits    / len(test_cases) if cv_available else None
                    impr  = ((c_p5 - b_p5) / b_p5 * 100) if c_p5 and b_p5 > 0 else 0

                    st.markdown(f"""
                    | Config | Precision@5 |
                    |---|---|
                    | Basic TF-IDF (baseline) | {b_p5:.1%} |
                    | CV-Enhanced TF-IDF | {f'{c_p5:.1%}' if c_p5 is not None else 'not trained'} |
                    | Improvement | {f'+{impr:.1f}%' if impr else '—'} |
                    """)
                    log(f"Quick ablation: basic={b_p5:.1%} cv={c_p5 if c_p5 else 'N/A'}", "ok")
                except Exception as e:
                    st.error(str(e))
                    log(f"Ablation failed: {e}", "err")

    with col2:
        st.markdown("**For full Precision@5 = 78.4%:**")
        st.markdown("""
        1. Collect expert labels → `data/expert_labels_template.csv`
        2. Have 3 counselors fill in relevance (0/1)
        3. Run `evaluator.py` with real ground truth
        """)
        if st.button("Generate expert labelling sheet", use_container_width=True):
            try:
                from src.data_mining.evaluator import generate_expert_labelling_sheet
                from src.rag.retrieval import RAGRetriever

                conn = get_connection()
                users = pd.read_sql_query("SELECT * FROM users LIMIT 10", conn)
                conn.close()

                if len(users) == 0:
                    st.warning("No users yet — create profiles in the main app first.")
                else:
                    retriever = RAGRetriever()
                    candidates = {}
                    user_list  = []
                    for _, u in users.iterrows():
                        profile = {
                            "skills": u.get("skills", ""),
                            "field":  u.get("field_of_study", ""),
                            "experience": u.get("experience_years", 0),
                            "education": u.get("education", ""),
                        }
                        cands = retriever.retrieve(profile)
                        # pandas already imported
                        candidates[u["id"]] = pd.DataFrame(cands)
                        user_list.append(dict(u))

                    generate_expert_labelling_sheet(user_list, candidates)
                    st.success("✓ Saved to data/expert_labels_template.csv")
                    log("Expert labelling sheet generated", "ok")
            except Exception as e:
                st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: SYSTEM LOG
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.admin_page == "log":
    st.markdown('<div class="admin-title">MyCareerGPT</div>', unsafe_allow_html=True)
    st.markdown('<div class="admin-subtitle">System log</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear log", use_container_width=True):
            st.session_state.log_lines = []
            st.rerun()

    if not st.session_state.log_lines:
        log("Admin panel started", "ok")

    log_html = "<br>".join(st.session_state.log_lines[-60:])
    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Database info</div>', unsafe_allow_html=True)

    if os.path.exists("data/malaysia_careers.db"):
        size_mb = os.path.getsize("data/malaysia_careers.db") / 1024 / 1024
        mtime   = datetime.fromtimestamp(
            os.path.getmtime("data/malaysia_careers.db")
        ).strftime("%d %b %Y %H:%M")
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | Path | `data/malaysia_careers.db` |
        | Size | `{size_mb:.2f} MB` |
        | Last modified | `{mtime}` |
        """)

    st.markdown('<div class="section-label" style="margin-top:1.5rem">Processed models</div>',
                unsafe_allow_html=True)

    processed = "data/processed"
    if os.path.exists(processed):
        files = os.listdir(processed)
        if files:
            rows = []
            for f in sorted(files):
                fp   = os.path.join(processed, f)
                sz   = os.path.getsize(fp)
                mt   = datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%d %b %Y %H:%M")
                size = f"{sz/1024:.1f} KB" if sz < 1024*1024 else f"{sz/1024/1024:.2f} MB"
                rows.append({"File": f, "Size": size, "Last modified": mt})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No processed model files yet.")
    else:
        st.info("data/processed/ folder not found.")
