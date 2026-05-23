# 🇲🇾 MalaysiaCareerGPT

An AI-powered career recommendation system for Malaysian graduates, combining CV-Enhanced TF-IDF matching with Google Gemini 2.5 Flash LLM reranking. Built with Streamlit, OpenGauss (Docker), and a Random Forest career path predictor trained on 1,500 real Malaysian CVs.

---

## 📁 Project Structure

```
MalaysiaCareerGPT/
├── src/
│   ├── app/
│   │   └── app.py                        ← Main Streamlit application
│   ├── llm/
│   │   └── llm_interface.py              ← Google Gemini API interface
│   ├── rag/
│   │   └── retrieval.py                  ← RAG pipeline & prompt builder
│   ├── data_mining/
│   │   ├── tfidf_matcher.py              ← Basic TF-IDF job matcher
│   │   ├── cv_enhanced_tfidf.py          ← CV-trained TF-IDF matcher
│   │   ├── skill_gap_predictor.py        ← ML-based skill gap predictor
│   │   ├── career_path_learner.py        ← Random Forest career predictor
│   │   └── cv_training_pipeline.py       ← Training pipeline runner
│   ├── database.py                       ← Database manager (OpenGauss/SQLite)
│   └── resume_parser.py                  ← PDF/DOCX resume parser
├── data/
│   ├── raw/                              ← Place your jobs CSV here
│   └── processed/                        ← Generated .pkl model files
├── docker-compose.yml                    ← Docker setup for OpenGauss
├── .env                                  ← Your environment variables (not in git)
├── .env.example                          ← Template for .env
├── requirements.txt                      ← Python dependencies
└── README.md
```

---

## 📄 What Each File Does

| File | Purpose |
|------|---------|
| `src/app/app.py` | Main Streamlit UI — profile wizard, recommendations page, ATS resume generator, feedback page with SUS questionnaire |
| `src/llm/llm_interface.py` | Sends prompts to Google Gemini 2.5 Flash API, parses structured recommendations, checks for hallucinations |
| `src/rag/retrieval.py` | RAG pipeline — retrieves top-20 candidates via TF-IDF, enriches with skill gaps, builds the LLM prompt |
| `src/database.py` | Database manager — supports both OpenGauss (Docker) and SQLite, handles all CRUD operations and feedback storage |
| `src/resume_parser.py` | Parses uploaded PDF/DOCX resumes to extract name, email, skills, education, work history and projects |
| `src/data_mining/cv_enhanced_tfidf.py` | CV-Enhanced TF-IDF matcher trained on 1,500 Malaysian CVs — primary matching engine |
| `src/data_mining/skill_gap_predictor.py` | Predicts skill gaps using ML models trained on job requirements data |
| `src/data_mining/career_path_learner.py` | Random Forest classifier that predicts career path (98.2% CV accuracy) |
| `src/data_mining/cv_training_pipeline.py` | One-time training script that generates all `.pkl` model files |
| `src/data_mining/tfidf_matcher.py` | Basic TF-IDF fallback matcher used when CV-enhanced model is not available |

---

## ⚙️ Requirements

### Python Version
```
Python 3.9 or higher
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
streamlit
pandas
numpy
scikit-learn
python-dotenv
google-genai
reportlab
pdfplumber
python-docx
psycopg2-binary
requests
```

### Additional Tools
- **Docker Desktop** — for running OpenGauss database
- **Anaconda** (recommended) — for environment management

---

## 🗄️ Database Options

You have two options — **SQLite** (simple, no setup) or **OpenGauss** (production-grade, requires Docker).

---

### Option A — SQLite (Simple, No Docker Needed)

Best for: local development and testing

**Step 1** — In your `.env` file, leave OpenGauss fields empty:
```
GOOGLE_API_KEY=your_gemini_api_key_here

OPENGAUSS_HOST=localhost
OPENGAUSS_PORT=5432
OPENGAUSS_DB=malaysia_careers
OPENGAUSS_USER=
OPENGAUSS_PASSWORD=
```

**Step 2** — The app will automatically use SQLite at `data/malaysia_careers.db`. No extra setup needed.

---

### Option B — OpenGauss in Docker (Recommended for Production/Demo)

Best for: UAT testing, demos, FYP presentation

**Step 1** — Install Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

**Step 2** — Choose a strong password for your OpenGauss database. It must meet these rules:
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character (e.g. `@`, `!`, `#`)

Example of a valid password: `MyCareer@2025`

**Step 3** — Update `docker-compose.yml` with your chosen password:
```yaml
environment:
  GS_PASSWORD: MyCareer@2025    ← change this to your password
```

**Step 4** — Update your `.env` file with the same password:
```
GOOGLE_API_KEY=your_gemini_api_key_here

OPENGAUSS_HOST=localhost
OPENGAUSS_PORT=5432
OPENGAUSS_DB=malaysia_careers
OPENGAUSS_USER=gaussdb
OPENGAUSS_PASSWORD=MyCareer@2025    ← must match docker-compose.yml
```

**Step 5** — Start OpenGauss container:
```bash
docker compose up -d
```

**Step 6** — Create the database inside OpenGauss (first time only):
```bash
docker exec -it mycareergpt-db bash -c "LD_LIBRARY_PATH=/usr/local/opengauss/lib /usr/local/opengauss/bin/gsql -U gaussdb -d postgres -c 'CREATE DATABASE malaysia_careers;'"
```
Enter your password when prompted.

---

## 🔑 Getting Your Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key and paste it into your `.env` file

---

## 📊 Dataset
This project uses the Jobstreet Malaysia Job Dataset hosted on Kaggle. You must download this dataset before setting up the database.

🔗 Download Link: [Jobstreet All Job Dataset by Azrai Mohamad](https://www.kaggle.com/datasets/azraimohamad/jobstreet-all-job-dataset)

Once downloaded, extract the ZIP file and place the jobstreet_all_job_dataset.csv file directly into the data/raw/ directory of this project.

---

## 🚀 How to Run the Project

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/MalaysiaCareerGPT.git
cd MalaysiaCareerGPT
```

### Step 2 — Create Your `.env` File
```bash
cp .env.example .env
```
Then fill in your values (see Database Options above).

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set Up the Database

**If using OpenGauss**, start Docker first:
```bash
docker compose up -d
```
Then create the database (first time only — see Step 6 in Option B above).

**Then run** (both SQLite and OpenGauss):
```bash
python src/database.py
python src/database.py --load data/raw/jobstreet_all_job_dataset.csv
```

### Step 5 — Train the ML Models (First Time Only)
```bash
python src/data_mining/cv_training_pipeline.py --generate
```
This creates all `.pkl` files in `data/processed/`. Skip this step if the `.pkl` files already exist.

### Step 6 — Run the App
```bash
streamlit run src/app/app.py
```
Open your browser at `http://localhost:8501`

---

## 🔄 Every Day Workflow

```bash
# Start Docker (if using OpenGauss)
docker compose up -d

# Activate conda environment
conda activate your_env_name

# Navigate to project folder
cd path/to/MalaysiaCareerGPT

# Run the app
streamlit run src/app/app.py

# Before stopping — shut down Docker
docker compose down
```

---

## 📋 Environment Variables Reference

Create a `.env` file in the project root with these variables:

```
# ── Google Gemini API ─────────────────────────────────────────────
GOOGLE_API_KEY=your_gemini_api_key_here

# ── OpenGauss Database (leave blank to use SQLite) ────────────────
OPENGAUSS_HOST=localhost
OPENGAUSS_PORT=5432
OPENGAUSS_DB=malaysia_careers
OPENGAUSS_USER=gaussdb
OPENGAUSS_PASSWORD=your_strong_password_here
```

## 👤 Author

Muhammad Affiq Abqari bin Syafiq Jasrin — FYP02-DS-T2610-0382 | Multimedia University (MMU)
