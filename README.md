# 🛒 Blinkit Growth Discovery Engine

An AI-powered Product Management (PM) discovery platform tailored for **Blinkit** (Quick-Commerce). The engine automatically collects, filters, and extracts actionable product growth insights, friction barriers, user behaviors, and prioritized feature requirements from customer reviews across **Google Play Store**, **Apple App Store**, **Reddit**, and custom CSV uploads using **Cloudflare Workers AI**, **Groq**, or **OpenRouter**.

---

## 🌟 Key Features

- 📥 **Multi-Channel Review Collection**: Scrape & parse user feedback from Google Play, App Store, Reddit, and custom CSV/JSON uploads.
- ⚡ **AI-Powered Relevance Filtering**: Filter out generic spam/one-word ratings and extract rich, actionable user feedback.
- 📊 **Structured Insights Extraction**: Theme taxonomy mapping (Delivery Speed, Item Accuracy, Out-of-Stock, Refund/Support, UI/UX Friction, Pricing/Charges).
- 💡 **Strategic Artifact Generation**: Automatically generates JSON & Markdown artifacts including:
  - Theme Summary Report (`theme_summary.md`)
  - Executive Discovery Report (`report.md`)
  - Structured Datasets (`behaviors.json`, `barriers.json`, `requirements.json`, `root_causes.json`, `opportunities.json`, `hypotheses.json`, `interview_plans.json`)
- 🎨 **Executive Streamlit Dashboard**: Dark mode UI styled in Blinkit brand colors (`#F7C600` / `#0C831F`) with interactive charts, real-time log streaming, and downloadable ZIP export bundles.
- 🔐 **Production Ready & Secure**: Built-in environment configuration with zero hardcoded API keys and full Streamlit Cloud support (`st.secrets`).

---

## 🏗️ Repository Architecture

```
blinkit-growth-discovery/
├── .streamlit/
│   └── config.toml          # Streamlit theme & server configuration
├── analysis/
│   ├── analyze_reviews.py   # AI thematic analysis pipeline
│   ├── filter_reviews.py    # Review relevance filtering logic
│   ├── llm_client.py        # LLM retry/batch execution wrapper
│   ├── llm_provider.py      # Cloudflare, Groq & OpenRouter provider drivers
│   ├── schema.py            # Pydantic/Data schemas for insights
│   └── theme_summary.py     # Artifact generator & aggregator
├── reviews/
│   ├── app_store.py         # Apple App Store reviewer scraper
│   ├── google_play.py       # Google Play Store reviewer scraper
│   ├── reddit.py            # Reddit discussion thread collector
│   ├── upload.py            # CSV / JSON / Manual text review parser
│   └── utils.py             # Data cleaning helpers
├── utils/
│   └── export.py            # Zip export packager
├── config.py                # Global settings & st.secrets handler
├── streamlit_app.py         # Main Streamlit Web Application
├── requirements.txt         # Production dependencies
├── .env.example             # Environment variable template
└── .gitignore               # Secrets and cache exclusion manifest
```

---

## 🚀 Quickstart (Local Setup)

### 1. Prerequisites
- Python 3.9+ installed.

### 2. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/ripunjaynarula/blinkit-growth-discovery.git
cd blinkit-growth-discovery
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your AI provider credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
LLM_PROVIDER=cloudflare # Or 'groq' / 'openrouter'

# Cloudflare Workers AI Credentials (Preferred)
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_API_TOKEN=your_api_token_here

# Groq Credentials (Optional Backup)
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Streamlit Application
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your web browser.

---

## ☁️ Deploying to Streamlit Cloud

To host this application for free on **Streamlit Cloud**:

1. Push this repository to your GitHub account: `ripunjaynarula/blinkit-growth-discovery`.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app** and select:
   - **Repository**: `ripunjaynarula/blinkit-growth-discovery`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Expand **Advanced Settings** -> **Secrets** and paste your environment secrets in TOML format:

```toml
LLM_PROVIDER = "cloudflare"

CLOUDFLARE_ACCOUNT_ID = "your_cloudflare_account_id_here"
CLOUDFLARE_API_TOKEN = "your_cloudflare_api_token_here"
CLOUDFLARE_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

# Optional backup
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

5. Click **Deploy!** Your app will be live with full AI insights extraction capabilities.

---

## 🛡️ Security & Secrets Guarantee

- This codebase is configured with strict `.gitignore` rules preventing `.env` and local credentials from ever being committed or pushed.
- All API keys are resolved dynamically at runtime using `config.get_env_var()`, which prioritizes `st.secrets` on Streamlit Cloud and falls back safely to environment variables.

---

## 📄 License
MIT License. Created for Blinkit Growth & PM Discovery.
