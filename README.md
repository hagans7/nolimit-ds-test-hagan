#Comment Intelligence (TikTok)

A small full-stack project that scrapes TikTok comments, cleans them (Indonesian-aware), runs sentiment + topic modeling, and lets you query with Hybrid RAG (BM25 + pgvector) — all with a FastAPI backend and a Streamlit UI.

Stack: FastAPI · Streamlit · PostgreSQL + pgvector · LangChain · BERTopic · SentenceTransformers
Flow: Scrape TikTok → Preprocess (ID) → Sentiment → Topic Modeling → Hybrid RAG → Save to DB & Files

##✨ Screenshots

Replace these placeholders with your own screenshots.

Pipeline Analysis Page	RAG Query Page

	
##🚀 Features

TikTok Scraper (Apify): Pulls comments by video URL

Indonesian Text Preprocessing: removes URLs, mentions, emojis, slang, etc.

Sentiment Analysis (IndoBERT): positive / neutral / negative + confidence

Topic Modeling (BERTopic): topic labels using SentenceTransformers embeddings

Hybrid RAG: BM25 (lexical) + pgvector (semantic), optional LLM (Qwen) to draft answers

Persistence: saves analysis to PostgreSQL and artifacts to CSV/JSON/TXT

CPU-friendly by default: works fine without a GPU

##🧱 Tech Stack

Backend: FastAPI

Frontend: Streamlit

Database: PostgreSQL 14+ with pgvector

Orchestration & LLM: LangChain

Models: BERTopic, SentenceTransformers, IndoBERT

Deployment: Docker & Docker Compose

##📋 Prerequisites

Option A (recommended): Docker & Docker Compose

Option B (local): Python 3.11+, PostgreSQL 14+ with pgvector extension enabled

API keys:

APIFY_API_TOKEN (required) – to scrape TikTok comments

QWEN_API_KEY (optional) – to generate RAG answers with Qwen (OpenAI-compatible)

##⚙️ Setup
1) Environment Variables

Copy the example and fill it:

cp .env.example .env


###.env example

# ==== Backend ====
LOG_LEVEL=INFO
DATA_DIR=/app/data
# "" (no suffix), AUTO (timestamp), or your custom label:
SAVE_TS_SUFFIX=AUTO

# TikTok scraping
APIFY_API_TOKEN=apify_xxxxxxxxxxxxxxxxxxxxxxx

# Postgres/pgvector (service name follows docker-compose)
PGVECTOR_URL=postgresql://user:pass@doc_pgvector:5432/appdb
PGVECTOR_COLLECTION=comments

# Qwen (optional, OpenAI-compatible)
QWEN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# ==== Frontend ====
BACKEND_URL=http://backend:8000


##⚠️ Security: Don’t commit .env to public repos.

2) Run with Docker (Recommended)
docker compose up -d --build


Access:

Backend (Swagger UI): http://localhost:8000/docs

Frontend (Streamlit): http://localhost:8601

Tail logs:

docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f db


Rebuild a stale backend:

docker compose build --no-cache backend
docker compose up -d --force-recreate backend

3) Run Locally (No Docker)

Make sure PostgreSQL is running and pgvector is enabled:

CREATE EXTENSION IF NOT EXISTS vector;


Create venv & install deps:

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt


Export the same variables from .env.

Start backend:

uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload


Start frontend (another terminal):

streamlit run frontend/Home.py --server.port 8601

##🗂️ Project Structure
.
├── backend/
│   ├── api/main.py               # FastAPI routes
│   ├── graph/pipeline.py         # Orchestrates: scrape → preprocess → sentiment → topic → persist
│   ├── modules/
│   │   ├── scraper.py            # Apify TikTok scraper
│   │   ├── preprocessing.py      # IndonesianPreprocessor
│   │   ├── sentiment.py          # IndoBERT sentiment
│   │   ├── topic.py              # BERTopic + SentenceTransformer
│   │   ├── rag.py                # Hybrid RAG (BM25 + pgvector + optional Qwen)
│   │   └── storage.py            # Save artifacts (CSV/JSON/TXT) to DATA_DIR
│   └── utils/db.py               # DB init + CRUD for comments/topics/vectors
│
├── frontend/
│   ├── pages/Pipeline_Analisis.py # Run pipeline, view insight, download CSV/JSON/TXT
│   ├── pages/rag.py               # RAG search & answer view
│   └── Home.py                    # Streamlit landing page
│
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md

##🧭 How It Works (Data Flow)
flowchart LR
  A[Input TikTok URL] --> B[Scraper (Apify)]
  B --> C[Indonesian Preprocess]
  C --> D[Sentiment (IndoBERT)]
  D --> E[Topic Modeling (BERTopic)]
  E --> F[Persist → Postgres + pgvector]
  F --> G[Artifacts CSV/JSON/TXT]
  F --> H[Hybrid RAG (BM25 + Vector)]
  H --> I[LLM Answer (Qwen, optional)]
  I --> J[Streamlit UI]

##🖥️ Using the App
1) Pipeline_Analisis Page

Open Pipeline_Analisis (sidebar).

Paste TikTok video URL, set a unique Content ID, Content Date, and Max Comments.

Click Run Pipeline.

You’ll see:

Insight Summary: dominant topics & sentiment distribution

Comments Table: text, sentiment, confidence, topic label, date

Downloads: JSON / CSV / TXT (insight)

2) RAG Page

Open RAG (sidebar).

Ask a question (e.g., “What do people think about product X?”).

You’ll get:

Answer: summarized with Qwen (if QWEN_API_KEY set; otherwise just sources)

Sources: top comment snippets + metadata (topic, sentiment, hybrid score)

🔌 API (FastAPI)
POST /analyze

Run the full pipeline (scrape → analyze → persist → artifacts → optional RAG).

Request

{
  "video_url": "https://www.tiktok.com/@user/video/123",
  "content_id": "indomie",
  "content_date": "2025-09-27",
  "max_comments": 50
}


Response (truncated)

{
  "insight": { "...": "..." },
  "merged_comments_count": 50,
  "merged_comments": [
    {
      "document_id": "indomie",
      "text": "...",
      "sentiment": "positive",
      "confidence": 0.98,
      "topic_label": "rasa - enak - pedas",
      "date": "2025-09-27"
    }
  ],
  "artifacts": {
    "json": "comments_indomie_20250928-011234.json",
    "csv": "comments_indomie_20250928-011234.csv",
    "insight_txt": "insight_indomie_20250928-011234.txt"
  },
  "rag": {
    "answer": "A summary answer from the LLM...",
    "sources": [{ "...": "..." }]
  }
}

POST /rag/query

Query the Hybrid RAG index.

Request

{
  "query": "What is the audience's main opinion about X?",
  "k": 3
}


Response (truncated)

{
  "answer": "Short answer with [1] [2] citations when relevant",
  "sources": [
    {
      "rank": 1,
      "snippet": "top context ...",
      "document_id": "indomie",
      "topic_label": "rasa - enak - pedas",
      "sentiment": "positive",
      "score_final": 0.75
    }
  ]
}

GET /files/{type}/{content_id}.{ext}

Download the latest analysis file.

Examples:

/files/comments/indomie.csv
/files/insight/indomie.txt

##🧪 Tips & Troubleshooting

RAG shows “[No LLM]”
You didn’t set QWEN_API_KEY. That’s okay—search + sources still work.

AttributeError / stale code
Rebuild backend and clear caches:
docker compose build --no-cache backend && docker compose up -d --force-recreate backend

Apify returns 0 comments
Try another video, verify actor permissions, or increase max_comments.

404 when downloading files
Check that artifacts exist under DATA_DIR inside the backend container (/app/data by default).

pgvector not found
Ensure CREATE EXTENSION vector; on your Postgres.

Windows + Docker Desktop hiccups
If ports are busy, stop old containers or change host ports in docker-compose.yml.

##🧑‍💻 Development Notes

Keep Streamlit expanders flat (avoid nested expanders).

Use CPU by default; no special CUDA setup required.

For long pipelines, prefer idempotent saves (timestamped filenames via SAVE_TS_SUFFIX=AUTO).

##🤝 Contributing

PRs welcome!
Keep docs friendly for first-timers, avoid huge diffs, and prefer small focused changes.

##📜 License

MIT — see LICENSE.

##🙏 Acknowledgements

Apify
 for scraping infra

pgvector
 for vector search in Postgres

BERTopic
 & SentenceTransformers

LangChain
 for orchestration helpers

Streamlit
 & FastAPI

Happy shipping! If you use this in the wild, drop a star ⭐ and share what you built.
