
---

# Comment Intelligence (TikTok)

A small full-stack app that **scrapes TikTok comments** and turns them into **actionable insights** with **sentiment**, **topics**, and **hybrid RAG search** — powered by a **FastAPI** backend and a **Streamlit** UI.

> **Stack:** FastAPI · Streamlit · PostgreSQL + pgvector · LangChain · BERTopic · SentenceTransformers
> **Flow:** Scrape TikTok → Preprocess (ID) → Sentiment → Topic Modeling → Hybrid RAG → Save to DB & Files

---

## 🔎 Product Brief (at a glance)

* **Description:** A small full-stack app that scrapes TikTok comments and turns them into actionable insights with sentiment, topics, and hybrid RAG search.
* **Background:** Manually reading social comment sections is noisy, repetitive, and slow—especially for Indonesian content.
* **Goal:** Give analysts and marketers a one-click way to understand what audiences are saying about any TikTok post.
* **What it does:** Scrapes, cleans, runs sentiment + topic modeling, indexes to pgvector/BM25, and answers ad-hoc questions over the comments.
* **What we achieved:** An end-to-end Dockerized pipeline with per-comment outputs, downloadable CSV/JSON/TXT, and a simple Streamlit UI.
* **Tech/Models:** IndoBERT for sentiment, BERTopic + Indonesian Sentence-BERT for topics, PGVector + BM25 for search, and optional Qwen for generation.
* **Why it matters:** Cuts analysis time from hours to minutes and backs every answer with traceable snippets and scores.
* **Limitations:** Relies on Apify access, short/noisy user texts, and model bias; RAG quality dips when comments are scarce.
* **What can be improved:** Add YouTube/Instagram sources, better dedup/emoji handling, richer dashboards, user auth, and fine-tuned local models.
* **Next steps:** Schedule auto-crawls, compare time ranges, alert on topic shifts, and export insights to Slack/Sheets.
* **Who it’s for:** Social teams, agencies, product research, and anyone tracking brand chatter at scale.
* **Success metrics:** Faster turnaround, broader coverage per campaign, clearer trend lines for topics and sentiment.
* **Risk/Compliance:** Respect platform ToS, avoid storing PII, enable data removal on request.

---

## ✨ Screenshots
**Flowchart**  
<img width="1000" alt="flowchart-scraper_tiktok" src="https://github.com/user-attachments/assets/b3caafef-2f2d-4b68-87c8-40dfe3756232" />

###**1) Pipeline Analysis Page**  
<img src="https://github.com/user-attachments/assets/d989bc6d-8915-4c27-a452-9a42499e6fd4" alt="Pipeline Analysis Page" width="1000" />

<br>

###**2) Sentiment Page**  
<img src="https://github.com/user-attachments/assets/0ee504ff-0170-46bd-abfb-8b3a0f695832" alt="Sentiment Page" width="1000" />

<br>

###**3) RAG Query Page**  
<img src="https://github.com/user-attachments/assets/d04c2b52-e045-440e-a34c-23ddb43fdc4a" alt="RAG Query Page" width="1000" />


---
---
 ## 🖥️ How to Use

All step-by-step usage and workflow documentation is maintained in a living Canva guide:

➡️ **Open the workflow documentation:** [https://www.canva.com/design/DAG0OuHTXqU/EWu5_bEazOY4LEVMfejQnw/edit?utm_content=DAG0OuHTXqU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton](https://www.canva.com/design/DAG0OuHTXqU/qtNWMKDiSiNRc2qh21rgnw/view?utm_content=DAG0OuHTXqU&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h7390eb4606)

**Inside the guide**
- **Program flow simulations**
- **Program outputs**
- **Short descriptions**
- 
## 🚀 Features

* **TikTok Scraper (Apify):** pull comments by video URL
* **Indonesian Text Preprocessing:** clean URLs, mentions, emojis, slang, etc.
* **Sentiment (IndoBERT):** `positive / neutral / negative` + confidence
* **Topic Modeling (BERTopic):** topic labels via Sentence-BERT embeddings
* **Hybrid RAG:** BM25 (lexical) + `pgvector` (semantic); optional Qwen to draft answers
* **Persistence:** store in **PostgreSQL** and export **CSV / JSON / TXT**
* **CPU-friendly:** runs without a GPU

---

## 🧱 Tech Stack

* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Database:** PostgreSQL 14+ with **pgvector**
* **Orchestration & LLM:** LangChain
* **Models:** BERTopic, SentenceTransformers, IndoBERT
* **Deployment:** Docker & Docker Compose

---

## 📋 Prerequisites

* **Option A (recommended):** Docker & Docker Compose
* **Option B (local):** Python 3.11+, PostgreSQL 14+ with `pgvector` enabled
* **API keys:**

  * `APIFY_API_TOKEN` (required) — scrape TikTok comments
  * `QWEN_API_KEY` (optional) — LLM answers (OpenAI-compatible Qwen)

---

## ⚡ Quickstart (Docker)

1. **Create your `.env`**

```bash
cp .env.example .env
```

2. **Fill it like this:**

```dotenv
# ==== Backend ====
LOG_LEVEL=INFO
DATA_DIR=/app/data
# "", AUTO (timestamp), or your custom label
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
```

> ⚠️ Do **not** commit `.env` to public repositories.

3. **Run the stack**

```bash
docker compose up -d --build
```

Open:

* **Backend (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Frontend (Streamlit):** [http://localhost:8601](http://localhost:8601)

Logs:

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f db
```

Rebuild a stale backend:

```bash
docker compose build --no-cache backend
docker compose up -d --force-recreate backend
```

---

## 🧑‍💻 Local Development (No Docker)

> Ensure PostgreSQL is running and **pgvector** exists:
>
> ```sql
> CREATE EXTENSION IF NOT EXISTS vector;
> ```

Create venv & install:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Start **backend**:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Start **frontend** (new terminal):

```bash
streamlit run frontend/Home.py --server.port 8601
```

---

## 🗂️ Project Structure

```text
.
├── backend/
│   ├── api/main.py               # FastAPI routes
│   ├── graph/pipeline.py         # scrape → preprocess → sentiment → topic → persist
│   ├── modules/
│   │   ├── scraper.py            # Apify TikTok scraper
│   │   ├── preprocessing.py      # IndonesianPreprocessor
│   │   ├── sentiment.py          # IndoBERT sentiment
│   │   ├── topic.py              # BERTopic + SentenceTransformer
│   │   ├── rag.py                # Hybrid RAG (BM25 + pgvector + optional Qwen)
│   │   └── storage.py            # Save CSV/JSON/TXT to DATA_DIR
│   └── utils/db.py               # DB init + CRUD + vector ops
│
├── frontend/
│   ├── pages/Pipeline_Analisis.py # Run pipeline, view insight, download artifacts
│   ├── pages/rag.py               # RAG search & answer
│   └── Home.py                    # Streamlit landing
│
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```
---

## 🖥️ Using the App

### Pipeline_Analisis (Streamlit)

1. Open **Pipeline_Analisis** from the sidebar.
2. Paste TikTok **video URL**, set **Content ID**, **Content Date**, **Max Comments**.
3. Click **Run Pipeline**.

You’ll get:

* **Insight Summary:** dominant topics & sentiment distribution
* **Comments Table:** text, sentiment, confidence, topic label, date
* **Downloads:** JSON / CSV / TXT insight

### RAG (Streamlit)

1. Open **RAG** page.
2. Ask a question (e.g., “What do people think about product X?”).
3. See:

   * **Answer:** generated by Qwen if `QWEN_API_KEY` is set; otherwise only sources
   * **Sources:** top snippets with topic, sentiment, and hybrid scores

---

## 🔌 API (FastAPI)

### `POST /analyze` — run full pipeline

**Request**

```json
{
  "video_url": "https://www.tiktok.com/@user/video/123",
  "content_id": "indomie",
  "content_date": "2025-09-27",
  "max_comments": 50
}
```

**Response (example, truncated)**

```json
{
  "insight": { "summary": "..." },
  "merged_comments_count": 50,
  "merged_comments": [
    {
      "document_id": "indomie",
      "text": "....",
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
    "answer": "LLM summary (if enabled)",
    "sources": [{ "snippet": "...", "rank": 1 }]
  }
}
```

---

### `POST /rag/query` — query Hybrid RAG

**Request**

```json
{
  "query": "What is the audience's main opinion about X?",
  "k": 3
}
```

**Response (example, truncated)**

```json
{
  "answer": "Short answer with [1] [2] when relevant",
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
```

---

### `GET /files/{type}/{content_id}.{ext}` — download latest artifact

Examples:

```
/files/comments/indomie.csv
/files/insight/indomie.txt
```

---

## 🧪 Troubleshooting

* **RAG shows “[No LLM]”**
  `QWEN_API_KEY` not set. Search + sources still work.

* **Backend AttributeError / stale code**
  Rebuild backend:

  ```bash
  docker compose build --no-cache backend
  docker compose up -d --force-recreate backend
  ```

* **Apify returns 0 comments**
  Try a different video, verify actor permissions, increase `max_comments`.

* **404 on file download**
  Ensure artifacts exist in `DATA_DIR` inside backend (`/app/data` by default).

* **pgvector missing**
  Enable the extension in Postgres:

  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

* **Windows port already in use**
  Stop old containers or adjust host ports in `docker-compose.yml`.

---

## 🧭 Roadmap

* Add YouTube/Instagram sources
* Better dedup & emoji handling
* Richer dashboards and topic drift charts
* User authentication
* Fine-tuned/local models
* Scheduled crawls & Slack/Sheets export

---

## 🤝 Contributing

PRs welcome! Keep docs friendly, changes focused, and diffs small.

---

## 📜 License

MIT — see `LICENSE`.

---

## 🙏 Acknowledgements

Apify · pgvector · BERTopic · SentenceTransformers · LangChain · Streamlit · FastAPI

---

**Happy shipping!** If this helps, ⭐ the repo and share what you build.
