# backend/api/main.py
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from backend.graph.pipeline import get_compiled_graph
from backend.modules.sentiment import SentimentAnalyzer
from backend.modules.rag import HybridRAG
from backend.utils.db import init_all
from backend.utils.db import migrate_vector_metadata_to_jsonb
from pathlib import Path
from backend.utils.db import init_tables
from datetime import datetime
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="Comment Intelligence Backend", version="1.0.0")

# CORS agar Streamlit bisa akses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sesuaikan jika perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals / singletons ---
GRAPH = get_compiled_graph()
SENTI = SentimentAnalyzer()  # untuk input manual
RAG = HybridRAG(
    pgvector_url=os.getenv("PGVECTOR_URL", ""),
    collection=os.getenv("PGVECTOR_COLLECTION", "comments"),
    qwen_api_key=os.getenv("QWEN_API_KEY", ""),
    qwen_model="qwen-flash",
)

# ---------- Schemas ----------


class AnalyzeRequest(BaseModel):
    video_url: str
    content_id: str
    content_date: Optional[str] = None  # "YYYY-MM-DD"
    max_comments: int = Field(default=50, ge=1, le=500)


class SentimentRequest(BaseModel):
    texts: List[str]


class RagQuery(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)

# ---------- Utilities ----------


def _invoke_graph(req: AnalyzeRequest) -> Dict[str, Any]:
    state = {
        "video_url": req.video_url,
        "content_id": req.content_id,
        "content_date": req.content_date,
        "max_comments": req.max_comments,
        "apify_token": os.getenv("APIFY_API_TOKEN", ""),
        "qwen_api_key": os.getenv("QWEN_API_KEY", ""),
        "pgvector_url": os.getenv("PGVECTOR_URL", ""),
        "pgvector_collection": os.getenv("PGVECTOR_COLLECTION", "comments"),
    }

    result = GRAPH.invoke(state) or {}

    # ---------- Insight (selalu ada) ----------
    ins = result.get("insight")
    if ins is None:
        content_id = state["content_id"]
        date = state.get("content_date") or datetime.now().strftime("%Y-%m-%d")
        insight_dict = {
            "content_id": content_id,
            "date": date,
            "total_comments": len(result.get("comments_merged", []) or []),
            "num_topics": 0,
            "dominant_topic": "N/A",
            "dominant_topic_percentage": 0.0,
            "topic_details": [],
            "summary": "Tidak ada insight yang berhasil dihasilkan (pipeline berhenti lebih awal).",
        }
    else:
        # ins adalah dataclass ContentInsight
        insight_dict = {
            "content_id": getattr(ins, "content_id", state["content_id"]),
            "date": getattr(ins, "date", state.get("content_date") or datetime.now().strftime("%Y-%m-%d")),
            "total_comments": getattr(ins, "total_comments", 0),
            "num_topics": getattr(ins, "num_topics", 0),
            "dominant_topic": getattr(ins, "dominant_topic", "N/A"),
            "dominant_topic_percentage": getattr(ins, "dominant_topic_percentage", 0.0),
            "topic_details": getattr(ins, "topic_details", []) or [],
            "summary": getattr(ins, "summary", "") or "",
        }

    # ---------- Daftar per-komen (dinormalisasi) ----------
    raw_comments = result.get("comments_merged") or []
    default_date = state.get(
        "content_date") or datetime.now().strftime("%Y-%m-%d")

    merged_comments: List[Dict[str, Any]] = []
    for c in raw_comments:
        merged_comments.append({
            "document_id": (c.get("document_id") if isinstance(c, dict) else state["content_id"]) or state["content_id"],
            "text": c.get("text", "") if isinstance(c, dict) else str(c),
            "sentiment": c.get("sentiment", "unknown") if isinstance(c, dict) else "unknown",
            "confidence": float(c.get("confidence", 0.0)) if isinstance(c, dict) else 0.0,
            "topic_label": c.get("topic_label", "Lainnya") if isinstance(c, dict) else "Lainnya",
            "date": (c.get("date") if isinstance(c, dict) else None) or default_date,
        })

    # ---------- RAG (selalu dict, tidak None) ----------
    rag_answer = result.get("rag_answer") or (
        "RAG belum tersedia (indeks kosong atau tahap RAG dilewati)." if not merged_comments
        else "Ringkasan belum dibuat."
    )
    rag_contexts = result.get("rag_contexts") or []
    rag_sources = result.get("rag_sources") or []

    return {
        "insight": insight_dict,
        "merged_comments_count": len(merged_comments),
        "merged_comments": merged_comments,
        "artifacts": result.get("artifacts", {}) or {},
        "rag": {
            "answer": rag_answer,
            "contexts": rag_contexts,
            "sources": rag_sources,
        },
    }


# ---------- Routes ----------


@app.on_event("startup")
def _startup():
    init_tables()
    try:
        migrate_vector_metadata_to_jsonb()
    except Exception as e:
        logger.warning(f"Skip JSONB migration on startup: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """End-to-end: scrape → preprocess → sentiment → topic → RAG → persist (DB + files)."""
    return _invoke_graph(req)


@app.post("/sentiment/predict")
def predict_sentiment(req: SentimentRequest):
    """Prediksi sentiment untuk input manual user (digunakan Page 2 Streamlit)."""
    results = SENTI.run(req.texts)
    return [
        {
            "text": r.text,
            "sentiment": r.sentiment,
            "confidence": r.confidence,
            "scores": r.scores,
            "processed_text": r.processed_text,
        }
        for r in results
    ]


# BASE_DIR = Path(".").resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _safe_id(s: str) -> str:
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9_-]+", "_", s or "").strip("_")
    return s[:64] or "content"

def _find_latest(pattern: str) -> Optional[Path]:
    # Return newest file matching pattern, or None
    matches = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _latest(globpat: str) -> Optional[Path]:
    files = list(DATA_DIR.glob(globpat))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


@app.get("/files/comments/{content_id}.csv")
def dl_comments_csv(content_id: str):
    f = _latest(f"comments_{content_id}*.csv")
    if not f:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, media_type="text/csv", filename=f.name)


@app.get("/files/comments/{content_id}.json")
def dl_comments_json(content_id: str):
    f = _latest(f"comments_{content_id}*.json")
    if not f:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, media_type="application/json", filename=f.name)


@app.get("/files/insight/{content_id}.txt")
def dl_insight_txt(content_id: str):
    f = _latest(f"insight_{content_id}*.txt")
    if not f:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, media_type="text/plain", filename=f.name)

@app.post("/admin/migrate-jsonb")
def admin_migrate_jsonb():
    try:
        migrate_vector_metadata_to_jsonb()
        return {"status": "ok", "message": "Vector metadata migrated to JSONB (if needed)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query")
def rag_query(req: RagQuery):
    try:
        if hasattr(RAG, "search_and_answer"):
            answer, contexts, sources = RAG.search_and_answer(
                req.query, k=req.k)
        else:
            # fallback versi lama
            ctx = RAG.retrieve(req.query, k=req.k)
            answer, sources = RAG.generate(req.query, ctx)
            contexts = [getattr(c, "text", str(c)) for c in ctx]

        if not (contexts or sources):
            return {
                "answer": "Maaf, belum ada dokumen yang bisa dirujuk (indeks kosong). Jalankan analisis end-to-end dulu.",
                "sources": [],
            }
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.exception("/rag/query failed: %s", e)
        return {"answer": "[RAG-ERROR] Terjadi kesalahan saat menjalankan RAG.", "sources": []}

# @app.post("/rag/query")
# def rag_query(req: RagQuery):
#     """Tanya dokumen yang sudah diindeks (PGVector + BM25). Digunakan Page 3 Streamlit."""
#     ctx = RAG.retrieve(req.query, k=req.k)
#     answer = RAG.generate(req.query, ctx)
#     return {
#         "answer": answer,
#         "contexts": [
#             {
#                 "text": c.text,
#                 "score_lex": c.score_lex,
#                 "score_sem": c.score_sem,
#                 "score_final": c.score_final,
#                 "metadata": c.metadata,
#             }
#             for c in ctx
#         ],
#     }
