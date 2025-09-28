
# backend/graph/pipeline.py
import os
import logging
from typing import Any, List, Dict, Optional
try:
    from typing_extensions import TypedDict, NotRequired
except ImportError:
    from typing import TypedDict
from langgraph.graph import StateGraph, END
import re
from datetime import datetime
from pathlib import Path
from datetime import datetime
from backend.modules.scraper import scrape_tiktok_comments
from backend.modules.preprocessing import IndonesianPreprocessor
from backend.modules.sentiment import SentimentAnalyzer
from backend.modules.topic import TopicModeler, DocumentTopic, ContentInsight
from backend.modules.rag import HybridRAG
from backend.utils.db import init_tables, save_comments, save_insight
from backend.modules.storage import save_to_json, save_to_csv, save_insight_summary

logger = logging.getLogger(__name__)

# ------------ State type ------------


class State(TypedDict, total=False):
    # ---- input ----
    video_url: str
    content_id: str
    content_date: NotRequired[str]
    max_comments: int
    apify_token: str
    qwen_api_key: str
    pgvector_url: str
    pgvector_collection: str

    # ---- intermediates/outputs ----
    comments_raw: List[Dict[str, Any]]
    sentiment_results: List[Any]
    sentiment_input_texts: List[str]
    topic_results: List[Any]
    insight: Any  # ContentInsight
    comments_merged: List[Dict[str, Any]]

    rag_answer: str
    rag_contexts: List[Any]
    rag_sources: List[Any]
    rag_indexed: int

    artifacts: Dict[str, str]


# ------------ Helpers ------------
_TEXT_KEYS = (
    "text", "comment", "commentText", "content", "comment_text", "desc",
    "body", "message", "caption", "title"
)


def _extract_text(obj) -> str:
    """
    Recursively search dicts/lists for comment-like strings.
    Priority: known keys in _TEXT_KEYS; fallback to any string found.
    Joins multiple snippets with a space.
    """
    found = []

    def walk(x):
        if isinstance(x, str):
            s = x.strip()
            if s:
                found.append(s)
        elif isinstance(x, dict):
            # First: pick known keys if present
            for k in _TEXT_KEYS:
                if k in x and isinstance(x[k], str) and x[k].strip():
                    found.append(x[k].strip())
            # Then: traverse everything else
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)

    # De-duplicate while preserving order
    seen = set()
    dedup = []
    for s in found:
        if s not in seen:
            seen.add(s)
            dedup.append(s)

    return " ".join(dedup).strip()

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\u200b", "").strip()


def should_continue(state: State) -> str:
    """Dipakai untuk conditional edge setelah scrape."""
    return "continue" if state.get("comments_raw") else "stop"


# ------------ Nodes ------------
def node_scrape(state: State) -> State:
    url = state.get("video_url")
    token = state.get("apify_token") or os.getenv("APIFY_API_TOKEN", "")
    max_n = int(state.get("max_comments") or 50)

    if not token:
        logger.error("APIFY token missing (APIFY_API_TOKEN).")
        state["comments_raw"] = []
        return state
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        logger.error(f"Invalid video_url: {url!r}")
        state["comments_raw"] = []
        return state

    try:
        raw_rows = scrape_tiktok_comments(
            apify_token=token,
            video_url=url,
            max_comments=max_n,
        )
    except TypeError:
        raw_rows = scrape_tiktok_comments(token, url, max_n)
    except Exception as e:
        logger.error(f"Scrape failed: {e}")
        raw_rows = []

    rows = []
    dropped = 0
    for i, r in enumerate(raw_rows or []):
        txt = _extract_text(r)
        if not txt:
            dropped += 1
            continue
        rid = r.get("id") or r.get("commentId") or f"c_{i}"
        rows.append({"id": rid, "text": _clean_text(txt)})

    logger.info(
        f"[SCRAPE] kept={len(rows)} dropped_empty={dropped} total_raw={len(raw_rows or [])}")
    state["comments_raw"] = rows
    return state


def node_sentiment(state: State) -> State:
    """Analyze sentiment from scraped comments with robust extraction & clear logs."""
    # --- Debug awal
    logger.info(f"[SENTIMENT DEBUG] State keys: {list(state.keys())}")
    logger.info(
        f"[SENTIMENT DEBUG] comments_raw exists: {'comments_raw' in state}")

    raw = state.get("comments_raw", [])
    logger.info(
        f"[SENTIMENT DEBUG] Raw comments type: {type(raw)}, length: {len(raw) if raw else 0}")
    if raw:
        logger.info(f"[SENTIMENT DEBUG] First comment sample: {raw[0]}")

    # --- Tanpa komentar → kosongkan & keluar
    if not raw:
        logger.warning("[SENTIMENT] No comments_raw found in state")
        state["sentiment_results"] = []
        state["sentiment_input_texts"] = []
        return state

    # --- Ekstraksi teks yang konsisten dengan node_topic:
    texts: list[str] = []
    for comment in raw:
        t = None
        if isinstance(comment, dict):
            t = comment.get("text")
            if not (isinstance(t, str) and t.strip()):
                t = _extract_text(comment)
        elif isinstance(comment, str):
            t = comment
        else:
            t = _extract_text(comment)

        if isinstance(t, str):
            t = t.strip()
            if t:
                texts.append(t)

    logger.info(
        f"[SENTIMENT] Extracted {len(texts)} valid texts from {len(raw)} raw comments")
    logger.info(f"[SENTIMENT] Sample texts: {texts[:3] if texts else 'None'}")

    if not texts:
        logger.warning(
            "[SENTIMENT] No valid texts extracted for sentiment analysis")
        state["sentiment_results"] = []
        state["sentiment_input_texts"] = []
        return state

    # --- Jalankan analisis sentimen
    try:
        analyzer = SentimentAnalyzer(preprocessor=IndonesianPreprocessor())
        # diharapkan mengembalikan object dengan .text/.sentiment/.confidence
        results = analyzer.run(texts)
        state["sentiment_results"] = results
        # simpan urutan input untuk merge-by-index kalau perlu
        state["sentiment_input_texts"] = texts

        logger.info(f"[SENTIMENT] Analysis successful: {len(results)} results")
        if results:
            sample = results[0]
            # Beberapa implementasi .confidence bisa None/float → amankan
            conf = float(getattr(sample, "confidence", 0.0) or 0.0)
            logger.info(f"[SENTIMENT] Sample result: sentiment={getattr(sample, 'sentiment', 'unknown')}, "
                        f"confidence={conf:.2f}")
    except Exception as e:
        logger.error(f"[SENTIMENT] Analysis failed: {e}")
        state["sentiment_results"] = []
        state["sentiment_input_texts"] = []

    return state


def node_topic(state: State) -> State:
    """Process topics from scraped comments with robust extraction & per-comment fallback."""
    from datetime import datetime

    # --- Debug awal ---
    logger.info(f"[TOPIC DEBUG] State keys: {list(state.keys())}")
    logger.info(
        f"[TOPIC DEBUG] comments_raw exists: {'comments_raw' in state}")

    raw = state.get("comments_raw", [])
    logger.info(
        f"[TOPIC DEBUG] Raw comments type: {type(raw)}, length: {len(raw) if raw else 0}")
    if raw:
        logger.info(f"[TOPIC DEBUG] First comment sample: {raw[0]}")

    date_val = state.get("content_date") or datetime.now().strftime("%Y-%m-%d")
    content_id = state.get("content_id", "content")

    # --- No comments: graceful fallback ---
    if not raw:
        logger.warning("[TOPIC] No comments_raw found in state")
        empty_insight = ContentInsight(
            content_id=content_id,
            date=date_val,
            total_comments=0,
            num_topics=0,
            dominant_topic="N/A",
            dominant_topic_percentage=0.0,
            topic_details=[],
            summary="Tidak ada komentar yang dapat dianalisis.",
        )
        state["topic_results"] = []
        state["insight"] = empty_insight
        state["comments_merged"] = []
        return state

    # --- Ekstraksi teks yang robust: pakai field 'text' dulu, baru fallback recursive ---
    texts: list[str] = []
    ids: list[str] = []
    raw_texts_for_fallback: list[str] = []

    for i, comment in enumerate(raw):
        # id aman
        cid = f"comment_{i}"
        if isinstance(comment, dict):
            cid = comment.get("id", cid)

        # ambil text stabil
        t = None
        if isinstance(comment, dict):
            t = comment.get("text")
            if not (isinstance(t, str) and t.strip()):
                t = _extract_text(comment)
        elif isinstance(comment, str):
            t = comment
        else:
            t = _extract_text(comment)

        # simpan untuk fallback (meski mungkin kosong)
        raw_texts_for_fallback.append(t if isinstance(t, str) else "")

        # keep hanya non-empty
        if isinstance(t, str):
            t = t.strip()
            if t:
                texts.append(t)
                ids.append(cid)

    logger.info(
        f"[TOPIC] Extracted texts: {len(texts)} from {len(raw)} raw comments")
    logger.info(f"[TOPIC] Sample texts: {texts[:3] if texts else 'None'}")

    # --- Jika setelah ekstraksi tetap kosong: isi merged_fallback per-komentar ---
    if not texts:
        logger.warning(
            "[TOPIC] No valid texts extracted from comments; building merged_fallback")

        merged = []
        nonempty = 0
        for t in raw_texts_for_fallback:
            if isinstance(t, str) and t.strip():
                nonempty += 1
                merged.append({
                    "document_id": content_id,
                    "text": t.strip(),
                    "sentiment": "unknown",
                    "confidence": 0.0,
                    "topic_label": "Lainnya",
                    "date": date_val,
                })

        # Insight mencerminkan ada raw tapi tidak bisa dimodelkan
        empty_insight = ContentInsight(
            content_id=content_id,
            date=date_val,
            total_comments=len(raw),
            num_topics=0,
            dominant_topic="N/A",
            dominant_topic_percentage=0.0,
            topic_details=[],
            summary=f"Dari {len(raw)} komentar, tidak ada yang dapat dimodelkan untuk topik.",
        )
        state["topic_results"] = []
        state["insight"] = empty_insight
        # <- tetap berikan per-komentar kalau ada teks mentah
        state["comments_merged"] = merged
        logger.info(
            f"[TOPIC] texts usable=0, merged_fallback={len(merged)}, nonempty_raw={nonempty}")
        return state

    # --- Topic modeling normal ---
    try:
        modeler = TopicModeler(preprocessor=IndonesianPreprocessor())
        topic_results, insight = modeler.run(
            texts=texts,
            ids=ids,
            content_id=content_id,
            content_date=date_val,
        )
        state["topic_results"] = topic_results
        state["insight"] = insight
        logger.info(
            f"[TOPIC] Topic modeling successful: {len(topic_results)} results")
    except Exception as e:
        logger.error(f"[TOPIC] Topic modeling failed: {e}")
        # Fallback insight pada error
        fallback_insight = ContentInsight(
            content_id=content_id,
            date=date_val,
            total_comments=len(texts),
            num_topics=0,
            dominant_topic="Lainnya",
            dominant_topic_percentage=0.0,
            topic_details=[],
            summary=f"Error dalam analisis topik untuk {len(texts)} komentar. Semua diberi label 'Lainnya'.",
        )
        state["topic_results"] = []
        state["insight"] = fallback_insight

        # Meskipun gagal, tetap buat merged per-komentar
        state["comments_merged"] = [{
            "document_id": content_id,
            "text": t,
            "sentiment": "unknown",
            "confidence": 0.0,
            "topic_label": "Lainnya",
            "date": date_val,
        } for t in texts]
        logger.info(
            f"[TOPIC] Fallback merged built: {len(state['comments_merged'])} items")
        return state

    # --- Merge dengan hasil sentimen ---
    sentiment_results = state.get("sentiment_results", [])
    sentiment_map = {}
    for s in sentiment_results:
        # SentimentAnalyzer.run mengembalikan object dengan .text/.sentiment/.confidence
        if hasattr(s, "text") and isinstance(s.text, str) and s.text:
            sentiment_map[s.text] = s

    # Buat topic map dari hasil model
    topic_map = {}
    for t in state.get("topic_results", []):
        if hasattr(t, "text") and isinstance(t.text, str) and t.text:
            topic_map[t.text] = t

    merged = []
    for t in texts:
        s_obj = sentiment_map.get(t)
        tp_obj = topic_map.get(t)
        merged.append({
            "document_id": content_id,
            "text": t,
            "sentiment": getattr(s_obj, "sentiment", "unknown"),
            "confidence": float(getattr(s_obj, "confidence", 0.0)) if s_obj else 0.0,
            "topic_label": getattr(tp_obj, "topic_label", "Lainnya"),
            "date": date_val,
        })

    state["comments_merged"] = merged
    logger.info(
        f"[TOPIC] Final: texts={len(texts)}, merged={len(merged)}, topics={len(state.get('topic_results', []))}")
    return state


def node_rag(state: State) -> State:
    """RAG processing with proper error handling."""

    logger.info("[RAG] Starting RAG processing")

    # Get merged comments
    merged = state.get("comments_merged", [])

    if not merged:
        logger.warning("[RAG] No merged comments to process")
        state["rag_answer"] = "Tidak ada komentar untuk dianalisis."
        state["rag_contexts"] = []
        state["rag_sources"] = []
        state["rag_indexed"] = False
        return state

    logger.info(f"[RAG] Processing {len(merged)} merged comments")

    # Initialize RAG with error handling
    try:
        from backend.modules.rag import HybridRAG

        rag = HybridRAG(
            pgvector_url=state.get("pgvector_url", ""),
            collection=state.get("pgvector_collection", "comments"),
            qwen_api_key=state.get("qwen_api_key", ""),
            qwen_model="qwen-flash"
        )

        logger.info("[RAG] HybridRAG initialized successfully")

    except Exception as e:
        logger.error(f"[RAG] Failed to initialize HybridRAG: {e}")
        # Fallback without RAG
        state["rag_answer"] = f"RAG initialization failed: {str(e)}"
        state["rag_contexts"] = []
        state["rag_sources"] = []
        state["rag_indexed"] = False
        return state

    # Index documents
    try:
        rag.index_documents(merged)
        state["rag_indexed"] = True
        logger.info(f"[RAG] Successfully indexed {len(merged)} documents")

    except Exception as e:
        logger.error(f"[RAG] Failed to index documents: {e}")
        state["rag_indexed"] = False
        # Continue anyway, maybe BM25 will work

    # Generate query based on insight
    insight = state.get("insight")
    if insight and hasattr(insight, "summary"):
        query = f"Berikan analisis mendalam tentang: {insight.summary}"
    else:
        query = "Berikan rangkuman umum dari semua komentar yang ada"

    logger.info(f"[RAG] Query: {query[:100]}...")

    # Search and generate answer
    try:
        answer, contexts, sources = rag.search_and_answer(query, k=5)

        state["rag_answer"] = answer
        state["rag_contexts"] = contexts
        state["rag_sources"] = sources

        logger.info(f"[RAG] Generated answer with {len(contexts)} contexts")

    except Exception as e:
        logger.error(f"[RAG] Failed to search and generate: {e}")

        # Fallback: provide basic summary
        sample_texts = [m.get("text", "")[:100] for m in merged[:3]]
        fallback_answer = (
            f"[RAG Error] Tidak dapat menghasilkan analisis lengkap. "
            f"Total {len(merged)} komentar telah diproses. "
            f"Sample: {', '.join(sample_texts)}"
        )

        state["rag_answer"] = fallback_answer
        state["rag_contexts"] = sample_texts
        state["rag_sources"] = []

    return state

def node_persist(state: "State") -> "State":
    """
    Persist all outputs to files and database, lalu kembalikan state yang sudah dilengkapi 'artifacts'.
    """
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    def _safe_id(s: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s or "").strip("_")
        return s[:64] or "content"

    safe_cid = _safe_id(state.get("content_id", "content"))

    suffix_env = (os.getenv("SAVE_TS_SUFFIX") or "").strip()
    if suffix_env.upper() == "AUTO":
        suffix = "_" + datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    elif suffix_env:
        suffix = f"_{suffix_env}"
    else:
        suffix = ""

    ins = state.get("insight")
    if ins is None:
        ins = ContentInsight(
            content_id=safe_cid,
            date=state.get("content_date"),
            total_comments=0,
            num_topics=0,
            dominant_topic="N/A",
            dominant_topic_percentage=0.0,
            topic_details=[],
            summary="Tidak ada komentar yang dapat dianalisis.",
        )
        state["insight"] = ins

    base_name = f"comments_{safe_cid}{suffix}"
    json_name = f"{base_name}.json"
    csv_name = f"{base_name}.csv"
    insight_name = f"insight_{safe_cid}{suffix}.txt"

    json_path = data_dir / json_name
    csv_path = data_dir / csv_name
    insight_path = data_dir / insight_name

    try:
        save_to_json(state.get("comments_merged", []), str(json_path))
    except Exception as e:
        logger.error(f"Failed to save JSON '{json_path}': {e}")

    try:
        save_to_csv(state.get("comments_merged", []), str(csv_path))
    except Exception as e:
        logger.error(f"Failed to save CSV '{csv_path}': {e}")

    try:
        save_insight_summary(ins.summary, str(insight_path))
    except Exception as e:
        logger.error(f"Failed to save insight TXT '{insight_path}': {e}")

    state["artifacts"] = {
        "json": json_name,
        "csv": csv_name,
        "insight_txt": insight_name,
    }

    try:
        rows = state.get("comments_merged") or []
        if rows:
            save_comments(rows)
    except Exception as e:
        logger.error(f"Failed to save comments into DB: {e}")

    try:
        save_insight({
            "content_id": ins.content_id,
            "date": ins.date,
            "total_comments": ins.total_comments,
            "num_topics": ins.num_topics,
            "dominant_topic": ins.dominant_topic,
            "dominant_topic_percentage": ins.dominant_topic_percentage,
            "topic_details": ins.topic_details,
            "summary": ins.summary,
        })
    except Exception as e:
        logger.error(f"Failed to save insight into DB: {e}")

    return state


def _coerce_date(dt):
    if not dt:
        return None
    try:
        if isinstance(dt, (int, float)):
            return datetime.utcfromtimestamp(int(dt)).strftime("%Y-%m-%d")
        s = str(dt)
        return s[:10] if len(s) >= 10 else s
    except Exception:
        return None


# ------------ Builder ------------
def build_graph() -> StateGraph:
    g = StateGraph(State)

    g.add_node("scrape",   node_scrape)
    g.add_node("sentiment", node_sentiment)
    g.add_node("topic",    node_topic)
    g.add_node("rag",      node_rag)
    g.add_node("persist",  node_persist)

    g.set_entry_point("scrape")
    g.add_conditional_edges(
        "scrape",
        should_continue,
        {"continue": "sentiment", "stop": "persist"}
    )
    g.add_edge("sentiment", "topic")
    g.add_edge("topic", "rag")
    g.add_edge("rag", "persist")
    g.add_edge("persist", END)
    return g


def get_compiled_graph():
    return build_graph().compile()
