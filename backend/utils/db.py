# backend/utils/db.py
from __future__ import annotations

import json
import os
import re
import logging
from typing import Any, Dict, Iterable, List, Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------- DSN helpers ----------

def _normalize_dsn(uri: str) -> str:
    """
    Terima SQLAlchemy-style atau libpq-style, kembalikan URI libpq untuk psycopg2.
    - "postgresql+psycopg2://user:pass@host:5432/db" -> "postgresql://user:pass@host:5432/db"
    - "postgresql://..." -> apa adanya
    - "dbname=... user=... host=..." -> apa adanya
    """
    if not uri:
        return uri
    uri = uri.strip()
    if uri.startswith("postgresql+psycopg2://"):
        return "postgresql://" + uri.split("postgresql+psycopg2://", 1)[1]
    return uri


def _build_uri_from_env() -> Optional[str]:
    """
    Bangun URI libpq dari env jika PG_DSN kosong:
      PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    Juga fallback POSTGRES_*.
    """
    host = os.getenv("PGHOST") or os.getenv("POSTGRES_HOST")
    port = os.getenv("PGPORT") or os.getenv("POSTGRES_PORT") or "5432"
    db = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB")
    user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER")
    pwd = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD")

    if host and db and user:
        pwd_part = f":{pwd}" if pwd else ""
        return f"postgresql://{user}{pwd_part}@{host}:{port}/{db}"
    return None


def get_pg_dsn() -> str:
    """
    Ambil DSN final untuk psycopg2. Utamakan PG_DSN, normalize bila perlu.
    Jika tidak ada, bangun dari env lain.
    """
    raw = os.getenv("PG_DSN", "").strip()
    if raw:
        # kalau user tak sengaja isi sqlalchemy style, normalkan
        if raw.startswith("postgresql+psycopg2://"):
            raw = "postgresql://" + raw.split("postgresql+psycopg2://", 1)[1]
        return raw

    # fallback dari env lain kalau ada (opsional; hapus jika tak dipakai)
    user = os.getenv("PGUSER", "app")
    pwd = os.getenv("PGPASSWORD", "app")
    host = os.getenv("PGHOST", "doc_pgvector")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "appdb")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def _get_pg_dsn() -> str:
    return get_pg_dsn()


def get_conn():
    dsn = get_pg_dsn()
    safe_dsn = dsn
    try:
        # sembunyikan kredensial saat logging
        import re as _re
        safe_dsn = _re.sub(r'//[^@]+@', '//***@', dsn)
    except Exception:
        pass
    logger.info(f"Connecting to Postgres via DSN: {safe_dsn}")
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def migrate_vector_metadata_to_jsonb() -> None:
    """
    Ubah kolom metadata di tabel LangChain PGVector ke JSONB bila masih JSON.
    Aman dipanggil berulang. Jika tabel/kolom belum ada → skip.
    Tidak melempar error fatal (log warning saja).
    """
    try:
        dsn = get_pg_dsn()  # <-- perbaikan di sini (tidak pakai _get_pg_dsn)
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT data_type, udt_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'langchain_pg_embedding'
                  AND column_name  = 'metadata';
            """)
            row = cur.fetchone()
            if not row:
                logger.info(
                    "JSONB migration: kolom public.langchain_pg_embedding.metadata tidak ditemukan, skip.")
                return

            data_type, udt_name = row
            is_json = (data_type == 'json') or (udt_name == 'json')
            is_jsonb = (data_type == 'jsonb') or (udt_name == 'jsonb')

            if is_json:
                logger.info(
                    "JSONB migration: mengubah kolom metadata dari JSON -> JSONB...")
                cur.execute("""
                    ALTER TABLE public.langchain_pg_embedding
                    ALTER COLUMN metadata TYPE jsonb
                    USING metadata::jsonb;
                """)
                logger.info("JSONB migration: sukses.")
            elif is_jsonb:
                logger.info(
                    "JSONB migration: kolom metadata sudah JSONB, tidak ada perubahan.")
            else:
                logger.info(
                    f"JSONB migration: tipe kolom metadata = {data_type}/{udt_name}, tidak diubah.")

            # index GIN untuk metadata jsonb (idempotent)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lcpg_metadata_gin
                ON public.langchain_pg_embedding
                USING gin ((metadata));
            """)
    except Exception as e:
        logger.warning(f"JSONB migration dilewati: {e}")
# ---------- Schema management (JSONB + indexes) ----------

DDL = [
    # extension pgvector (idempotent)
    "CREATE EXTENSION IF NOT EXISTS vector",
    # comments_analysis: simpan baris komentar (structured)
    """
    CREATE TABLE IF NOT EXISTS comments_analysis (
        id              BIGSERIAL PRIMARY KEY,
        document_id     TEXT NOT NULL,
        text            TEXT NOT NULL,
        sentiment       TEXT,
        confidence      DOUBLE PRECISION,
        topic_label     TEXT,
        date            DATE,
        metadata        JSONB DEFAULT '{}'::jsonb
    )
    """,
    # unik untuk menghindari duplikasi baris sama dari scrape yang sama
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_comments_unique ON comments_analysis (document_id, md5(text))",
    "CREATE INDEX IF NOT EXISTS ix_comments_doc ON comments_analysis (document_id)",
    "CREATE INDEX IF NOT EXISTS ix_comments_topic ON comments_analysis (topic_label)",

    # content_insight: agregat (pakai JSONB)
    """
    CREATE TABLE IF NOT EXISTS content_insight (
        content_id                  TEXT PRIMARY KEY,
        date                        DATE,
        total_comments              INTEGER NOT NULL DEFAULT 0,
        num_topics                  INTEGER NOT NULL DEFAULT 0,
        dominant_topic              TEXT,
        dominant_topic_percentage   DOUBLE PRECISION,
        topic_details               JSONB NOT NULL DEFAULT '[]'::jsonb,
        summary                     TEXT
    )
    """,

    # users sederhana (username unik, password hash)
    """
    CREATE TABLE IF NOT EXISTS users (
        id              BIGSERIAL PRIMARY KEY,
        username        TEXT UNIQUE NOT NULL,
        password_hash   TEXT NOT NULL
    )
    """,
]


def init_tables():
    """Pastikan extension/tables/indexes ada. Aman dipanggil berulang."""
    with get_conn() as conn, conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    logger.info(
        "pgvector extension OK & tables ensured (comments_analysis, content_insight, users).")


# ---------- Persistence helpers ----------

def save_comments(rows: Iterable[Dict[str, Any]]) -> int:
    """
    Simpan banyak baris komentar ke comments_analysis, hindari duplikat via unique index (document_id, md5(text)).
    rows contoh:
    {
        "document_id": str, "text": str, "sentiment": str, "confidence": float,
        "topic_label": str, "date": "YYYY-MM-DD", "metadata": dict (opsional)
    }
    """
    rows = list(rows) or []
    if not rows:
        return 0

    sql = """
    INSERT INTO comments_analysis
        (document_id, text, sentiment, confidence, topic_label, date, metadata)
    VALUES
        %s
    ON CONFLICT (document_id, md5(text))
    DO UPDATE SET
        sentiment = EXCLUDED.sentiment,
        confidence = EXCLUDED.confidence,
        topic_label = EXCLUDED.topic_label,
        date = COALESCE(EXCLUDED.date, comments_analysis.date),
        metadata = COALESCE(EXCLUDED.metadata, comments_analysis.metadata)
    """

    values = []
    for r in rows:
        meta = r.get("metadata") or {}
        if not isinstance(meta, (dict, list)):
            # best effort
            try:
                meta = json.loads(str(meta))
            except Exception:
                meta = {}
        values.append((
            r.get("document_id"),
            r.get("text", ""),
            r.get("sentiment"),
            r.get("confidence"),
            r.get("topic_label"),
            r.get("date"),
            json.dumps(meta),
        ))

    with get_conn() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, sql, values, template=None, page_size=1000)
        return cur.rowcount or 0


def save_insight(ins: Dict[str, Any]) -> None:
    """
    Upsert ke content_insight (JSONB untuk topic_details).
    ins wajib berisi kunci:
      content_id, date, total_comments, num_topics, dominant_topic, dominant_topic_percentage, topic_details(list/dict), summary
    """
    sql = """
    INSERT INTO content_insight (
        content_id, date, total_comments, num_topics,
        dominant_topic, dominant_topic_percentage, topic_details, summary
    ) VALUES (
        %(content_id)s, %(date)s, %(total_comments)s, %(num_topics)s,
        %(dominant_topic)s, %(dominant_topic_percentage)s, %(topic_details)s, %(summary)s
    )
    ON CONFLICT (content_id) DO UPDATE SET
        date = EXCLUDED.date,
        total_comments = EXCLUDED.total_comments,
        num_topics = EXCLUDED.num_topics,
        dominant_topic = EXCLUDED.dominant_topic,
        dominant_topic_percentage = EXCLUDED.dominant_topic_percentage,
        topic_details = EXCLUDED.topic_details,
        summary = EXCLUDED.summary
    """
    payload = dict(ins)
    td = payload.get("topic_details")
    if not isinstance(td, (dict, list)):
        try:
            td = json.loads(str(td))
        except Exception:
            td = []
    payload["topic_details"] = json.dumps(td)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, payload)
    logger.info("Saved insight to content_insight.")


def get_pg_dsn() -> str:
    raw = os.getenv("PG_DSN", "").strip()
    if raw:
        return _normalize_dsn(raw)
    built = _build_uri_from_env()
    if built:
        return built
    # fallback default di jaringan compose-mu:
    return "postgresql://app:app@doc_pgvector:5432/appdb"


def init_all():
    """Panggil ini saat startup FastAPI untuk menjalankan semua inisialisasi DB."""
    logger.info("Starting database initialization...")
    init_tables()
    logger.info("Database initialization finished.")
