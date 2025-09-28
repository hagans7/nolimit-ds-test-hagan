# frontend/app.py
import os
import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Comment Intelligence Suite",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------


def check_backend_health(base_url: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
        r.raise_for_status()
        return True, "OK"
    except Exception as e:
        return False, str(e)


def env_badge(label: str, value: str | None):
    v = value if value else "(not set)"
    st.code(f"{label}={v}")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION", "comments")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

st.sidebar.subheader("Environment")
env_badge("BACKEND_URL", BACKEND_URL)
env_badge("PGVECTOR_COLLECTION", PGVECTOR_COLLECTION)
env_badge("LOG_LEVEL", LOG_LEVEL)

ok, msg = check_backend_health(BACKEND_URL)
if ok:
    st.sidebar.success("Backend: Healthy")
else:
    st.sidebar.error("Backend: Unreachable")
    st.sidebar.caption(msg)

st.sidebar.markdown("---")
st.sidebar.subheader("Pages")
st.sidebar.markdown(
    "1. End-to-End: scraping → preprocessing → sentiment → topic → persist")
st.sidebar.markdown("2. Sentiment Only: prediksi sentimen dari input manual")
st.sidebar.markdown("3. RAG QA: tanya dokumen yang sudah diindeks")
st.sidebar.info(
    "Gunakan navigator default Streamlit di sidebar kiri (section 'Pages').")

# -----------------------------
# Main
# -----------------------------
st.title("🧠 Comment Intelligence Suite")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown(
        """
### Selamat datang!
Suite ini terdiri dari **tiga halaman**:

1. **End-to-End**  
   Input URL TikTok → *Scrape* komentar (Apify) → *Preprocess* → *Sentiment & Topic Modeling* →  
   *Persist* ke PostgreSQL (structured) & PGVector (unstructured) → simpan file **CSV/JSON/TXT** →  
   *RAG* ringkas opini utama (dengan **citations**).

2. **Sentiment Only**  
   Prediksi sentimen dari **input manual** (tanpa scraping).

3. **RAG QA**  
   Ajukan pertanyaan atas dokumen yang sudah diindeks (**PGVector + BM25**) dan dapatkan jawaban dengan **sumber/citation**.

> Halaman-halaman itu otomatis tampil di sidebar (section **Pages**) karena disimpan di folder `frontend/pages/`.
"""
    )

    st.markdown("#### Quick links")
    # Tautan cepat ke pages (hanya anchor; Streamlit multipage tetap di sidebar)
    st.write("➡️ Buka dari sidebar kiri, atau klik tombol di bawah:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.page_link("pages/analisis.py", label="🔗 End-to-End", icon="🔗")
    with c2:
        st.page_link("pages/sentimen.py",
                     label="🔗 Sentiment Only", icon="🔗")
    with c3:
        st.page_link("pages/rag.py", label="🔗 RAG QA", icon="🔗")

with col2:
    st.markdown("### Status")
    st.metric("Backend Health", "Healthy" if ok else "Down")
    st.caption(f"Health check: {BACKEND_URL}/health")
    st.markdown("### Tips")
    st.write(
        "- Jalankan via Docker Compose agar **frontend** bisa akses **backend** memakai host `backend:8000`.\n"
        "- Cek `.env` untuk `APIFY_API_TOKEN`, `QWEN_API_KEY`, dan `PGVECTOR_URL` pada service **backend**.\n"
        "- Model HuggingFace akan di-cache ke volume `hf_cache` (lebih cepat setelah run pertama)."
    )

st.markdown("---")
st.caption(
    "Versi UI ini terhubung dengan FastAPI endpoints: `/analyze`, `/sentiment/predict`, `/rag/query`, dan `/files/...` "
    "untuk unduhan manual (summary .txt, comments .csv/.json)."
)
