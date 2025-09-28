# frontend/pages/rag.py
import os
import io
import csv
import json
import requests
import streamlit as st

st.header("📚 RAG: Tanya Dokumen yang Sudah Disimpan")

BACKEND = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")


def _norm_source(s, idx0=0):
    """Samakan struktur source jadi: rank, document_id, topic_label, sentiment, score_final, snippet."""
    meta = (s or {}).get("metadata") or {}
    rank = s.get("rank", idx0 + 1)
    doc_id = s.get("document_id") or meta.get("document_id") or "-"
    topic = s.get("topic_label") or meta.get("topic_label") or "-"
    sent = s.get("sentiment") or meta.get("sentiment") or "-"
    score = s.get("score_final", meta.get("score_final", 0.0)) or 0.0
    try:
        score = float(score)
    except Exception:
        score = 0.0
    snippet = s.get("snippet") or s.get("text") or meta.get("snippet") or ""
    return {
        "rank": rank,
        "document_id": doc_id,
        "topic_label": topic,
        "sentiment": sent,
        "score_final": score,
        "snippet": snippet,
    }


query = st.text_input(
    "Pertanyaan", placeholder="Apa pandangan utama audiens tentang topik X?")
K = 3  # tetap K=3

if st.button("Cari & Jawab"):
    if not query.strip():
        st.error("Pertanyaan tidak boleh kosong.")
        st.stop()

    with st.spinner("Memanggil backend..."):
        try:
            r = requests.post(
                f"{BACKEND}/rag/query",
                json={"query": query, "k": K},
                timeout=300,
            )
            r.raise_for_status()
            data = r.json() if r.content else {}
        except Exception as e:
            st.error(f"Gagal memanggil backend: {e}")
            st.stop()

    # --- Jawaban ---
    st.subheader("Jawaban")
    st.write(data.get("answer") or "—")

    # --- Sources ---
    st.subheader("Sources / Citations")
    raw_sources = data.get("sources") or []
    if not raw_sources:
        st.info("Tidak ada sumber yang dikembalikan.")
    else:
        sources = [_norm_source(s, i) for i, s in enumerate(raw_sources)]
        for s in sources:
            st.markdown(
                f"**[{int(s['rank'])}]** `{s['document_id']}` • "
                f"*{s['topic_label']}* • {s['sentiment']} "
                f"(score={s['score_final']:.3f})"
            )
            st.markdown(f"> {s['snippet']}")

        # Download buttons
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "💾 Download Sources (JSON)",
                data=json.dumps(sources, ensure_ascii=False,
                                indent=2).encode("utf-8"),
                file_name="rag_sources.json",
                mime="application/json",
            )
        with c2:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(sources[0].keys()))
            writer.writeheader()
            writer.writerows(sources)
            st.download_button(
                "💾 Download Sources (CSV)",
                data=buf.getvalue().encode("utf-8"),
                file_name="rag_sources.csv",
                mime="text/csv",
            )

    # Raw response (untuk debug)
    with st.expander("Raw response (debug)"):
        st.json(data)
