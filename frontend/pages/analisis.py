# frontend/pages/Pipeline_Analisis.py
import os
import io
import csv
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# fallback in-process
IN_PROCESS = False
BACKEND_URL = os.getenv("BACKEND_URL", "").strip()
if not BACKEND_URL:
    IN_PROCESS = True
    from backend.graph.pipeline import build_graph

st.header("🔗 End-to-End Pipeline")

with st.form("e2eform"):
    video_url = st.text_input(
        "TikTok Video URL", placeholder="https://www.tiktok.com/@user/video/..."
    )
    content_id = st.text_input("Content ID")
    content_date = st.date_input("Content Date", value=datetime.today())
    max_comments = st.number_input(
        "Max comments", min_value=1, max_value=500, value=20, step=1
    )
    submitted = st.form_submit_button("Run Pipeline")

if submitted:
    if not video_url:
        st.error("Mohon isi URL.")
        st.stop()

    with st.spinner("Running pipeline..."):
        if not IN_PROCESS:
            try:
                r = requests.post(
                    f"{BACKEND_URL}/analyze",
                    json={
                        "video_url": video_url,
                        "content_id": content_id,
                        "content_date": content_date.strftime("%Y-%m-%d"),
                        "max_comments": int(max_comments),
                    },
                    timeout=600,
                )
                r.raise_for_status()
                payload = r.json()
            except Exception as e:
                st.error(f"Gagal memanggil backend API: {e}")
                st.stop()
        else:
            # Run LangGraph in-process (dev)
            graph = build_graph()
            state = {
                "video_url": video_url,
                "content_id": content_id,
                "content_date": content_date.strftime("%Y-%m-%d"),
                "max_comments": int(max_comments),
                "apify_token": os.getenv("APIFY_API_TOKEN", ""),
                "qwen_api_key": os.getenv("QWEN_API_KEY", ""),
                "pgvector_url": os.getenv("PGVECTOR_URL", ""),
                "pgvector_collection": os.getenv("PGVECTOR_COLLECTION", "comments"),
            }
            result = graph.invoke(state)

            # samakan struktur dengan backend/_invoke_graph terbaru
            default_date = state["content_date"]
            merged_comments = []
            for c in (result.get("comments_merged") or []):
                merged_comments.append({
                    "document_id": c.get("document_id") or state["content_id"],
                    "text": c.get("text", ""),
                    "sentiment": c.get("sentiment", "unknown"),
                    "confidence": float(c.get("confidence", 0.0)),
                    "topic_label": c.get("topic_label", "Lainnya"),
                    "date": c.get("date") or default_date,
                })

            payload = {
                "insight": {
                    "content_id": result["insight"].content_id,
                    "date": result["insight"].date,
                    "total_comments": result["insight"].total_comments,
                    "num_topics": result["insight"].num_topics,
                    "dominant_topic": result["insight"].dominant_topic,
                    "dominant_topic_percentage": result["insight"].dominant_topic_percentage,
                    "topic_details": result["insight"].topic_details,
                    "summary": result["insight"].summary,
                },
                "merged_comments_count": len(merged_comments),
                "merged_comments": merged_comments,
                "artifacts": result.get("artifacts", {}),
                "rag": {
                    "answer": result.get("rag_answer") or "",
                    "contexts": result.get("rag_contexts", []),
                    "sources": result.get("rag_sources", []),
                },
            }

    st.success("Pipeline selesai.")
    col1, col2 = st.columns([2, 1])

    # ================== LEFT ==================
    with col1:
        st.subheader("Insight")
        st.write(payload.get("insight", {}))

        # ---------- Merged comments (utama) ----------
        st.subheader("Merged Comments (per-komentar)")

        mc = payload.get("merged_comments") or []
        if mc:
            # tampilkan sebagai tabel
            df = pd.DataFrame(mc, columns=[
                              "document_id", "text", "sentiment", "confidence", "topic_label", "date"])
            st.dataframe(df, use_container_width=True, height=420)

            # --- Client-side downloads (tidak bergantung endpoint file backend) ---
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "💾 Download JSON (inline)",
                    data=json.dumps(mc, ensure_ascii=False,
                                    indent=2).encode("utf-8"),
                    file_name=f"comments_{payload['insight']['content_id']}.json",
                    mime="application/json",
                )
            with c2:
                # buat CSV on the fly
                csv_buf = io.StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=df.columns)
                writer.writeheader()
                for row in mc:
                    writer.writerow({k: row.get(k, "") for k in df.columns})
                st.download_button(
                    "💾 Download CSV (inline)",
                    data=csv_buf.getvalue().encode("utf-8"),
                    file_name=f"comments_{payload['insight']['content_id']}.csv",
                    mime="text/csv",
                )
        else:
            st.info(
                "Belum ada komentar yang ter-merge (mungkin video tanpa komentar).")

        # ---------- Summary + file downloads (server-side opsional) ----------
        st.subheader("Summary & Server-side Files")
        backend = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
        content_id = payload.get("insight", {}).get("content_id", "")

        # URL endpoint lama (akan dicoba dulu)
        dl_insight_url = f"{backend}/files/insight/{content_id}.txt"
        dl_csv_url = f"{backend}/files/comments/{content_id}.csv"
        dl_json_url = f"{backend}/files/comments/{content_id}.json"

        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("⬇️ Download Summary (.txt) [server]"):
                try:
                    rr = requests.get(dl_insight_url, timeout=30)
                    rr.raise_for_status()
                    st.download_button(
                        label="Save Summary (server)",
                        data=rr.content,
                        file_name=f"insight_{content_id}.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.warning(f"Gagal ambil summary dari backend: {e}")

        with a2:
            if st.button("⬇️ Download Comments (.csv) [server]"):
                try:
                    rr = requests.get(dl_csv_url, timeout=30)
                    rr.raise_for_status()
                    st.download_button(
                        label="Save CSV (server)",
                        data=rr.content,
                        file_name=f"comments_{content_id}.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.warning(f"Gagal ambil CSV dari backend: {e}")

        with a3:
            if st.button("⬇️ Download Comments (.json) [server]"):
                try:
                    rr = requests.get(dl_json_url, timeout=30)
                    rr.raise_for_status()
                    st.download_button(
                        label="Save JSON (server)",
                        data=rr.content,
                        file_name=f"comments_{content_id}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.warning(f"Gagal ambil JSON dari backend: {e}")

        # ---------- RAG (taruh di expander / bukan utama) ----------
        rag = payload.get("rag") or {}
        with st.expander("RAG (opsional)"):
            st.write(rag.get("answer") or "—")
            st.markdown("**Sources / Citations**")
            srcs = rag.get("sources") or []
            for s in srcs:
                st.markdown(
                    f"**[{s.get('rank','-')}]** `{s.get('document_id','-')}` • "
                    f"*{s.get('topic_label','-')}* • {s.get('sentiment','-')} "
                    f"(score={float(s.get('score_final',0)):.3f})\n\n"
                    f"> {s.get('snippet','')}"
                )


    # ================== RIGHT ==================
    with col2:
        st.metric("Merged Comments", payload.get("merged_comments_count", 0))
        st.subheader("Artifacts (server)")
        st.json(payload.get("artifacts", {}))
