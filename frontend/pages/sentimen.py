# frontend/pages/2_Sentiment_Only.py
import os
import requests
import streamlit as st

st.header("💬 Sentiment Only (Manual Input)")

backend = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")

text = st.text_area("Masukkan kalimat / komentar", height=140,
                    placeholder="Contoh: videonya seru, tapi suaranya kurang jelas.")
if st.button("Prediksi Sentimen"):
    if not text.strip():
        st.error("Isi dulu teksnya ya.")
        st.stop()
    with st.spinner("Memanggil backend..."):
        try:
            r = requests.post(f"{backend}/sentiment/predict",
                              json={"texts": [text]}, timeout=120)
            r.raise_for_status()
            data = r.json()[0]
        except Exception as e:
            st.error(f"Gagal memanggil backend: {e}")
            st.stop()
    st.success(
        f"Prediksi: **{data['sentiment'].upper()}** (confidence: {data['confidence']:.3f})")
    with st.expander("Skor lengkap"):
        st.json(data["scores"])
