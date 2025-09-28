# modules/preprocessing.py
import os
import re
import string
import logging
import pandas as pd
from dataclasses import dataclass
from typing import List
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

logger = logging.getLogger(__name__)


@dataclass
class ProcessedText:
    original: str
    cleaned: str
    tokens: List[str]
    mode: str


class IndonesianPreprocessor:
    def __init__(self, slang_path: str = "/app/data/slang_dictionary.csv"):
        factory = StopWordRemoverFactory()
        self.base_stopwords = set(factory.get_stop_words())
        # Jangan buang kata emosional (mantap, keren, bagus, oke, wkwk, haha)
        self.custom_stopwords = {
            'nya', 'nih', 'sih', 'dong', 'deh', 'lah', 'yg', 'dgn', 'utk',
            'min', 'kak', 'bang', 'bro', 'sis', 'gan', 'guys', 'kuy'
        }
        self.all_stopwords = self.base_stopwords.union(self.custom_stopwords)
        self.slang_dict = self._load_slang_dictionary(slang_path)
        logger.info(f"Preprocessor initialized. Slang path={slang_path}")

    def _load_slang_dictionary(self, path: str) -> dict:
        """
        Coba beberapa lokasi umum:
          1) path yang diberikan (default: /app/data/slang_dictionary.csv)
          2) ./data/slang_dictionary.csv
          3) ./slang_dictionary.csv
        """
        candidates = [path, "data/slang_dictionary.csv",
                      "slang_dictionary.csv"]
        for p in candidates:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p, keep_default_na=False)
                    logger.info(f"Loaded slang dictionary from {p}")
                    return pd.Series(df.formal.values, index=df.slang).to_dict()
                except Exception as e:
                    logger.error(
                        f"Failed to load slang dictionary from {p}: {e}")
        logger.warning(
            f"Slang dictionary not found at any of: {', '.join(candidates)}. Normalization will be skipped.")
        return {}

    def process(self, text: str, mode: str) -> ProcessedText:
        original = text or ""
        cleaned = original.lower()

        # Buang URL & mention, konversi hashtag jadi kata
        cleaned = re.sub(r'http\S+|@\w+', ' ', cleaned)
        cleaned = re.sub(r'#(\w+)', r'\1', cleaned)

        # Kompres huruf berulang: "mantaapppp" -> "mantap"
        cleaned = re.sub(r'(.)\1{2,}', r'\1', cleaned)

        # Normalisasi slang sedini mungkin (berdampak baik untuk dua mode)
        cleaned = self.normalize_text(cleaned)

        if mode == "sentiment":
            # Tetap ringan: pertahankan !?
            cleaned = re.sub(r'[^\w\s!?]', ' ', cleaned)
            tokens = cleaned.split()

        else:
            # MODE: topic → pembersihan ringan agar sinyal tetap ada
            # Buang emoji dasar
            cleaned = re.sub(
                "["u"\U0001F600-\U0001F64F"
                u"\U0001F300-\U0001F5FF"
                u"\U0001F680-\U0001F6FF"
                u"\U0001F1E0-\U0001F1FF" "]+",
                " ",
                cleaned
            )
            # Buang tanda baca → spasi
            cleaned = re.sub(
                f'[{re.escape(string.punctuation)}]', ' ', cleaned)
            # Buang angka berdiri sendiri
            cleaned = re.sub(r'\b\d+\b', ' ', cleaned)

            # Ambil token alfanumerik/underscore saja
            words = re.findall(r"[a-z0-9_]+", cleaned)

            # Normalisasi ulang per-kata (aman jika slang_dict kosong)
            if self.slang_dict:
                words = [self.slang_dict.get(w, w) for w in words]

            # Filter ringan: jangan buang kata emosional, izinkan panjang >= 2
            tokens = [w for w in words if (
                w not in self.all_stopwords) and len(w) >= 2]

            # ===== Fallback agar tidak pernah kosong =====
            if not tokens:
                # Ambil 1–2 kata terpanjang dari teks asli yang sudah dibersihkan ringan
                raw = re.findall(
                    r"[a-z0-9_]+", (original.lower() if original else ""))
                raw = [
                    w for w in raw if w not in self.all_stopwords and len(w) >= 2]
                fallback = sorted(set(raw), key=len, reverse=True)
                tokens = fallback[:2] if fallback else ["teks"]
                logger.debug(f"[topic] tokens empty → fallback used: {tokens}")

            cleaned = " ".join(tokens)

        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return ProcessedText(original, cleaned, tokens, mode)

    def normalize_text(self, text: str) -> str:
        if not self.slang_dict:
            return text
        return " ".join(self.slang_dict.get(w, w) for w in text.split())
