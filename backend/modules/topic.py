# modules/topic.py
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from .preprocessing import IndonesianPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class DocumentTopic:
    document_id: str
    text: str
    processed_text: str
    topic_id: int
    topic_label: str
    keywords: List[str]
    probability: float


@dataclass
class ContentInsight:
    content_id: str
    date: str
    total_comments: int
    num_topics: int
    dominant_topic: str
    dominant_topic_percentage: float
    topic_details: List[Dict]
    summary: str


class TopicModeler:
    def __init__(self, topic_model: str = "firqaaa/indo-sentence-bert-base", min_topic_size: int = 3,
                 preprocessor: IndonesianPreprocessor | None = None):
        logger.info(f"Loading Topic embedding model: {topic_model}")
        self.embedding_model = SentenceTransformer(topic_model)
        self.min_topic_size = min_topic_size
        self.pre = preprocessor or IndonesianPreprocessor()

    def run(self, texts: List[str], ids: List[str],
            content_id: str, content_date: str | None = None) -> Tuple[List[DocumentTopic], ContentInsight]:
        processed = [self.pre.process(t, "topic") for t in texts]

        # gunakan cleaned bila ada; jika kosong, pakai fallback dari teks asli agar tidak blank
        docs = []
        for p in processed:
            if p.cleaned.strip():
                docs.append(p.cleaned.strip())
            else:
                # ambil 1–2 token alfanumerik dari teks asli sebagai minimal fallback
                import re as _re
                rough = _re.findall(r"\w+", (p.original or "").lower())
                rough = [w for w in rough if len(w) > 2]
                docs.append(" ".join(rough[:2]) if rough else "teks")

        n_nonempty = sum(1 for d in docs if d.strip())

        # FULL FALLBACK: kalau setelah semua upaya tetap tidak ada bahan
        if n_nonempty == 0:
            logger.warning(
                "All comments are empty after preprocessing for topic modeling. Fallback to 'Lainnya'.")
            results: List[DocumentTopic] = [
                DocumentTopic(cid, txt, proc.cleaned, -1, "Lainnya", [], 0.0)
                for cid, txt, proc in zip(ids, texts, processed)
            ]
            insight = self._generate_insight(
                results, content_id, content_date or datetime.now().strftime("%Y-%m-%d")
            )
            return results, insight

        # min_topic_size yang realistis untuk korpus kecil
        if n_nonempty <= 5:
            effective_min = 2
        else:
            effective_min = max(
                2, min(self.min_topic_size, n_nonempty // 5, n_nonempty))

        model = BERTopic(
            embedding_model=self.embedding_model,
            min_topic_size=effective_min,
            representation_model=KeyBERTInspired(),
            verbose=False
        )

        topics, probs = model.fit_transform(docs)

        results: List[DocumentTopic] = []
        for i, (cid, text, proc, tid) in enumerate(zip(ids, texts, processed, topics)):
            if tid == -1:
                label, keywords = "Lainnya", []
            else:
                items = model.get_topic(tid) or []
                keywords = [w for w, _ in items[:5]]
                label = " - ".join(keywords[:3]
                                   ) if keywords else f"Topik {tid}"
            prob = float(probs[i]) if probs is not None else 0.0
            results.append(DocumentTopic(
                cid, text, proc.cleaned, tid, label, keywords, prob
            ))

        insight = self._generate_insight(
            results, content_id, content_date or datetime.now().strftime("%Y-%m-%d"))
        return results, insight

    def _generate_insight(self, topic_results: List[DocumentTopic], content_id: str, date: str) -> ContentInsight:
        total = len(topic_results)
        if total == 0:
            return ContentInsight(content_id, date, 0, 0, "N/A", 0.0, [], "No insight available.")

        topic_counts = Counter(
            tr.topic_label for tr in topic_results if tr.topic_id != -1)
        num_topics = len(topic_counts)
        topic_dist = {label: count / total *
                      100 for label, count in topic_counts.items()}

        if not topic_counts:
            # semua masuk -1 (Lainnya)
            others = sum(1 for tr in topic_results if tr.topic_id == -1)
            dominant_topic = "Lainnya" if others == total else "Beragam"
            dominant_pct = 100.0 if others == total else 0.0
            topic_details = []
            summary = (
                f"Dari {total} komentar, mayoritas tidak memuat kata-kata yang dapat dimodelkan "
                f"({dominant_pct:.1f}% 'Lainnya')."
                if dominant_topic == "Lainnya"
                else f"Dari {total} komentar, variasi topik terlalu beragam."
            )
            return ContentInsight(content_id, date, total, num_topics, dominant_topic, dominant_pct, topic_details, summary)

        dominant_topic, dominant_count = topic_counts.most_common(1)[0]
        dominant_pct = (dominant_count / total) * 100

        topic_details = []
        for label, count in topic_counts.most_common():
            topic_docs = [
                tr for tr in topic_results if tr.topic_label == label]
            topic_details.append({
                "topic": label,
                "percentage": topic_dist.get(label, 0),
                "count": count,
                "keywords": topic_docs[0].keywords if topic_docs else [],
                "examples": [d.text[:80] for d in topic_docs[:2]]
            })
        summary = f"Dari {total} komentar, audiens paling banyak membahas '{dominant_topic}' ({dominant_pct:.1f}%)."
        return ContentInsight(content_id, date, total, num_topics, dominant_topic, dominant_pct, topic_details, summary)
