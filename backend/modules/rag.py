# backend/modules/rag.py
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import PGVector
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score_lex: float
    score_sem: float
    score_final: float
    metadata: Dict[str, Any]


def _with_defaults(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    m = dict(meta or {})
    m.setdefault("document_id", "-")
    m.setdefault("topic_label", "-")
    m.setdefault("sentiment", "-")
    return m


def _mk_source(c: "RetrievedChunk", i: int) -> Dict[str, Any]:
    meta = _with_defaults(c.metadata)
    return {
        "rank": i + 1,
        "snippet": (c.text or "")[:200],
        "document_id": meta.get("document_id", "-"),
        "topic_label": meta.get("topic_label", "-"),
        "sentiment": meta.get("sentiment", "-"),
        "metadata": meta,
        "score_lex": round(c.score_lex, 4),
        "score_sem": round(c.score_sem, 4),
        "score_final": round(c.score_final, 4),
    }


def _make_qwen_client(api_key: Optional[str]) -> Optional[OpenAI]:
    key = api_key or os.getenv("QWEN_API_KEY") or os.getenv(
        "OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        logger.warning(
            "Qwen client not created: no API key in QWEN_API_KEY/OPENAI_API_KEY/DASHSCOPE_API_KEY")
        return None
    base_url = (
        os.getenv("QWEN_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    try:
        return OpenAI(api_key=key, base_url=base_url)
    except Exception as e:
        logger.error(f"Failed to create Qwen client: {e}")
        return None


class HybridRAG:
    """
    Hybrid RAG = BM25 (in-memory) + PGVector (semantic) + generasi jawaban via Qwen.
    """

    def __init__(
        self,
        pgvector_url: str,
        collection: str = "comments",
        qwen_api_key: Optional[str] = None,
        qwen_model: str = "qwen-flash",
    ):
        self.log = logger
        self.qwen_api_key = qwen_api_key or os.getenv("QWEN_API_KEY")
        self.qwen_model = qwen_model

        # Embeddings
        self.emb = HuggingFaceEmbeddings(
            model_name="firqaaa/indo-sentence-bert-base")

        # Vector store
        try:
            self.db = PGVector(
                connection_string=pgvector_url,
                collection_name=collection,
                embedding_function=self.emb,
                pre_delete_collection=False,
                use_jsonb=True,
            )
        except TypeError:
            self.db = PGVector(
                connection_string=pgvector_url,
                collection_name=collection,
                embedding_function=self.emb,
                pre_delete_collection=False,
            )
        except Exception as e:
            self.log.error(f"Failed to initialize PGVector: {e}")
            self.db = None

        # BM25 in-memory
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokens: List[List[str]] = []
        self._bm25_texts: List[str] = []
        self._bm25_meta: List[Dict[str, Any]] = []

    # ---------------- Index ----------------
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index dokumen ke vector store + build BM25."""
        if not documents:
            self.log.warning("No documents to index")
            return

        texts, metas = [], []
        for d in documents:
            t = d.get("text", "")
            if t:
                texts.append(t)
                metas.append({k: v for k, v in d.items() if k != "text"})

        if not texts:
            self.log.warning("No valid texts found in documents")
            return

        if self.db:
            try:
                docs = [Document(page_content=t, metadata=m)
                        for t, m in zip(texts, metas)]
                self.db.add_documents(docs)
                self.log.info(f"Indexed {len(docs)} documents in vector store")
            except Exception as e:
                self.log.error(f"Failed to index in vector store: {e}")

        self._bm25_texts = texts
        self._bm25_meta = metas
        self._bm25_tokens = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(
            self._bm25_tokens) if self._bm25_tokens else None

    # ---------------- Retrieve ----------------
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        results: List[RetrievedChunk] = []

        # BM25 pass
        if self._bm25 and self._bm25_texts:
            try:
                q = query.lower().split()
                scores = self._bm25.get_scores(q)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                    : min(k * 2, len(scores))]
                maxv = max(scores) if scores else 1.0
                for idx in top_idx:
                    if idx < len(self._bm25_texts):
                        norm = (scores[idx] / maxv) if maxv > 0 else 0.0
                        meta = _with_defaults(
                            self._bm25_meta[idx] if idx < len(self._bm25_meta) else {})
                        results.append(
                            RetrievedChunk(
                                text=self._bm25_texts[idx],
                                score_lex=norm,
                                score_sem=0.0,
                                score_final=norm,
                                metadata=meta,
                            )
                        )
            except Exception as e:
                self.log.error(f"BM25 search failed: {e}")

        # Semantic pass
        sem_docs = []
        if self.db:
            try:
                sem_docs = self.db.similarity_search(query, k=k)
            except Exception as e:
                self.log.error(f"Semantic search failed: {e}")

        bm_by_text: Dict[str, RetrievedChunk] = {r.text: r for r in results}

        for rank, d in enumerate(sem_docs):
            s = 1.0 / (rank + 1)
            meta = _with_defaults(d.metadata or {})
            if d.page_content in bm_by_text:
                ch = bm_by_text[d.page_content]
                ch.score_sem = s
                ch.score_final = 0.5 * ch.score_lex + 0.5 * s
                # isi metadata yang kosong dari semantic
                if ch.metadata.get("document_id") in (None, "-", ""):
                    ch.metadata = meta
            else:
                results.append(
                    RetrievedChunk(
                        text=d.page_content,
                        score_lex=0.0,
                        score_sem=s,
                        score_final=s,
                        metadata=meta,
                    )
                )

        results.sort(key=lambda r: r.score_final, reverse=True)
        return results[:k]

    # ---------------- Generate ----------------
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            return "Tidak ada konteks yang tersedia untuk menjawab pertanyaan."

        client = _make_qwen_client(self.qwen_api_key)
        if not client:
            top = "\n".join([f"- {c[:100]}..." for c in contexts[:3]])
            return f"[No LLM] Konteks yang ditemukan:\n{top}"

        ctx_block = "\n\n".join(
            [f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        prompt = (
            "Berdasarkan konteks berikut, jawab pertanyaan dengan ringkas dan jelas. "
            "Jika informasi tidak tersedia dalam konteks, katakan 'tidak diketahui'. "
            "Gunakan kutipan [angka] bila relevan.\n\n"
            f"KONTEKS:\n{ctx_block}\n\nPERTANYAAN: {query}\n\nJAWABAN:"
        )

        try:
            resp = client.chat.completions.create(
                model=self.qwen_model,
                messages=[
                    {"role": "system",
                        "content": "Kamu asisten yang menjawab berdasarkan konteks."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            ans = (resp.choices[0].message.content or "").strip()
            return ans or "Tidak dapat menghasilkan jawaban."
        except Exception as e:
            self.log.error(f"Failed to generate answer: {e}")
            top = "\n".join([f"- {c[:100]}..." for c in contexts[:3]])
            return f"[Error generating answer] Konteks:\n{top}"

    # ---------------- Search + Answer ----------------
    def search_and_answer(self, query: str, k: int = 5) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        chunks = self.retrieve(query, k)
        if not chunks:
            return ("Tidak ada dokumen relevan yang ditemukan.", [], [])
        contexts = [c.text for c in chunks]
        sources = [_mk_source(c, i) for i, c in enumerate(chunks)]
        answer = self.generate_answer(query, contexts)
        return (answer, contexts, sources)

    # ---------------- Backward-compat (untuk pemanggil lama) ----------------
    def index_comments(self, comments: List[Dict[str, Any]]):
        return self.index_documents(comments)

    def generate(self, query: str, contexts: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
        if contexts and isinstance(contexts[0], RetrievedChunk):
            sources = [_mk_source(c, i) for i, c in enumerate(contexts)]
            ctx_texts = [c.text for c in contexts]
        else:
            ctx_texts = [str(c) for c in contexts]
            sources = [
                {
                    "rank": i + 1,
                    "snippet": (t or "")[:200],
                    "metadata": {},
                    "score_lex": 0.0,
                    "score_sem": 0.0,
                    "score_final": 0.0,
                }
                for i, t in enumerate(ctx_texts)
            ]
        answer = self.generate_answer(query, ctx_texts)
        return answer, sources
