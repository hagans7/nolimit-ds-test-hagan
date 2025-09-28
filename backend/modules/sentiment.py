# modules/sentiment.py
import torch
import logging
from transformers import pipeline
from dataclasses import dataclass
from typing import Dict, List
from .preprocessing import IndonesianPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    text: str
    processed_text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]


class SentimentAnalyzer:
    def __init__(self, sentiment_model: str = "niejanee/tokopedia-sentiment-analysis-indobert",
                 preprocessor: IndonesianPreprocessor | None = None):
        self.pre = preprocessor or IndonesianPreprocessor()
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading Sentiment model: {sentiment_model}")
        self.pipe = pipeline("sentiment-analysis", model=sentiment_model,
                             device=self.device, return_all_scores=True)
        self.label_map = {"LABEL_0": "negative",
                          "LABEL_1": "neutral", "LABEL_2": "positive"}

    def run(self, texts: List[str]) -> List[SentimentResult]:
        processed = [self.pre.process(t, "sentiment") for t in texts]
        cleaned = [p.cleaned if p.cleaned else p.original for p in processed]
        outputs = self.pipe(cleaned)
        results: List[SentimentResult] = []
        for orig, proc, out in zip(texts, processed, outputs):
            scores = {self.label_map.get(
                s['label'], s['label'].lower()): s['score'] for s in out}
            top = max(scores, key=scores.get)
            results.append(SentimentResult(
                orig, proc.cleaned, top, scores[top], scores))
        return results
