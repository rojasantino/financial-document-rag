"""
Step 5: re-score the top-K vector search results with a CrossEncoder,
which reads (query, chunk) together and is far more accurate than pure
vector similarity -- at the cost of being slower, hence only run on a
short candidate list rather than the whole corpus.
"""
from functools import lru_cache
from typing import List, Dict
from sentence_transformers import CrossEncoder
from app.config import settings


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(settings.reranker_model)

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        if not results:
            return []
        pairs = [(query, item["document"]["text"]) for item in results]
        scores = self.model.predict(pairs)

        ranked = [
            {**item, "rerank_score": float(score)}
            for item, score in zip(results, scores)
        ]
        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
