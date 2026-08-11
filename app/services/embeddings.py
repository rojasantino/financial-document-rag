"""
Step 3: text -> vector. Loaded once and reused (loading a transformer
model per request would be far too slow).
"""
from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model = SentenceTransformer(model_name or settings.embedding_model)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
