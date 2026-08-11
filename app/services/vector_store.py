"""
Step 4: a thin, persisted wrapper around a FAISS inner-product index.
Swap this class out for a pgvector-backed implementation later without
touching any other part of the pipeline (see README "Going to production").
"""
import os
import pickle
from functools import lru_cache
from typing import List, Dict

import faiss
import numpy as np

from app.config import settings

INDEX_PATH = os.path.join(settings.vectorstore_dir, "index.faiss")
DOCS_PATH = os.path.join(settings.vectorstore_dir, "documents.pkl")


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Dict] = []

    def add(self, embeddings, documents: List[Dict]):
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding, top_k: int = 10) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        query_embedding = np.asarray([query_embedding], dtype="float32")
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), "document": self.documents[idx]})
        return results

    def save(self):
        os.makedirs(settings.vectorstore_dir, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, dimension: int) -> "VectorStore":
        store = cls(dimension)
        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            store.index = faiss.read_index(INDEX_PATH)
            with open(DOCS_PATH, "rb") as f:
                store.documents = pickle.load(f)
        return store


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    # all-MiniLM-L6-v2 produces 384-dim embeddings
    return VectorStore.load(dimension=384)
