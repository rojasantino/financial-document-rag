"""
Step 7 + full pipeline: question -> embedding -> FAISS -> CrossEncoder
rerank -> grounded prompt -> LLM -> answer with page-level citations.
"""
from typing import Dict
from app.config import settings
from app.services.embeddings import get_embedding_model
from app.services.vector_store import get_vector_store
from app.services.reranker import get_reranker
from app.services.llm import get_llm


def build_prompt(question: str, documents: list) -> str:
    context = "\n\n".join(
        f"[Source: {doc['document']['source']}, Page {doc['document']['page']}]\n"
        f"{doc['document']['text']}"
        for doc in documents
    )

    return f"""You are a financial document analysis assistant.

Answer the user's question using ONLY the context below, which was
retrieved from the user's uploaded documents. Do not use outside
knowledge and do not invent numbers or facts.

If the answer cannot be found in the context, reply exactly:
"I could not find this information in the provided documents."

When you do answer, mention which source/page the figure came from.

Context:
{context}

Question:
{question}

Answer:"""


def answer_question(question: str) -> Dict:
    store = get_vector_store()
    if store.index.ntotal == 0:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    embedding_model = get_embedding_model()
    query_vector = embedding_model.encode([question])[0]

    candidates = store.search(query_vector, top_k=settings.top_k_retrieve)

    reranker = get_reranker()
    reranked = reranker.rerank(question, candidates)
    top_documents = reranked[: settings.top_k_rerank]

    prompt = build_prompt(question, top_documents)
    llm = get_llm()
    answer = llm.generate(prompt)

    sources = [
        {
            "source": item["document"]["source"],
            "page": item["document"]["page"],
            "relevance": round(item["rerank_score"], 3),
        }
        for item in top_documents
    ]

    return {"answer": answer, "sources": sources}
