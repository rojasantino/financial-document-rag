"""
Central configuration for the Financial Document RAG system.
All values are loaded from environment variables / .env file so
nothing sensitive is hard-coded.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Embeddings / retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieve: int = 10
    top_k_rerank: int = 4

    # Storage
    documents_dir: str = "data/documents"
    vectorstore_dir: str = "data/vectorstore"


settings = Settings()
