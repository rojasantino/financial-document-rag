from fastapi import FastAPI
from app.api import upload, chat

app = FastAPI(
    title="Financial Document Intelligence API",
    description="RAG system for question-answering over financial documents with page-level citations.",
    version="1.0.0",
)

app.include_router(upload.router, tags=["Upload"])
app.include_router(chat.router, tags=["Chat"])


@app.get("/")
def root():
    return {"message": "Financial Document RAG API is running. See /docs for the API reference."}


@app.get("/health")
def health():
    return {"status": "ok"}
