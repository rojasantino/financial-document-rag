"""
Step 2: split page text into overlapping chunks small enough to embed
and to fit into an LLM prompt, while keeping track of which page (and
source file) each chunk came from.
"""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


def create_chunks(pages: List[Dict], source_file: str) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    documents = []
    for page in pages:
        for chunk_text in splitter.split_text(page["text"]):
            documents.append(
                {
                    "text": chunk_text,
                    "page": page["page"],
                    "source": source_file,
                }
            )
    return documents
