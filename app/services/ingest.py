"""
Wires together pdf_loader -> chunker -> embeddings -> vector_store.
Called once per uploaded document.
"""
from app.services.pdf_loader import extract_text_from_pdf, is_scanned_pdf
from app.services.chunker import create_chunks
from app.services.embeddings import get_embedding_model
from app.services.vector_store import get_vector_store


def ingest_pdf(file_path: str, source_file: str) -> int:
    """Ingests one PDF into the vector store. Returns number of chunks added."""
    if is_scanned_pdf(file_path):
        raise ValueError(
            "This looks like a scanned/image-only PDF. Add an OCR step "
            "(see README 'Extending the project') before ingesting it."
        )

    pages = extract_text_from_pdf(file_path)
    chunks = create_chunks(pages, source_file=source_file)

    if not chunks:
        raise ValueError("No extractable text found in this PDF.")

    embedding_model = get_embedding_model()
    vectors = embedding_model.encode([c["text"] for c in chunks])

    store = get_vector_store()
    store.add(vectors, chunks)
    store.save()

    return len(chunks)
