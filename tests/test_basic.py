from app.services.chunker import create_chunks
from app.services.rag_pipeline import build_prompt


def test_create_chunks_preserves_page_and_source():
    pages = [{"page": 1, "text": "Revenue grew 15% year over year. " * 40}]
    chunks = create_chunks(pages, source_file="report.pdf")

    assert len(chunks) > 0
    assert all(c["page"] == 1 for c in chunks)
    assert all(c["source"] == "report.pdf" for c in chunks)


def test_build_prompt_includes_context_and_question():
    docs = [
        {"document": {"text": "Revenue was $150M in 2025.", "source": "report.pdf", "page": 42}}
    ]
    prompt = build_prompt("What was revenue in 2025?", docs)

    assert "report.pdf" in prompt
    assert "Page 42" in prompt
    assert "What was revenue in 2025?" in prompt
    assert "I could not find this information" in prompt
