# 💰 Financial Document Intelligence & RAG

A retrieval-augmented generation system that answers questions about financial
documents (annual reports, bank statements, loan documents, policy documents)
**strictly grounded in the uploaded files**, with page-level citations.

## Architecture

```
                         USER
                           │
                           ▼
                    Streamlit Frontend
                           │
                           ▼
                       FastAPI Backend
                           │
              ┌────────────┴─────────────┐
              │                          │
        Document Upload              Question
              │                          │
              ▼                          ▼
         PyMuPDF Parser            Query Embedding
              │                    (MiniLM-L6-v2)
              ▼                          │
        Text Extraction                  │
        (per page)                       │
              ▼                          │
     RecursiveCharacter                  │
        Chunking                         │
              ▼                          │
     Sentence-Transformer                │
        Embeddings                       │
              ▼                          │
        FAISS Vector Index ◄─────────────┘
              │
              ▼
       Top-K Retrieval (10)
              │
              ▼
    CrossEncoder Reranker (MS-MARCO)
              │
              ▼
        Top-N Chunks (4)
              │
              ▼
     Grounded Prompt Builder
              │
              ▼
      LLM (Claude / GPT)
              │
              ▼
      Answer + Page Citations
```

### Why this design

| Stage | Choice | Reason |
|---|---|---|
| Parsing | PyMuPDF | Fast, keeps per-page boundaries for citations |
| Chunking | Recursive character splitter, 800 chars / 150 overlap | Keeps context coherent, overlap avoids cutting facts in half |
| Embeddings | `all-MiniLM-L6-v2` | Small, fast, runs on CPU, good enough for retrieval |
| Vector store | FAISS (in-memory + persisted to disk) | Zero infra to start; swappable for pgvector later |
| Reranker | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Vector similarity alone is noisy; reranking on (query, chunk) pairs meaningfully improves precision |
| LLM | Claude or GPT via a single `LLM` abstraction | Answer generation only — never used as the source of facts |
| Grounding | Prompt explicitly forbids outside knowledge and requires citing source/page | Reduces hallucination, which is the whole point of RAG |

## Project structure

```
financial-document-rag/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── config.py                # env-driven settings
│   ├── api/
│   │   ├── upload.py             # POST /upload
│   │   └── chat.py               # POST /ask
│   └── services/
│       ├── pdf_loader.py         # PDF -> per-page text
│       ├── chunker.py            # text -> overlapping chunks
│       ├── embeddings.py         # text -> vectors
│       ├── vector_store.py       # FAISS index + persistence
│       ├── reranker.py           # CrossEncoder reranking
│       ├── llm.py                # Claude/GPT abstraction
│       ├── ingest.py             # upload -> indexed pipeline
│       └── rag_pipeline.py       # question -> answer pipeline
├── frontend/
│   └── app.py                   # Streamlit UI
├── data/
│   ├── documents/                # uploaded PDFs
│   └── vectorstore/              # persisted FAISS index
├── tests/
│   └── test_basic.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## How to run it — step by step

### Option A: Run locally with Python

**1. Clone/open the project and create a virtual environment**
```bash
cd financial-document-rag
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your LLM key**
```bash
cp .env.example .env
```
Open `.env` and set either:
- `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...`, or
- `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...`

**4. Start the backend API**
```bash
uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` to see the interactive API docs (Swagger UI).
The first request will download the embedding/reranker models (a few hundred MB) —
this only happens once.

**5. Start the frontend (in a second terminal, same venv activated)**
```bash
streamlit run frontend/app.py
```
This opens `http://localhost:8501`.

**6. Use it**
- Upload a financial PDF in the sidebar and click "Index document".
- Ask a question like *"What was revenue in 2025?"* or *"What risks are mentioned in this report?"*
- The answer appears with the source filename, page number, and a relevance score.

### Option B: Run with Docker Compose (backend + frontend together)

```bash
cp .env.example .env     # fill in your API key
docker compose up --build
```
- Backend: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

### Running the tests

```bash
pip install pytest
pytest tests/
```
(`test_basic.py` checks chunking and prompt construction — it doesn't call any
external API, so it runs offline and free.)

## Using the API directly (no UI)

```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@Annual_Report_2025.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the company revenue in 2025?"}'
```
