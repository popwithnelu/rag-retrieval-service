# RAG Retrieval Service

A retrieval-augmented generation (RAG) building block: ingest documents,
chunk and embed them, and query them by semantic similarity.

## Features

- **Ingestion + chunking** with configurable size/overlap (`app/chunking.py`)
- **Embeddings** — deterministic local hashing embedder as an offline fallback;
  OpenAI embeddings when `OPENAI_API_KEY` is set (`app/embeddings.py`)
- **Vector search API** — `/ingest`, `/search`, `/ingest-samples`, `/health`
- **pgvector-ready design** — the store interface maps directly to PostgreSQL +
  pgvector (`embedding <=> query` cosine search; see `app/store.py`)
- **Recall@k evaluation** — `python -m scripts.evaluate_recall`

## Tech stack
Python, FastAPI, Pydantic; optional OpenAI API, PostgreSQL/pgvector.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
curl -X POST localhost:8000/ingest-samples
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "storing vectors in postgres", "k": 3}'
python -m scripts.evaluate_recall
```

## Tests

```bash
python -m pytest   # fully offline with the local embedder
```

## Project structure

```
app/ (embeddings, chunking, store, main) · data/sample_docs/ · scripts/evaluate_recall.py · tests/test_rag.py
```

MIT License — see LICENSE.
