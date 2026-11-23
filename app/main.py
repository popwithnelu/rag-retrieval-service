import os

from fastapi import FastAPI
from pydantic import BaseModel

from .chunking import chunk_text
from .embeddings import default_embedder
from .store import InMemoryStore

app = FastAPI(title="RAG Retrieval Service")
store = InMemoryStore(default_embedder())


class IngestRequest(BaseModel):
    doc_id: str
    text: str


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(store.rows)}


@app.post("/ingest")
def ingest(req: IngestRequest):
    n = store.add(req.doc_id, chunk_text(req.text))
    return {"doc_id": req.doc_id, "chunks_added": n}


@app.post("/search")
def search(req: SearchRequest):
    return {"results": store.search(req.query, req.k)}


@app.post("/ingest-samples")
def ingest_samples():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
    total = 0
    for name in sorted(os.listdir(data_dir)):
        with open(os.path.join(data_dir, name)) as f:
            total += store.add(name, chunk_text(f.read()))
    return {"chunks_added": total}
