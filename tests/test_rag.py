import os

from fastapi.testclient import TestClient

from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app.main import app
from app.store import InMemoryStore
from scripts.evaluate_recall import QUERIES, recall_at_k


def test_chunking_overlap_and_empty():
    chunks = chunk_text(" ".join(f"w{i}" for i in range(50)), chunk_size=20, overlap=5)
    assert len(chunks) > 1 and all(len(c.split()) <= 20 for c in chunks)
    assert chunk_text("") == []


def test_embedder_deterministic_and_normalized():
    e = HashingEmbedder(dim=64)
    v1, v2 = e.embed("vector search"), e.embed("vector search")
    assert v1 == v2 and abs(sum(x * x for x in v1) - 1.0) < 1e-6


def test_store_search_ranks_relevant_doc_first():
    store = InMemoryStore(HashingEmbedder())
    store.add("cats", chunk_text("Cats are small domestic animals that sleep a lot."))
    store.add("stocks", chunk_text("Stocks and bonds are financial instruments traded on markets."))
    top = store.search("domestic animals that sleep", k=1)
    assert top[0]["doc_id"] == "cats"


def test_recall_at_k_on_samples():
    store = InMemoryStore(HashingEmbedder())
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
    for name in sorted(os.listdir(data_dir)):
        with open(os.path.join(data_dir, name)) as f:
            store.add(name, chunk_text(f.read()))
    assert recall_at_k(store, QUERIES, 3) == 1.0


def test_store_list_delete_and_min_score():
    store = InMemoryStore(HashingEmbedder())
    store.add("cats", chunk_text("Cats are small domestic animals that sleep a lot."))
    store.add("cats", chunk_text("Cats groom themselves and purr when content."))
    store.add("stocks", chunk_text("Stocks and bonds are financial instruments traded on markets."))
    docs = store.list_documents()
    assert docs == [{"doc_id": "cats", "chunks": 2}, {"doc_id": "stocks", "chunks": 1}]
    # A very high min_score filters out weak/irrelevant matches.
    assert store.search("domestic animals that sleep", k=5, min_score=0.99) == []
    assert store.search("domestic animals that sleep", k=5, min_score=0.0)
    assert store.delete("cats") == 2
    assert store.delete("cats") == 0
    assert store.list_documents() == [{"doc_id": "stocks", "chunks": 1}]


def test_api_documents_list_and_delete():
    client = TestClient(app)
    client.post("/ingest", json={"doc_id": "tmp-doc", "text": "Temporary document about vector databases and embeddings."})
    docs = client.get("/documents").json()["documents"]
    assert any(d["doc_id"] == "tmp-doc" for d in docs)
    removed = client.delete("/documents/tmp-doc").json()
    assert removed["chunks_removed"] >= 1
    docs = client.get("/documents").json()["documents"]
    assert not any(d["doc_id"] == "tmp-doc" for d in docs)


def test_api_ingest_and_search():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ingest", json={"doc_id": "d1", "text": "FastAPI is a Python web framework for building APIs."})
    assert r.json()["chunks_added"] >= 1
    results = client.post("/search", json={"query": "python web framework", "k": 3}).json()["results"]
    assert results and results[0]["doc_id"] == "d1"
