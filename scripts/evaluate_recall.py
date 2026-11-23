"""Recall@k evaluation over the bundled sample docs.

Run: python -m scripts.evaluate_recall
Each query has one gold document; recall@k is 1.0 if it appears in the top k.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app.store import InMemoryStore

QUERIES = [
    ("how are similar texts represented as vectors", "embeddings.txt"),
    ("storing vectors in postgres with sql similarity search", "vector_databases.txt"),
    ("splitting long documents into overlapping passages", "chunking.txt"),
]


def recall_at_k(store, queries, k):
    hits = 0
    for query, gold in queries:
        top = store.search(query, k)
        if any(r["doc_id"] == gold for r in top):
            hits += 1
    return hits / len(queries)


def main():
    store = InMemoryStore(HashingEmbedder())
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
    for name in sorted(os.listdir(data_dir)):
        with open(os.path.join(data_dir, name)) as f:
            store.add(name, chunk_text(f.read()))
    for k in (1, 3, 5):
        print(f"recall@{k}: {recall_at_k(store, QUERIES, k):.2f}")


if __name__ == "__main__":
    main()
