"""Vector store.

InMemoryStore is the offline default. In production the same interface maps
to PostgreSQL + pgvector, e.g.:

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE chunks (id serial PRIMARY KEY, doc_id text, text text,
                         embedding vector(256));
    -- query: SELECT text FROM chunks ORDER BY embedding <=> $1 LIMIT $2;
"""
import math


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized


class InMemoryStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.rows: list[dict] = []

    def add(self, doc_id: str, chunks: list[str]) -> int:
        for ch in chunks:
            self.rows.append({"doc_id": doc_id, "text": ch,
                              "embedding": self.embedder.embed(ch)})
        return len(chunks)

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[dict]:
        q = self.embedder.embed(query)
        scored = sorted(self.rows, key=lambda r: -_cosine(q, r["embedding"]))
        results = []
        for r in scored:
            score = _cosine(q, r["embedding"])
            if score < min_score:
                continue
            results.append({"doc_id": r["doc_id"], "text": r["text"],
                            "score": round(score, 4)})
            if len(results) >= k:
                break
        return results

    def list_documents(self) -> list[dict]:
        counts: dict[str, int] = {}
        for r in self.rows:
            counts[r["doc_id"]] = counts.get(r["doc_id"], 0) + 1
        return [{"doc_id": doc_id, "chunks": n}
                for doc_id, n in sorted(counts.items())]

    def delete(self, doc_id: str) -> int:
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["doc_id"] != doc_id]
        return before - len(self.rows)
