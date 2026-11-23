"""Embeddings with a deterministic local fallback.

HashingEmbedder maps tokens into a fixed-size vector (bag-of-words hashing),
which gives real cosine-similarity behaviour offline. If OPENAI_API_KEY is
set and the SDK is installed, OpenAIEmbedder can be used instead.
"""
import hashlib
import math
import os
import re

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        import openai  # type: ignore

        self._client = openai.OpenAI()
        self.model = model

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


def default_embedder():
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except Exception:
            pass
    return HashingEmbedder()
