"""Split documents into overlapping chunks for embedding."""


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]
