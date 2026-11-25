from sentence_transformers import SentenceTransformer

from src.chunking.chunk import Chunk


class Embedder:
    model: SentenceTransformer

    def __init__(self, model: str ="Qwen/Qwen3-Embedding-0.6B") -> None:
        self.model = SentenceTransformer(model)

    def vectorize_chunks(self, chunks: list[Chunk]):
        ...
