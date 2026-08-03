from chunking.chunk import Chunk
from collections import Counter

class BM25():
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks: list[Chunk] = chunks
        self.global_df: Counter = Counter()

    def compute_stats(self) -> None:
        for c in self.chunks:
            self.global_df.update(c.df.keys())
