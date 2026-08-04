from chunking.chunk import Chunk
from collections import Counter
from math import log
from functools import lru_cache

class BM25():
    def __init__(
                self, chunks: list[Chunk],
                k: float = 1.2,
                b: float = 0.75,
                l: float = 1
            ) -> None:
        self.chunks: list[Chunk] = chunks
        self.global_df: Counter = Counter()
        self.k = k
        self.b = b
        self.l = l
        self.avg_len: float = 1
        self.n_chunks: int = len(chunks)

    def get_best_chunk(self, query : str) -> tuple[float, Chunk | None]:
        chunks: dict[Chunk, float] = {}
        for c in self.chunks:
            chunks.update({c: self.get_score(query, c)})

        return chunks

    def get_score(self, query: str, chunk: Chunk) -> float:
        tokenized_query: list[str] = Chunk.tokenize(query)
        total_score: float = 0
        for token in tokenized_query:
            total_score += self.compute_tf(token, chunk)\
                * self.compute_idf(query, chunk)

        return total_score

    @lru_cache(maxsize=None)
    def compute_tf(self, token: str, chunk: Chunk) -> float:
        token_freq: int = chunk.df.get(token, 0)
        compared_len: float = self.k * (1 - self.b + self.b * chunk.len / self.avg_len)

        tf_score = token_freq / (token_freq + self.k*compared_len)

        if self.l > 0:
            token_length_boost = 1 + self.l * log(len(token) + 1)
            tf_score *= token_length_boost
        
        return tf_score

    @lru_cache(maxsize=None)
    def compute_idf(self, query: str, chunk: Chunk) -> float:
        n_query: int = self.global_df.get(query, 0)
        return log((self.n_chunks - n_query + 0.5)/(n_query + 0.5))

    def compute_stats(self) -> None:
        total_len: int = 0
        for c in self.chunks:
            self.global_df.update(c.df.keys())
            total_len += c.len

        self.avg_len = total_len / self.n_chunks
