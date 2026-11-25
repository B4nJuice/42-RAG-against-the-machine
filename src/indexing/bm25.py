from collections import Counter
from math import log

import numpy as np

from src.chunking.chunk import Chunk
from src.utils.logger import Logger


class BM25:
    chunks: list[Chunk]
    k: float
    b: float
    l: float
    delta: float
    global_df: Counter[str]
    stats_cache: dict[int, bool]
    avg_len: float
    n_chunks: int
    inverted_index: dict[str, list[tuple[int, int]]]
    compared_lens: np.ndarray
    token_boost: dict[str, float]

    def __init__(
        self,
        chunks: list[Chunk],
        k: float = 1.2,
        b: float = 0.75,
        l: float = 1,
        delta: float = 0
    ) -> None:
        self.chunks = chunks
        self.global_df = Counter()

        self.stats_cache = {}

        self.k = k
        self.b = b
        self.l = l
        self.delta = delta

        self.avg_len = 1.0
        self.n_chunks = len(chunks)

        self.inverted_index = {}

        self.compared_lens = np.empty(
            self.n_chunks,
            dtype=np.float32,
        )

        self.token_boost = {}

    def compute_stats(self) -> None:
        total_len = 0

        key = hash(tuple(self.chunks))

        if self.stats_cache.get(key):
            return

        for i, chunk in enumerate(Logger.progress(
                    self.chunks, "Computing BM25 stats", leave=False
                )):
            self.global_df.update(chunk.df.keys())
            total_len += chunk.len

            for token, tf in chunk.df.items():
                self.inverted_index.setdefault(
                    token,
                    [],
                ).append((i, tf))

        self.avg_len = total_len / self.n_chunks

        self._compute_parameter_stats()

        self.stats_cache[key] = True

    def _compute_parameter_stats(self) -> None:
        lengths = np.array(
            [chunk.len for chunk in self.chunks],
            dtype=np.float32,
        )

        self.compared_lens = (
            self.k
            * (
                1
                - self.b
                + self.b * lengths / self.avg_len
            )
        )

        self.token_boost = {
            token: 1 + self.l * log(len(token) + 1)
            for token in self.global_df
        }

    def compute_idf(self, token: str) -> float:
        n_token = self.global_df.get(token, 0)

        return log(
            (self.n_chunks - n_token + 0.5)
            / (n_token + 0.5)
        )

    def get_best_chunk(self, query: str, k: int) -> list[Chunk]:
        tokens = set(Chunk.tokenize(query))

        scores = np.zeros(
            self.n_chunks,
            dtype=np.float32,
        )

        for token in tokens:
            postings = self.inverted_index.get(token)

            if not postings:
                continue

            idf = self.compute_idf(token)
            boost = self.token_boost[token]

            for chunk_index, tf in postings:
                compared_len = self.compared_lens[
                    chunk_index
                ]

                tf_score = tf * (self.k + 1) / (
                    tf + compared_len
                ) + self.delta

                tf_score *= boost

                scores[chunk_index] += tf_score * idf

            k = min(self.n_chunks, k)

            indices = np.argpartition(
                scores,
                -k,
            )[-k:]

            indices = indices[
                np.argsort(scores[indices])[::-1]
            ]

        return [self.chunks[i] for i in indices]
