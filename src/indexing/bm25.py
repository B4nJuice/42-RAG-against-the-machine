from chunking.chunk import Chunk
from collections import Counter
from math import log
import numpy as np


class BM25:
    def __init__(
        self,
        chunks: list[Chunk],
        k: float = 1.2,
        b: float = 0.75,
        l: float = 1,
    ) -> None:
        self.chunks = chunks
        self.global_df: Counter = Counter()

        self.k = k
        self.b = b
        self.l = l

        self.avg_len = 1.0
        self.n_chunks = len(chunks)

        self.inverted_index: dict[
            str,
            list[tuple[int, int]],
        ] = {}

        self.compared_lens = np.empty(
            self.n_chunks,
            dtype=np.float32,
        )

        self.token_boost: dict[str, float] = {}

    def compute_stats(self) -> None:
        total_len = 0

        for i, chunk in enumerate(self.chunks):
            self.global_df.update(chunk.df.keys())
            total_len += chunk.len

            for token, tf in chunk.df.items():
                self.inverted_index.setdefault(
                    token,
                    [],
                ).append((i, tf))

        self.avg_len = total_len / self.n_chunks

        self._compute_parameter_stats()

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

    def get_best_chunk(self, query: str) -> np.ndarray:
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

                tf_score = tf / (
                    tf + compared_len
                )

                tf_score *= boost

                scores[chunk_index] += (
                    tf_score * idf
                )

        return scores