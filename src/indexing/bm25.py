from collections import Counter
from math import log

import numpy as np

from src.chunking.chunk import Chunk
from src.chunking.chunk_list import ChunkList
from src.utils.logger import Logger, LogLevel
from src.utils.models import (
    MinimalSearchResults,
    StudentSearchResults,
)
from src.utils.result_manager import ResultManager
from src.utils.terminal import Colors, TerminalStyler


class BM25:
    chunks: ChunkList
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
        chunks: ChunkList,
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
        self.n_chunks = len(chunks.chunks)

        self.inverted_index = {}

        self.compared_lens = np.empty(
            self.n_chunks,
            dtype=np.float32,
        )

        self.token_boost = {}

    @classmethod
    def from_file(cls, chunk_path: str) -> 'BM25':
        return cls(chunks=ChunkList.from_file(chunk_path))

    def search(
                self,
                query: str,
                k: int = 10,
                output_path: str = "data/processed/search_results.json",
                no_output: bool = False
            ) -> list[Chunk]:
        self.compute_stats()

        scores: list[Chunk] = self.get_best_chunk(query, k)

        top_k = min(k, len(scores))

        if top_k != k:
            Logger.log(f"Impossible to retrieve top {k} sources because there\
    is only {top_k} chunks.", LogLevel.DEBUG)

        if no_output:
            return scores

        search_results = StudentSearchResults(
            search_results = [
                MinimalSearchResults(
                    question_id = "None",
                    question = query,
                    retrieved_sources = [c.minimal_source for c in scores]
                )
            ],
        k = k)

        Logger.log(f"Top {top_k} sources for \"{query}\" :")

        for top, chunk in enumerate(scores):
            Logger.log(
                TerminalStyler.colored_text(
                    [Colors.BLUE], f"[{top + 1}] "
                ) + chunk.file_path + f" [{chunk.start_index}:{chunk.end_index}]"
            )

        ResultManager.store_result(output_path, search_results)

        return scores

    def compute_stats(self) -> None:
        total_len = 0

        if self.stats_cache.get(self.chunks.get_key()):
            return

        for i, chunk in enumerate(Logger.progress(
                    self.chunks.chunks, "Computing BM25 stats", leave=False
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

        self.stats_cache[self.chunks.get_key()] = True

    def _compute_parameter_stats(self) -> None:
        lengths = np.array(
            [chunk.len for chunk in self.chunks.chunks],
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

        return [self.chunks.chunks[i] for i in indices]
