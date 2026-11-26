from itertools import batched

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

from src.chunking.chunk import Chunk
from src.chunking.chunk_list import ChunkList
from src.utils.logger import Logger, LogLevel
from src.utils.models import (
    MinimalSearchResults,
    StudentSearchResults,
)
from src.utils.result_manager import ResultManager
from src.utils.terminal import Colors, TerminalStyler


class Embedder:
    chunks: ChunkList
    model: SentenceTransformer
    batch_size: int
    n_chunks: int
    embeddings: Tensor | None
    embeddings_cache: dict[int, bool]

    def __init__(
                self,
                chunks: ChunkList,
                model: str ="sentence-transformers/all-MiniLM-L6-v2",
                batch_size: int = 32,
            ) -> None:
        self.chunks = chunks
        Logger.hook_transformers()
        self.model = SentenceTransformer(
            model,
            model_kwargs={"torch_dtype": torch.float16}
        )
        self.batch_size = batch_size
        self.n_chunks = len(chunks.chunks)
        self.embeddings = None
        self.embeddings_cache = {}

    @classmethod
    def from_file(cls, chunk_path: str) -> 'Embedder':
        return cls(chunks=ChunkList.from_file(chunk_path))

    def search(
                self,
                query: str,
                k: int = 10,
                output_path: str = "data/processed/search_results.json",
                no_output: bool = False
            ) -> list[Chunk]:
        self.vectorize_chunks()

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

    def vectorize_chunks(self) -> Tensor:
        if self.embeddings is not None \
                and self.embeddings_cache.get(self.chunks.get_key()):
            return self.embeddings

        chunks_contents = [c.content for c in self.chunks.chunks]

        embeddings = [
            self.model.encode(
                list(batch),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for batch in Logger.progress(
                batched(chunks_contents, self.batch_size),
                desc="Vectorizing chunks",
                total=(
                    len(chunks_contents) + self.batch_size - 1
                ) // self.batch_size,
                leave=False,
            )
        ]

        self.embeddings = torch.cat(embeddings, dim=0)

        for c, t in zip(self.chunks, self.embeddings):
            c.set_tensor(t)

        self.embeddings_cache[self.chunks.get_key()] = True

        return self.embeddings

    def get_best_chunk(self, query: str, k: int) -> list[Chunk]:
        embeddings = self.vectorize_chunks()

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).to(embeddings.device, embeddings.dtype)

        scores = embeddings @ query_embedding

        k = min(self.n_chunks, k)

        indices = torch.topk(scores, k).indices.tolist()

        return [self.chunks.chunks[i] for i in indices]
