from itertools import batched

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

from src.chunking.chunk_list import ChunkList
from src.utils.logger import Logger


class Embedder:
    chunks: ChunkList
    model: SentenceTransformer
    batch_size: int

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

    @classmethod
    def from_file(cls, chunk_path: str) -> 'Embedder':
        return cls(chunks=ChunkList.from_file(chunk_path))

    def vectorize_chunks(self) -> Tensor:
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
            )
        ]

        embeddings = torch.cat(embeddings, dim=0)

        for c, t in zip(self.chunks, embeddings):
            print(c.set_tensor(t))

        return embeddings
