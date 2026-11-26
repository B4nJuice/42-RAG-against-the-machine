from collections.abc import Iterator

from src.chunking.chunk import Chunk
from src.utils.logger import Logger
from src.utils.models import ChunkData
from src.utils.result_manager import ResultManager


class ChunkList:
    chunks: list[Chunk]
    n_chunks: int
    key: int

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.n_chunks = len(chunks)
        self.chunks.sort(key=lambda c: len(c.content))
        self.key = hash(tuple(self.chunks))

    def __iter__(self) -> Iterator[Chunk]:
            yield from self.chunks

    def get_key(self) -> int:
        return self.key

    @classmethod
    def from_file(cls, chunk_path: str) -> 'ChunkList':
        return cls(chunks=[
            Chunk.from_chunk_data(ChunkData.model_validate(chunk))
            for chunk in Logger.progress(
                ResultManager.read_result(chunk_path), "Loading chunks"
            )
        ])
