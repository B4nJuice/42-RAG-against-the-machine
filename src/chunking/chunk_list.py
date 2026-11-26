from typing import cast

from src.chunking.chunk import Chunk
from src.utils.logger import Logger
from src.utils.models import ChunkData
from src.utils.result_manager import ResultManager


class ChunkList:
    chunks: list[Chunk]
    key: int

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.chunks.sort(key=lambda c: len(c.content))
        self.key = hash(tuple(self.chunks))

    def get_key(self) -> int:
        return self.key

    @classmethod
    def from_file(cls, chunk_path: str) -> 'ChunkList':
        return cls(chunks=[
            Chunk.from_chunk_data(ChunkData.model_validate(chunk))
            for chunk in Logger.progress(
                cast(list[Chunk], ResultManager.read_result(chunk_path)), "Loading chunks"
            )
        ])
