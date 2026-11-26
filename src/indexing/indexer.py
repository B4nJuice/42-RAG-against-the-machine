from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.chunking.chunk import Chunk
from src.chunking.chunk_list import ChunkList


class Indexer(ABC):
    chunks: ChunkList
    _cache: dict[tuple[str, int], Any]

    def __init__(self, chunks: ChunkList) -> None:
        self.chunks = chunks
        self._cache = {}

    @staticmethod
    def chunk_cache[T](method: Callable[..., T]) -> Callable[..., T]:
        @wraps(method)
        def wrapper(self: 'Indexer', *args: Any, **kwargs: Any) -> T:
            key = (method.__qualname__, self.chunks.get_key())
            if key not in self._cache:
                self._cache[key] = method(self, *args, **kwargs)
            return self._cache[key]
        return wrapper

    @classmethod
    def from_file(cls, chunk_path: str, **kwargs: Any) -> 'Indexer':
        return cls(chunks=ChunkList.from_file(chunk_path, **kwargs))

    @abstractmethod
    def search(
                self,
                query: str,
                k: int = 10,
                output_path: str = "data/processed/search_results.json",
                no_output: bool = False
            ) -> list[Chunk]:
        pass

    @abstractmethod
    def get_best_chunk(self, query: str, k: int) -> list[Chunk]:
        pass

    @abstractmethod
    def compute(self) -> None:
        pass
