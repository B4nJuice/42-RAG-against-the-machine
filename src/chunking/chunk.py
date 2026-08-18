from src.models import MinimalSource, ChunkData

from functools import lru_cache
from collections import Counter
import re
import unicodedata

class Chunk:
    def __init__(
                self,
                file_path: str,
                content: str,
                start_index: int,
                end_index: int
            ):
        self.file_path: str = file_path
        self.content: str = content
        self.start_index: int = start_index
        self.end_index: int = end_index

    @staticmethod
    def tokenize(content: str) -> list[str]:
        text = unicodedata.normalize("NFKC", content)

        raw_tokens = re.findall(r"[A-Za-z0-9_]+", text)

        tokens = []

        for token in raw_tokens:
            lower_token = token.lower()
            tokens.append(lower_token)

            for part in token.split("_"):
                if part:
                    tokens.append(part.lower())

            parts = re.findall(
                r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)|[0-9]+",
                token
            )

            tokens.extend(
                part.lower()
                for part in parts
                if part.lower() != lower_token
            )

        return tokens

    @property
    @lru_cache
    def tokenized_content(self) -> list[str]:
        return self.tokenize(self.content)

    @property
    @lru_cache
    def df(self) -> Counter:
        return Counter(self.tokenized_content)

    @property
    @lru_cache
    def len(self) -> int:
        return len(self.content)

    @property
    def chunk_data(self) -> MinimalSource:
        return ChunkData(
            content = self.content,
            metadata = MinimalSource(
                    file_path = self.file_path,
                    first_character_index = self.start_index,
                    last_character_index = self.end_index,
                )
            )