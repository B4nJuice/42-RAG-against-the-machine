from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language
from typing import Any

LANGUAGE_PARAMS = {
    Language.PYTHON: {"chunk_size": 500, "chunk_overlap": 100},
    "default": {"chunk_size": 2000, "chunk_overlap": 100}
}


class ChunkerSelector:
    def __init__(self):
        self.chunkers: dict[Language | str, RecursiveCharacterTextSplitter] =\
            {}

    def get_language(self, file_path: str) -> Language | str:
        ...

    def get_or_create_chunker(
                self,
                language: Language | str
            ) -> RecursiveCharacterTextSplitter:
        if language in self.chunkers:
            return self.chunkers.get(language)

        language_params: dict[str, Any] = LANGUAGE_PARAMS.get(language)
        chunker = RecursiveCharacterTextSplitter().from_language(
                self.language, *language_params
            )

        self.chunkers.update({language: chunker})

        return chunker

    def split_text(self, file_path: str) -> list[str]:
        language: Language | str = self.get_language(file_path)
        chunker: RecursiveCharacterTextSplitter = self.get_or_create_chunker(
                language
            )

        with open(file_path) as f:
            file_content = f.read()
            splitted_text: list[str] = chunker.split_text(file_content)

        return splitted_text
