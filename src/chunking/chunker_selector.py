from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language
from langchain_core.documents import Document
from .chunk import Chunk
from typing import Any
from os import walk

LANGUAGE_PARAMS = {
    Language.PYTHON: {"chunk_size": 1000, "chunk_overlap": 100},
    Language.CPP: {"chunk_size": 1000, "chunk_overlap": 100},
    Language.JAVA: {"chunk_size": 1000, "chunk_overlap": 100},
    Language.PHP: {"chunk_size": 1000, "chunk_overlap": 100},
    Language.MARKDOWN: {"chunk_size": 1000, "chunk_overlap": 100},
    Language.HTML: {"chunk_size": 1000, "chunk_overlap": 100},
    Language.C: {"chunk_size": 1000, "chunk_overlap": 100},
    "default": {"chunk_size": 2000, "chunk_overlap": 100}
}

LANGUAGE_TRANSLATION = {
    "py" : Language.PYTHON,
    "cpp" : Language.CPP,
    "java" : Language.JAVA,
    "php" : Language.PHP,
    "md" : Language.MARKDOWN,
    "html" : Language.HTML,
    "c" : Language.C
}

class ChunkerSelector:
    def __init__(self):
        self.chunkers: dict[Language | str, RecursiveCharacterTextSplitter] =\
            {}

    @staticmethod
    def get_suffix(file_name: str) -> str:
        return file_name.split(".")[-1]

    def get_language(self, file_path: str) -> Language | str:
        file_suffix: str = self.get_suffix(file_path)
        language : Language | str = LANGUAGE_TRANSLATION.get(
                file_suffix,
                "default"
            )
        return language

    def get_or_create_chunker(
                self,
                file_path: str
            ) -> RecursiveCharacterTextSplitter:
        language: Language | str = self.get_language(file_path)

        if language in self.chunkers:
            return self.chunkers.get(language)

        language_params: dict[str, Any] = LANGUAGE_PARAMS.get(language)
        chunker = RecursiveCharacterTextSplitter().from_language(
                language, **language_params, add_start_index=True
            )

        self.chunkers.update({language: chunker})

        return chunker

    def split_text(self, file_path: str) -> list[Chunk]:
        chunker: RecursiveCharacterTextSplitter = self.get_or_create_chunker(
                file_path
            )

        with open(file_path) as f:
            file_content = f.read()
            documents: Document = chunker.create_documents([file_content])

        chunks: list[Chunk] = []

        for doc in documents:
            start_index = doc.metadata.get("start_index", 0)
            end_index = start_index + len(doc.page_content)
            chunk = Chunk(file_path, doc.page_content, start_index, end_index)
            chunks.append(chunk)

        return chunks

    def folder_chunking(self, folder_path: str) -> list[Chunk]:
        chunks: list[Chunk] = []

        for (dirpath, _, filenames) in walk(folder_path):
            for file_name in filenames:
                suffix: str = self.get_suffix(file_name)
                if suffix not in LANGUAGE_TRANSLATION.keys():
                    continue
                chunks += self.split_text(f"{dirpath}/{file_name}")

        return chunks

