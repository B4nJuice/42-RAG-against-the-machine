from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language
from typing import Any

LANGUAGE_PARAMS = {
    Language.PYTHON: {"chunk_size": 500, "chunk_overlap": 100},
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

    def get_language(self, file_path: str) -> Language | str:
        file_suffix: str = file_path.split(".")[-1]
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
                language, **language_params
            )

        self.chunkers.update({language: chunker})

        return chunker

    def split_text(self, file_path: str) -> list[str]:
        chunker: RecursiveCharacterTextSplitter = self.get_or_create_chunker(
                file_path
            )

        with open(file_path) as f:
            file_content = f.read()
            splitted_text: list[str] = chunker.split_text(file_content)

        return splitted_text


if __name__ == "__main__":
    selector = ChunkerSelector()
    text = selector.split_text("./data/raw/vllm-0.10.1/tests/basic_correctness/test_basic_correctness.py")
    print("\n".join(text))