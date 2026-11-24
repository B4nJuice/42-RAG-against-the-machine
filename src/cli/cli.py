import os
from typing import Any

from src.chunking.chunk import Chunk
from src.chunking.chunker_selector import ChunkerSelector
from src.generating.generate import Generator
from src.indexing.bm25 import BM25
from src.utils.logger import Logger, LogLevel
from src.utils.models import (
    ChunkData,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.utils.result_manager import ResultManager
from src.utils.terminal import Colors, TerminalStyler


class Cli:
    def __init__(self) -> None:
        self.generator: Generator = Generator()

    def index(
                self,
                input_path: str = "data/raw/",
                output_path: str = "data/processed/chunks.json",
                max_chunk_size: int = 2000,
                max_chunk_overlap: int = 200,
                debug: bool = False
            ) -> None:
        if debug:
            os.environ["DEBUG"] = "1"
        try:
            chunker_selector: ChunkerSelector = ChunkerSelector(
                    max_chunk_size=max_chunk_size,
                    max_chunk_overlap=max_chunk_overlap
                )

            Logger.log("chunker selector initialized.", LogLevel.DEBUG)
            chunks: list[Chunk] = chunker_selector.folder_chunking(input_path)
            Logger.log(
                    f"Successfully chunked {len(chunks)} chunks from {input_path}."
                )

            formatted_chunks: list[dict[str, Any]] = [
                c.chunk_data.model_dump()
                for c in Logger.progress(chunks, "Formatting chunks")
            ]

            ResultManager.store_result(output_path, formatted_chunks)

            Logger.log(f"Chunks writted in {output_path}.", LogLevel.DEBUG)

        except Exception as e:
            Logger.log(e, LogLevel.ERROR)

    def search_dataset(
                self,
                dataset_path: str =\
                "data/public/UnansweredQuestions/dataset_docs_public.json",
                chunk_path: str = "data/processed/chunks.json",
                output_path: str = "data/processed/search_results.json",
                force_index: bool = False,
                k: int = 10,
                debug: bool = False
            ) -> None:
        if debug:
            os.environ["DEBUG"] = "1"
        if force_index:
            Logger.log("Indexing forced", LogLevel.DEBUG)
            self.index(output_path=chunk_path)

        dataset = RagDataset.model_validate(
            ResultManager.read_result(dataset_path)
        )
        Logger.log(f"Dataset at {dataset_path} parsed.", LogLevel.DEBUG)

        chunks: list[Chunk] = []

        for chunk in Logger.progress(
                    ResultManager.read_result(chunk_path), "Loading chunks"
                ):
            chunks.append(
                Chunk.from_chunk_data(ChunkData.model_validate(chunk))
            )

        search_results = []

        bm25: BM25 = BM25(chunks=chunks)

        for question in Logger.progress(
                    dataset.rag_questions, "Searching questions"
                ):
            scores = self.search(
                question.question,
                k=k,
                chunk_path=chunk_path,
                no_output=True,
                chunks=chunks,
                bm25=bm25,
            )

            search_results.append(MinimalSearchResults(
                question_id = question.question_id,
                question = question.question,
                retrieved_sources = [c.minimal_source for c in scores]
            ))

            ResultManager.store_result(output_path, StudentSearchResults(
                    search_results = search_results,
                    k = k
                )
            )

    def search(
                self,
                query: str,
                k: int = 10,
                chunk_path: str = "data/processed/chunks.json",
                output_path: str = "data/processed/search_results.json",
                chunks: list[Chunk] = [],
                bm25: BM25 | None = None,
                force_index: bool = False,
                no_output: bool = False,
                debug: bool = False
            ) -> list[Chunk]:
        if debug:
                os.environ["DEBUG"] = "1"
        if force_index:
            Logger.log("Indexing forced", LogLevel.DEBUG)
            self.index(output_path=chunk_path)

        if not chunks:
            for chunk in Logger.progress(
                        ResultManager.read_result(chunk_path), "Loading chunks"
                    ):
                chunks.append(
                    Chunk.from_chunk_data(ChunkData.model_validate(chunk))
                )

        if not bm25:
            bm25 = BM25(chunks=chunks)

        bm25.compute_stats()

        scores: list[Chunk] = bm25.get_best_chunk(query, k)

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

    def answer_dataset(
                self,
                dataset_path: str =\
                "data/public/UnansweredQuestions/dataset_docs_public.json",
                chunk_path: str = "data/processed/chunks.json",
                output_path: str = "data/processed/answer_results.json",
                force_index: bool = False,
                k: int = 10,
                debug: bool = False
            ) -> None:
        if debug:
            os.environ["DEBUG"] = "1"
        if force_index:
            Logger.log("Indexing forced", LogLevel.DEBUG)
            self.index(output_path=chunk_path)

        dataset = RagDataset.model_validate(
            ResultManager.read_result(dataset_path)
        )
        Logger.log(f"Dataset at {dataset_path} parsed.", LogLevel.DEBUG)

        chunks: list[Chunk] = []

        for chunk in Logger.progress(
                    ResultManager.read_result(chunk_path), "Loading chunks"
                ):
            chunks.append(
                Chunk.from_chunk_data(ChunkData.model_validate(chunk))
            )

        answers = []
        bm25: BM25 = BM25(chunks=chunks)

        for question in Logger.progress(
                    dataset.rag_questions, "Answering questions"
                ):
            rephrased_query = question.question + "\n" +\
                self.generator.rephrase(question.question)
            scores = self.search(
                question.question,
                k=k,
                chunk_path=chunk_path,
                no_output=True,
                chunks=chunks,
                bm25=bm25,
            )
            answer = MinimalAnswer(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[c.minimal_source for c in scores],
                answer=self.generator.answer(
                    rephrased_query,
                    scores
                ).strip()
            )
            answers.append(answer)

            ResultManager.store_result(
                output_path,
                StudentSearchResultsAndAnswer(
                    search_results=answers,
                    k=len(answers)
                )
            )

    def answer(
                self,
                query: str,
                k: int = 10,
                output_path: str = "data/processed/answer_results.json",
                force_index: bool = False,
                no_output: bool = False,
                debug: bool = False
            ) -> MinimalAnswer:

        if debug:
                os.environ["DEBUG"] = "1"

        rephrased_query = query +"\n" + self.generator.rephrase(query)

        chunks: list[Chunk] = self.search(
                query,
                k=k,
                force_index=force_index,
                no_output=True
            )

        retrieved_sources = [c.minimal_source for c in chunks]

        answer_str = self.generator.answer(
                        rephrased_query,
                        chunks
                    ).strip()

        answer = MinimalAnswer(
            question = query,
            question_id = "None",
            retrieved_sources=retrieved_sources,
            answer = answer_str
        )

        if no_output:
            return answer

        Logger.log(f"Answer for \"{query}\":")
        Logger.log(answer_str)

        ResultManager.store_result(output_path, StudentSearchResultsAndAnswer(
                search_results = [answer],
                k = 1
            )
        )

        return answer
