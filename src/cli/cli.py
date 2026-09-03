from src.chunking.chunker_selector import ChunkerSelector
from src.utils.terminal import Colors, TerminalStyler
from src.utils.models import RagDataset, ChunkData
from src.utils.logger import Logger, LogLevel
from src.chunking.chunk import Chunk
from src.indexing.bm25 import BM25

from pydantic import ValidationError
from pathlib import Path
import numpy as np 
import json
import os


def create_dataset(dataset_path: str) -> RagDataset:
    with open(dataset_path) as f:
        return RagDataset.model_validate_json(f.read())


def normalize_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(Path.cwd())
        except ValueError:
            pass

    return p.as_posix().lstrip("./")

class Cli:
    @staticmethod
    def index(
                input_path: str = "./data/raw/",
                output_path: str = "./data/processed/chunks.json",
                max_chunk_size: int = 2000,
                max_chunk_overlap: int = 200,
                debug: bool = False
            ):
        if debug:
            os.environ["DEBUG"] = "1"
        try:
            chunker_selector: ChunkerSelector = ChunkerSelector(
                    max_chunk_size=max_chunk_size,
                    max_chunk_overlap=max_chunk_overlap
                )

            Logger.log("chunker selector initialized.", LogLevel.DEBUG)
            chunks: Chunk = chunker_selector.folder_chunking(input_path)
            Logger.log(
                    f"successfully chunked {len(chunks)} chunks from {input_path}."
                )

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            Logger.log(f"path: {output_path} initialized.", LogLevel.DEBUG)

            formatted_chunks: list[str] = [c.chunk_data.model_dump() for c in chunks]

            with open(output_path, "w") as f:
                json.dump(formatted_chunks, f, indent="\t", ensure_ascii=False)
                f.write("\n")

            Logger.log(f"chunks writted in {output_path}.", LogLevel.DEBUG)

        except Exception as e:
            Logger.log(e, LogLevel.ERROR)

    @staticmethod
    def search_dataset(
                dataset_path: str =\
                    "./data/public/UnansweredQuestions/dataset_docs_public.json",
                k: int = 10,
                debug: bool = False
            ):
        if debug:
            os.environ["DEBUG"] = "1"
        try:
            dataset: RagDataset = create_dataset(dataset_path)
            Logger.log(f"dataset at {dataset_path} parsed.", LogLevel.DEBUG)

        except Exception as e:
            Logger.log(e, LogLevel.ERROR)


    @staticmethod
    def search(
                query: str,
                k: int = 10,
                chunk_path: str = "./data/processed/chunks.json",
                force_index: bool = False,
                debug: bool = False
            ) -> None:
        if debug:
                os.environ["DEBUG"] = "1"    
        if force_index:
            Logger.log("indexing forced")
            index(output_path=chunk_path)

        chunks: list[Chunk] = []

        try:
            with open(chunk_path) as f:
                for chunk in json.loads(f.read()):
                    chunks.append(Chunk.from_chunk_data(ChunkData.model_validate(chunk)))
        except Exception as e:
            Logger.log(e, LogLevel.ERROR)

        bm25: BM25 = BM25(chunks=chunks)

        bm25.compute_stats()

        scores = bm25.get_best_chunk(query, k)

        top_k = min(k, len(scores))

        if top_k != k:
            Logger.log(f"Impossible to retrieve top {k} sources because there\
    is only {top_k} chunks.", LogLevel.DEBUG)

        Logger.log(f"Top {top_k} sources for \"{query}\" :")

        for top, chunk in enumerate(scores):
            Logger.log(
                TerminalStyler.colored_text(
                    [Colors.BLUE], f"[{top + 1}] "
                ) + chunk.file_path + f" [{chunk.start_index}:{chunk.end_index}]"
            )