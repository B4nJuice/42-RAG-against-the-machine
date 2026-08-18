from src.chunking.chunker_selector import ChunkerSelector
from src.utils.terminal import Colors
from src.utils.logger import Logger, LogLevel
from src.indexing.bm25 import BM25
from src.models import RagDataset

from pydantic import ValidationError
from pathlib import Path
import numpy as np 
import json
import fire

def normalize_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(Path.cwd())
        except ValueError:
            pass

    return p.as_posix().lstrip("./")

def evaluate_bm25_on_dataset():
    
    selector = ChunkerSelector()
    chunks = selector.folder_chunking("./data/raw/vllm-0.10.1")
    
    bm25 = BM25(chunks, b=0.6, k=1.2, l=0.9)
    bm25.compute_stats()
    print(bm25.n_chunks)
    
    with open("./data/public/AnsweredQuestions/dataset_docs_public.json", "r") as f:
        dataset = json.load(f)
    
    questions = dataset["rag_questions"]
    
    rankings = []
    successes = 0
    
    for idx, q in enumerate(questions):
        question = q["question"]
        expected_file_paths = {
            normalize_path(source["file_path"])
            for source in q.get("sources", [])
            if source.get("file_path")
        }

        scores = bm25.get_best_chunk(question)

        top_k = min(10, len(scores))

        indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        indices = indices[
            np.argsort(scores[indices])[::-1]
        ]

        ranked_chunks = [
            bm25.chunks[i]
            for i in indices
        ]
        
        position = None
        for rank, chunk in enumerate(ranked_chunks):
            chunk_path = normalize_path(chunk.file_path)
            if chunk_path in expected_file_paths:
                position = rank + 1  # Position 1-indexed
                break
        
        if position:
            rankings.append(position)
            if position == 1:
                successes += 1
            print(f"Q{idx+1}: '{question[:60]}...' -> Position {position}")
        else:
            top_path = normalize_path(ranked_chunks[0].file_path) if ranked_chunks else "N/A"
            expected_preview = next(iter(expected_file_paths), "N/A")
            print(
                f"Q{idx+1}: '{question[:60]}...' -> CHUNK NOT FOUND "
                f"(attendu: {expected_preview}, top-1: {top_path})"
            )
    
    if rankings:
        avg_ranking = sum(rankings) / len(rankings)
        success_rate = (successes / len(questions)) * 100
        top3_successes = sum(1 for r in rankings if r <= 3)
        top5_successes = sum(1 for r in rankings if r <= 5)
        top10_successes = sum(1 for r in rankings if r <= 10)
        top3_rate = (top3_successes / len(questions)) * 100
        top5_rate = (top5_successes / len(questions)) * 100
        top10_rate = (top10_successes / len(questions)) * 100
        
        print("\n" + "="*60)
        print(f"RESULTS:")
        print(f"  Questions: {len(questions)}")
        print(f"  Average ranking: {avg_ranking:.2f}")
        print(f"  Recall@1: {success_rate:.2f}%")
        print(f"  Success@1: {successes}/{len(questions)}")
        print(f"  Recall@3: {top3_rate:.2f}%")
        print(f"  Success@3: {top3_successes}/{len(questions)}")
        print(f"  Recall@5: {top5_rate:.2f}%")
        print(f"  Success@5: {top5_successes}/{len(questions)}")
        print(f"  Recall@10: {top10_rate:.2f}%")
        print(f"  Success@10: {top10_successes}/{len(questions)}")
        print("="*60)
    else:
        print("No results")


def create_dataset(input_path: str) -> RagDataset:
    with open(input_path) as f:
        return RagDataset.model_validate_json(f.read())

def index(
            input_path: str = "./data/public/UnansweredQuestions/dataset_docs_public.json",
            output_path: str = "./data/processed/",
            max_chunk_size: int | None = None
        ):
    try:
        dataset: RagDataset = create_dataset(input_path)
    except ValidationError as e:
        Logger.log(e, LogLevel.ERROR)
    except Exception as e:
        print(e)

def search():
    ...

if __name__ == "__main__":
    fire.Fire({
        "index": index,
        "search": search,
    })