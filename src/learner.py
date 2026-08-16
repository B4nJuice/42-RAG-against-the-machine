from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import random
import json
import os
import time

from chunking.chunker_selector import ChunkerSelector
from chunking.chunk import Chunk
from indexing.bm25 import BM25


PARAM_RANGES = {
    "k": (0.1, 3.0),
    "b": (0.0, 2.0),
    "l": (0.0, 5.0),
}


def normalize_path(path: str) -> str:
    p = Path(path)

    if p.is_absolute():
        try:
            p = p.relative_to(Path.cwd())
        except ValueError:
            pass

    return p.as_posix().lstrip("./")


def evaluate_bm25_on_dataset(
    chunks: list[Chunk],
    avg_len: float,
    global_df: Counter,
    inverted_index: dict[str, list[tuple[int, int]]],
    dataset: dict[Any, Any],
    **params: Any,
) -> dict[str, Any]:

    bm25 = BM25(chunks, **params)

    bm25.avg_len = avg_len
    bm25.global_df = global_df
    bm25.inverted_index = inverted_index
    bm25._compute_parameter_stats()

    questions = dataset["rag_questions"]

    rankings = []
    successes = 0

    for q in questions:
        question = q["question"]

        expected_file_paths = {
            normalize_path(source["file_path"])
            for source in q.get("sources", [])
            if source.get("file_path")
        }

        scores = bm25.get_best_chunk(question)

        top_k = min(10, len(scores))

        if top_k == 0:
            continue

        indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        indices = indices[
            np.argsort(scores[indices])[::-1]
        ]

        for rank, index in enumerate(indices, start=1):
            chunk_path = normalize_path(
                bm25.chunks[index].file_path
            )

            if chunk_path in expected_file_paths:
                rankings.append(rank)

                if rank == 1:
                    successes += 1

                break

    if not rankings:
        return {
            "avg_ranking": 0.0,
            "success_rate": 0.0,
            "top3_successes": 0,
            "top5_successes": 0,
            "top10_successes": 0,
            "top3_rate": 0.0,
            "top5_rate": 0.0,
            "top10_rate": 0.0,
        }

    avg_ranking = sum(rankings) / len(rankings)

    success_rate = (
        successes / len(questions)
    ) * 100

    top3_successes = sum(
        1 for rank in rankings
        if rank <= 3
    )

    top5_successes = sum(
        1 for rank in rankings
        if rank <= 5
    )

    top10_successes = sum(
        1 for rank in rankings
        if rank <= 10
    )

    top3_rate = (
        top3_successes / len(questions)
    ) * 100

    top5_rate = (
        top5_successes / len(questions)
    ) * 100

    top10_rate = (
        top10_successes / len(questions)
    ) * 100

    return {
        "avg_ranking": avg_ranking,
        "success_rate": success_rate,
        "top3_successes": top3_successes,
        "top5_successes": top5_successes,
        "top10_successes": top10_successes,
        "top3_rate": top3_rate,
        "top5_rate": top5_rate,
        "top10_rate": top10_rate,
    }


def compete_worker(
    index: int,
    total: int,
    params: dict[str, Any],
    chunks: list[Chunk],
    avg_len: float,
    global_df: Counter,
    inverted_index: dict[str, list[tuple[int, int]]],
    dataset: dict[Any, Any],
) -> dict[str, Any]:

    start = time.perf_counter()

    print(
        f"[START {index}/{total}] "
        f"k={params['k']:.4f} "
        f"b={params['b']:.4f} "
        f"l={params['l']:.4f}",
        flush=True,
    )

    evaluation = evaluate_bm25_on_dataset(
        chunks=chunks,
        avg_len=avg_len,
        global_df=global_df,
        inverted_index=inverted_index,
        dataset=dataset,
        **params,
    )

    elapsed = time.perf_counter() - start

    result = {
        **params,
        **evaluation,
    }

    print(
        f"[DONE  {index}/{total}] "
        f"top5={result['top5_rate']:.2f}% "
        f"top3={result['top3_rate']:.2f}% "
        f"time={elapsed:.2f}s",
        flush=True,
    )

    return result


def compete_generation(
    generation: list[dict[str, Any]],
    chunks: list[Chunk],
    dataset: dict[Any, Any],
    selector: ChunkerSelector,
) -> list[dict[str, Any]]:

    total = len(generation)

    print(
        f"\n[INFO] Preparing generation "
        f"with {total} configurations...",
        flush=True,
    )

    stats_start = time.perf_counter()

    bm_stats = BM25(chunks, 0, 0, 0)
    bm_stats.compute_stats()

    avg_len = bm_stats.avg_len
    global_df = bm_stats.global_df
    inverted_index = bm_stats.inverted_index

    stats_time = time.perf_counter() - stats_start

    print(
        f"[INFO] BM25 statistics computed "
        f"in {stats_time:.2f}s",
        flush=True,
    )

    workers = min(
        total,
        os.cpu_count() or 1,
    )

    print(
        f"[INFO] Starting {workers} worker processes...",
        flush=True,
    )

    start = time.perf_counter()

    results = []

    with ProcessPoolExecutor(
        max_workers=workers,
    ) as executor:

        futures = {
            executor.submit(
                compete_worker,
                index,
                total,
                params,
                chunks,
                avg_len,
                global_df,
                inverted_index,
                dataset,
            ): index
            for index, params in enumerate(
                generation,
                start=1,
            )
        }

        completed = 0

        for future in as_completed(futures):
            completed += 1

            try:
                result = future.result()
                results.append(result)

                print(
                    f"[PROGRESS] "
                    f"{completed}/{total} configurations "
                    f"completed",
                    flush=True,
                )

            except Exception as e:
                index = futures[future]

                print(
                    f"[ERROR] Configuration "
                    f"{index}/{total} failed: {e}",
                    flush=True,
                )

    elapsed = time.perf_counter() - start

    print(
        f"[INFO] Generation completed "
        f"in {elapsed:.2f}s",
        flush=True,
    )

    return results


def create_generation(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    best = max(
        results,
        key=lambda x: x.get("top5_rate", 0),
    )

    print(
        f"\n[EVOLUTION] Best parameters:"
        f"\n    k = {best['k']:.4f}"
        f"\n    b = {best['b']:.4f}"
        f"\n    l = {best['l']:.4f}"
        f"\n    top5 = {best['top5_rate']:.2f}%",
        flush=True,
    )

    generation: list[dict[str, Any]] = []

    for _ in range(5):
        generation.append({
            param: random.uniform(
                min_value,
                max_value,
            )
            for param, (
                min_value,
                max_value,
            ) in PARAM_RANGES.items()
        })

    for _ in range(4):
        params = {}

        for param, (
            min_value,
            max_value,
        ) in PARAM_RANGES.items():

            value = best[param]

            range_size = (
                max_value - min_value
            )

            value += random.uniform(
                -range_size * 0.2,
                range_size * 0.2,
            )

            value = max(
                min_value,
                min(max_value, value),
            )

            params[param] = value

        generation.append(params)

    generation.append({
        param: best[param]
        for param in PARAM_RANGES
    })

    return generation


if __name__ == "__main__":

    print("=" * 60)
    print("BM25 PARAMETER OPTIMIZER")
    print("=" * 60)

    print("\n[INFO] Loading chunks...", flush=True)

    selector = ChunkerSelector()

    chunks: list[Chunk] = selector.folder_chunking(
        "./data/raw/vllm-0.10.1"
    )

    print(
        f"[INFO] Loaded {len(chunks)} chunks",
        flush=True,
    )

    print(
        "[INFO] Loading dataset...",
        flush=True,
    )

    with open(
        "./data/public/AnsweredQuestions/dataset_docs_public.json",
        "r",
    ) as f:
        dataset = json.load(f)

    print(
        f"[INFO] Loaded "
        f"{len(dataset['rag_questions'])} questions",
        flush=True,
    )

    results_file = "./learning/results_docs.json"

    if os.path.exists(results_file):

        print(
            f"\n[INFO] Found previous results: "
            f"{results_file}",
            flush=True,
        )

        with open(
            results_file,
            "r",
        ) as f:
            previous_results = json.load(f)

        print(
            "[INFO] Creating next generation "
            "from previous results...",
            flush=True,
        )

        generation = create_generation(
            previous_results
        )

        generation_number = 1

    else:

        print(
            "\n[INFO] No previous results found.",
            flush=True,
        )

        print(
            "[INFO] Creating first random generation...",
            flush=True,
        )

        generation = [
            {
                param: random.uniform(
                    min_value,
                    max_value,
                )
                for param, (
                    min_value,
                    max_value,
                ) in PARAM_RANGES.items()
            }
            for _ in range(10)
        ]

        generation_number = 0

    while True:

        print("\n")
        print("=" * 60)
        print(
            f"GENERATION {generation_number}"
        )
        print("=" * 60)

        generation_start = time.perf_counter()

        results = compete_generation(
            generation=generation,
            chunks=chunks,
            dataset=dataset,
            selector=selector,
        )

        results.sort(
            key=lambda x: x["top5_rate"],
            reverse=True,
        )

        print("\n[RESULTS]")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"#{rank:2d} "
                f"k={result['k']:.4f} "
                f"b={result['b']:.4f} "
                f"l={result['l']:.4f} "
                f"top5={result['top5_rate']:.2f}% "
                f"top3={result['top3_rate']:.2f}% "
                f"top10={result['top10_rate']:.2f}%"
            )

        best = results[0]

        print("\n" + "-" * 60)

        print(
            f"BEST OF GENERATION {generation_number}"
        )

        print(
            f"k      = {best['k']:.6f}"
        )

        print(
            f"b      = {best['b']:.6f}"
        )

        print(
            f"l      = {best['l']:.6f}"
        )

        print(
            f"top5   = {best['top5_rate']:.2f}%"
        )

        print(
            f"top3   = {best['top3_rate']:.2f}%"
        )

        print(
            f"top10  = {best['top10_rate']:.2f}%"
        )

        print(
            f"avg rank = {best['avg_ranking']:.2f}"
        )

        print("-" * 60)

        print(
            f"[INFO] Saving results to {results_file}...",
            flush=True,
        )

        os.makedirs(
            os.path.dirname(results_file),
            exist_ok=True,
        )

        with open(
            results_file,
            "w",
        ) as f:
            json.dump(
                results,
                f,
                indent=4,
            )

        print(
            "[INFO] Results saved.",
            flush=True,
        )

        generation = create_generation(
            results
        )

        generation_number += 1

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        print(
            f"\n[INFO] Generation finished "
            f"in {generation_time:.2f}s",
            flush=True,
        )

        print(
            "[INFO] Starting next generation...\n",
            flush=True,
        )