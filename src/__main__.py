from chunking.chunker_selector import ChunkerSelector
from indexing.bm25 import BM25
if __name__ == "__main__":
    selector = ChunkerSelector()
    chunks = selector.folder_chunking("./data/raw/vllm-0.10.1")

    bm25 = BM25(chunks)
    bm25.compute_stats()
    print(bm25.global_df)
    # chunks = selector.split_text("./data/raw/vllm-0.10.1/tests/basic_correctness/test_basic_correctness.py")

    # for c in chunks:
    #     print(c.start_index)
    #     print(c.file_path)
    #     print(c.tokenized_content)
    #     print(c.df)

    # print(len(chunks))

    # print("\n".join(text))