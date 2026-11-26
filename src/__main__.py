import os

import fire

from src.cli.cli import Cli
from src.indexing.embedder import Embedder
from src.utils.logger import Logger, LogLevel

if __name__ == "__main__":

    os.environ["DEBUG"] = "1"

    embedder = Embedder.from_file("data/processed/chunks.json")

    embedder.vectorize_chunks()

    try:
        fire.Fire(Cli(), serialize=lambda _: None)
    except Exception as e:
        Logger.log(f"Fatal error: {e}", LogLevel.FATAL)
