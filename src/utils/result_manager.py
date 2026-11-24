import json
from pathlib import Path
from typing import Any

from src.errors.rag_error import RagError
from src.utils.logger import Logger, LogLevel
from src.utils.models import RagEncoder


class ResultManager:
    @staticmethod
    def store_result(output_path: str, result: Any) -> None:
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if not path.exists():
                path.touch()
                Logger.log(f"Path: {output_path} initialized.", LogLevel.DEBUG)

            with open(output_path, "w") as f:
                json.dump(
                        result,
                        f,
                        indent="\t",
                        ensure_ascii=False,
                        cls=RagEncoder
                    )
                f.write("\n")
        except Exception as e:
            Logger.log(e, LogLevel.ERROR)
            raise RagError("Stopped")

    @staticmethod
    def read_result(input_path: str) -> Any:
        try:
            Logger.log(f"Reading {input_path}.", LogLevel.DEBUG)
            with open(input_path, "r") as f:
                return json.load(f)
        except Exception as e:
            Logger.log(e, LogLevel.ERROR)
            raise RagError("Stopped")
