import fire

from src.cli.cli import Cli
from src.utils.logger import Logger, LogLevel

if __name__ == "__main__":
    try:
        fire.Fire(Cli(), serialize=lambda _: None)
    except Exception as e:
        Logger.log(f"Fatal error: {e}", LogLevel.FATAL)
