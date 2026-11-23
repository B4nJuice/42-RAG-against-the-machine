import fire

from src.cli.cli import Cli
from src.utils.logger import Logger, LogLevel

if __name__ == "__main__":
    try:
        fire.Fire(Cli())
    except BaseException as e:
        Logger.log(f"fatal error: {e}", LogLevel.FATAL)
