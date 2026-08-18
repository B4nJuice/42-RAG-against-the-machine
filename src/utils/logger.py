from src.utils.terminal import Colors, TerminalStyler

from os import environ
from enum import Enum


class LogLevel(Enum):
    DEBUG = [[Colors.BLUE, Colors.BOLD], "[DEBUG]"]
    INFO = [[Colors.GREEN, Colors.BOLD], "[INFO]"]
    WARN = [[Colors.YELLOW, Colors.BOLD], "[WARN]"]
    ERROR = [[Colors.RED, Colors.BOLD], "[ERROR]"]
    FATAL = [[Colors.MAGENTA, Colors.BOLD], "[FATAL]"]


class Logger():
    @staticmethod
    def log(message: str, log_level: LogLevel = LogLevel.INFO) -> None:
        if log_level is LogLevel.DEBUG and environ.get("DEBUG") != "1":
            return
        print(TerminalStyler.colored_text(*(log_level.value)), message)