from src.utils.terminal import Colors, TerminalStyler

from enum import Enum


class LogLevel(Enum):
    DEBUG = [[Colors.GREEN, Colors.BOLD], "[DEBUG]"]
    INFO = [[Colors.CYAN, Colors.BOLD], "[INFO]"]
    WARN = [[Colors.YELLOW, Colors.BOLD], "[WARN]"]
    ERROR = [[Colors.RED, Colors.BOLD], "[ERROR]"]
    FATAL = [[Colors.MAGENTA, Colors.BOLD], "[FATAL]"]


class Logger():
    @staticmethod
    def log(message: str, log_level: LogLevel = LogLevel.INFO) -> None:
        print(TerminalStyler.colored_text(*(log_level.value)), message)