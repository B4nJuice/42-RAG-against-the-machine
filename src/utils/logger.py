from enum import Enum
from os import environ
from typing import cast

from src.utils.terminal import Colors, TerminalStyler


class LogLevel(Enum):
    DEBUG = [[Colors.BLUE, Colors.BOLD], "[DEBUG]"]
    INFO = [[Colors.GREEN, Colors.BOLD], "[INFO]"]
    WARN = [[Colors.YELLOW, Colors.BOLD], "[WARN]"]
    ERROR = [[Colors.RED, Colors.BOLD], "[ERROR]"]
    FATAL = [[Colors.MAGENTA, Colors.BOLD], "[FATAL]"]


class Logger:
    @staticmethod
    def log(message: object, log_level: LogLevel = LogLevel.INFO) -> None:
        if log_level is LogLevel.DEBUG and environ.get("DEBUG") != "1":
            return
        colors = cast(list[Colors], log_level.value[0])
        label = cast(str, log_level.value[1])
        print(TerminalStyler.colored_text(colors, label), str(message))
