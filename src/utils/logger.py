from collections.abc import Iterable
from enum import Enum
from os import environ
from typing import Any, cast

from tqdm import tqdm

from src.utils.terminal import Colors, TerminalStyler


class LogLevel(Enum):
    DEBUG = [[Colors.BLUE, Colors.BOLD], "[DEBUG]"]
    INFO = [[Colors.GREEN, Colors.BOLD], "[INFO]"]
    WARN = [[Colors.YELLOW, Colors.BOLD], "[WARN]"]
    ERROR = [[Colors.RED, Colors.BOLD], "[ERROR]"]
    FATAL = [[Colors.MAGENTA, Colors.BOLD], "[FATAL]"]


class _ProgressBar(tqdm):

    def close(self) -> None:
        if not self.disable and self.leave is not False:
            self.bar_format = "{desc}: done in {elapsed}"
        super().close()


class Logger:
    @staticmethod
    def _label(log_level: LogLevel) -> str:
        colors = cast(list[Colors], log_level.value[0])
        label = cast(str, log_level.value[1])
        return TerminalStyler.colored_text(colors, label)

    @staticmethod
    def _debug_enabled() -> bool:
        return environ.get("DEBUG") == "1"

    @staticmethod
    def log(message: object, log_level: LogLevel = LogLevel.INFO) -> None:
        if log_level is LogLevel.DEBUG and not Logger._debug_enabled():
            return
        tqdm.write(f"{Logger._label(log_level)} {message}")

    @staticmethod
    def progress[T](
                iterable: Iterable[T],
                desc: str = "",
                **kwargs: Any
            ) -> Iterable[T]:
        return Logger._bar(iterable, desc=desc, **kwargs)

    @staticmethod
    def _bar(*args: Any, desc: str | None = "", **kwargs: Any) -> tqdm:
        kwargs.pop("disable", None)
        return _ProgressBar(
            *args,
            desc=f"{Logger._label(LogLevel.DEBUG)} {desc or ''}",
            disable=not Logger._debug_enabled(),
            **kwargs
        )

    @staticmethod
    def hook_transformers() -> None:
        from transformers.utils import logging as hf_logging

        hf_logging.set_tqdm_hook(lambda _factory, args, kwargs: Logger._bar(*args, **kwargs))
