import atexit
from typing import Protocol, runtime_checkable

from streamlet.utils.typing import PyTree


@runtime_checkable
class Logger(Protocol):
    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None: ...
    def log_summary(self, data: PyTree, **kwargs) -> None: ...
    def log_artifact(self, state: PyTree, step: int, **kwargs) -> None: ...
    def finish(self) -> None: ...


class MultiLogger:
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers
        atexit.register(self.finish)

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        for logger in self.loggers:
            logger.log(data, steps, **kwargs)

    def log_summary(self, data: PyTree, **kwargs) -> None:
        for logger in self.loggers:
            logger.log_summary(data, **kwargs)

    def log_artifact(self, state: PyTree, step: int, **kwargs) -> None:
        for logger in self.loggers:
            log_artifact = getattr(logger, "log_artifact", None)
            if log_artifact is not None:
                log_artifact(state, step, **kwargs)

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()
