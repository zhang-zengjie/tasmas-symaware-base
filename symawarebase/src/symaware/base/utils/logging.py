import functools
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    # String type hinting to support python 3.9
    import sys
    from typing import TypeVar

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    Param = ParamSpec("Param")
    Return = TypeVar("Return")


def initialize_logger(level: "str | int" = logging.INFO):
    """
    Set the logging level for all loggers

    Args
    ----
    level:
        The logging level to set
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(file_name: str, class_name: "str | None" = None) -> logging.Logger:
    """
    Returns a logger with an appropriate name.
    If no class name is provided, the last part of the file name is capitalized and used instead.

    Example
    -------
    The logger utility can be used in any module.
    Is is recommended give the logger the name of the module and the class that uses it.

    >>> from symaware.base import get_logger
    >>> logger = get_logger(__name__, "MyClass")
    >>> logger.info("Hello world!")

    Args
    ----
    file_name:
        The name of the file where the logger is defined
    class_name:
        The name of the class that will use the logger

    Returns
    -------
        The logger
    """

    parts = file_name.split(".")
    package = ".".join(parts[:-1])
    name = f"{package}.{class_name or parts[-1].capitalize()}"
    return logging.getLogger(name)


def log(
    logger: logging.Logger = get_logger(__name__), level: int = logging.DEBUG
) -> "Callable[[Callable[Param, Return]], Callable[Param, Return]]":
    """
    Decorator to log the input and output of a function.

    Examples
    --------
    Using the decorator to log the input and output of a function

    >>> import logging
    >>> from symaware.base import log, get_logger, initialize_logger
    >>>
    >>> initialize_logger(logging.INFO)
    >>> logger = get_logger(__name__)
    >>>
    >>> @log(logger, logging.INFO)
    ... def my_function(x):
    ...     return x + 1
    >>>
    >>> # INFO:__main__:my_function(1, {})
    >>> # INFO:__main__:my_function(...) -> 2
    >>> my_function(1)
    2


    >>> import logging
    >>> from symaware.base import log, get_logger, initialize_logger
    >>>
    >>> initialize_logger(logging.DEBUG)
    >>> logger = get_logger(__name__)
    >>>
    >>> class MyClass:
    ...     __LOGGER = get_logger(__name__, "MyClass")
    ...     @log(__LOGGER)
    ...     def my_function(self, x):
    ...         return x + 2
    >>>
    >>> my_instance = MyClass()
    >>> # DEBUG:__main__.MyClass:my_function(4, {})
    >>> # DEBUG:__main__.MyClass:my_function(...) -> 6
    >>> my_instance.my_function(4)
    6

    Args
    ----
    logger:
        what logger to use. If not provided, a default logger from this module is used
    level:
        what logging level to use

    Returns
    -------
        Log function decorator
    """

    def _log(func: "Callable[Param, Return]") -> "Callable[Param, Return]":
        @functools.wraps(func)
        def decorator(*args: Any, **kwargs: Any) -> Any:
            logger.log(level, "%s(%s, %s)", func.__name__, args, kwargs)
            result = func(*args, **kwargs)
            logger.log(level, "%s(...) -> %s", func.__name__, result)
            return result

        return decorator

    return _log
