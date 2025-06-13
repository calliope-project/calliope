# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""Create the Calliope logger object and apply other logging tools/functionality."""

import datetime
import logging
import sys

_time_format = "%Y-%m-%d %H:%M:%S"

_orig_py_warning_handlers = logging.getLogger("py.warnings").handlers


def setup_root_logger(
    verbosity: str | int, capture_warnings: bool = True
) -> logging.Logger:
    """Setup Calliope root logger.

    Here, we set the logger format, clear any existing "handlers",
    set the Calliope-wide logging level (i.e. verbosity), and optionally fold in python warnings into logging.

    Args:
        verbosity (str | int):
            Logging level to use in all Calliope loggers.
            Can be a string (e.g. `INFO`, `DEBUG`) or an integer (e.g. `20` is equivalent to `INFO`).
            See https://docs.python.org/3/library/logging.html#logging-levels for more information.
        capture_warnings (bool, optional):
            If True, capture Python warnings in the logger (at the `WARNING` level).
            This results in more consistent output when running in Python.
            Defaults to True.

    Returns:
        logging.Logger: Calliope root logger with all setup applied.
    """
    root_logger = logging.getLogger("calliope")  # Get the root logger

    # Remove any existing output handlers from root logger
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a console log handler with decent formatting and attach it
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s", datefmt=_time_format
    )
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    root_logger.setLevel(verbosity)

    py_warnings_logger = logging.getLogger("py.warnings")
    if capture_warnings:
        logging.captureWarnings(True)
        py_warnings_logger.handlers = _orig_py_warning_handlers + [console]
        py_warnings_logger.setLevel(verbosity)
    else:
        logging.captureWarnings(False)
        logging.getLogger("py.warnings").setLevel("WARNING")
        py_warnings_logger.handlers = _orig_py_warning_handlers

    return root_logger


def set_log_verbosity(
    verbosity: str | int,
    include_solver_output: bool = True,
    capture_warnings: bool = True,
):
    """Set the verbosity of logging and setup the root logger to log to console (stdout) with timestamp output formatting.

    Args:
        verbosity (str | int):
            Logging level to use in all Calliope loggers.
            Can be a string (e.g. `INFO`, `DEBUG`) or an integer (e.g. `20` is equivalent to `INFO`).
            See https://docs.python.org/3/library/logging.html#logging-levels for more information.
        include_solver_output (bool, optional):
            If True, the logging level for just the backend model is set to
            DEBUG, which turns on display of solver output.
            Defaults to True.
        capture_warnings (bool, optional):
            If True, capture Python warnings in the logger (at the `WARNING` level).
            This results in more consistent output when running in Python.
            Defaults to True.
    """
    if isinstance(verbosity, str):
        verbosity = verbosity.upper()
    backend_logger = logging.getLogger("calliope.backend.backend_model.<solve>")
    if include_solver_output is True:
        backend_logger.setLevel("DEBUG")
    else:
        backend_logger.setLevel(verbosity)

    setup_root_logger(verbosity=verbosity, capture_warnings=capture_warnings)


def log_time(
    logger: logging.Logger,
    timings: dict,
    identifier: str,
    comment: str | None = None,
    level: str = "info",
    time_since_solve_start: bool = False,
) -> float:
    """Simultaneously log the time of a Calliope event to dictionary and to the logger.

    Args:
        logger (logging.Logger): Logger to use for logging the time.
        timings (dict): Dictionary of model timings.
        identifier (str): Short description to use as the event key in `timings`.
        comment (str | None, optional):
            Long description of the event.
            If not given, `identifier` will be used.
            Defaults to None.
        level (str, optional): Level at which to log the event with the `logger`. Defaults to "info".
        time_since_solve_start (bool, optional):
            If True, append comment in log message on the event's time compared to the time since the model was sent to the solver (in seconds).
            Defaults to False.

    Returns:
        timestamp (float): POSIX timestamp of the logged event
    """
    if comment is None:
        comment = identifier

    timings[identifier] = now = datetime.datetime.now().timestamp()

    if time_since_solve_start and "solve_start" in timings:
        time_diff = datetime.timedelta(seconds=now - timings["solve_start"])
        comment += f". Time since start of solving optimisation problem: {time_diff}"

    getattr(logger, level.lower())(comment)
    return now


class LogWriter:
    """Log writing helper class."""

    def __init__(self, logger, level, strip=False):
        """Custom logger to redirect solver outputs to avoid message duplication."""
        self.logger = logger
        self.level = level
        self.strip = strip

    def write(self, message):
        """Save a message to the logger."""
        if message != "\n":
            if self.strip:
                message = message.strip()
            getattr(self.logger, self.level)(message)

    def flush(self):
        """Placeholder for future flush functionality."""
        pass
