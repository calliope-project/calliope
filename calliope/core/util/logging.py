"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

logger.py
~~~~~~~~~~

Create the Calliope logger object and apply other logging tools/functionality

"""

import datetime
import logging
import sys

_time_format = "%Y-%m-%d %H:%M:%S"


def setup_root_logger(verbosity, capture_warnings):
    root_logger = logging.getLogger()  # Get the root logger

    # Remove any existing output handlers from root logger
    if root_logger.hasHandlers():
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Create a console log handler with decent formatting and attach it
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s", datefmt=_time_format
    )
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    root_logger.setLevel(verbosity.upper())

    if capture_warnings:
        logging.captureWarnings(True)
        pywarning_logger = logging.getLogger("py.warnings")
        pywarning_logger.setLevel(verbosity.upper())

    return root_logger


def set_log_verbosity(
    verbosity="info", include_solver_output=True, capture_warnings=True
):
    """
    Set the verbosity of logging and setup the root logger to log to
    console (stdout) with timestamp output formatting.

    Parameters
    ----------
    verbosity : str, default 'info'
        Logging level to display across all of Calliope. Can be one of
        'debug', 'info', 'warning', 'error', or 'critical'.
    include_solver_output : bool, default True
        If True, the logging level for just the backend model is set to
        DEBUG, which turns on display of solver output.
    capture_warnings : bool, default True
        If True, also capture all warnings and log them to the WARNING
        level. This results in more consistent output when running
        interactively.

    """
    backend_logger = logging.getLogger("calliope.backend.pyomo.model")
    if include_solver_output is True:
        backend_logger.setLevel("DEBUG")
    else:
        backend_logger.setLevel(verbosity.upper())

    setup_root_logger(verbosity=verbosity, capture_warnings=capture_warnings)


def log_time(
    logger, timings, identifier, comment=None, level="info", time_since_run_start=False
):
    if comment is None:
        comment = identifier

    timings[identifier] = now = datetime.datetime.now()

    if time_since_run_start and "run_start" in timings:
        time_diff = now - timings["run_start"]
        comment += ". Time since start of model run: {}".format(time_diff)

    getattr(logger, level)(comment)


class LogWriter:
    def __init__(self, logger, level, strip=False):
        self.logger = logger
        self.level = level
        self.strip = strip

    def write(self, message):
        if message != "\n":
            if self.strip:
                message = message.strip()
            getattr(self.logger, self.level)(message)

    def flush(self):
        pass
