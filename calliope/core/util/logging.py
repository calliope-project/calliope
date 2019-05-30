
"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

logger.py
~~~~~~~~~~

Create the Calliope logger object and apply other logging tools/functionality

"""

import datetime
import logging
import sys


def set_log_verbosity(verbosity='info', include_solver_output=True, logger=None):
    """
    Set the verbosity of logging.

    Parameters
    ----------
    verbosity : str, default 'info'
        Logging level to display across all of Calliope. Can be one of
        'debug', 'info', 'warning', 'error', or 'critical'.
    include_solver_output : bool, default True
        If True, the logging level for just the backend model is set to
        DEBUG, which turns on display of solver output.
    logger : logging.Logger, optional
        For most cases this can be ignored. If not given, the root logger
        is used, and a StreamHandler to log to sys.stdout is attached to it.

    """
    if include_solver_output:
        backend_logger = logging.getLogger('calliope.backend.pyomo.model')
        backend_logger.setLevel(logging.DEBUG)
    if logger is None:
        logger = logging.getLogger()  # Root logger
        logger.propagate = False
        console = logging.StreamHandler(stream=sys.stdout)
        logger.addHandler(console)
    logger.setLevel(verbosity.upper())


def log_time(logger, timings, identifier, comment=None, level='info', time_since_start=False):
    if comment is None:
        comment = identifier

    timings[identifier] = now = datetime.datetime.now()

    if time_since_start:
        time_diff = now - timings['model_creation']
        comment += '. Time since start: {}'.format(time_diff)

    getattr(logger, level)(comment)


class LogWriter:
    def __init__(self, logger, level, strip=False):
        self.logger = logger
        self.level = level
        self.strip = strip

    def write(self, message):
        if message != '\n':
            if self.strip:
                message = message.strip()
            getattr(self.logger, self.level)(message)

    def flush(self):
        pass
