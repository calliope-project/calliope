
"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

logger.py
~~~~~~~~~~

Create the Calliope logger object and apply other logging tools/functionality

"""

import logging
import sys
import datetime
import pyomo

SOLVER = 19
logging.addLevelName(SOLVER, 'SOLVER')
logger = logging.getLogger('calliope')
logger.propagate = False

if logger.hasHandlers():
    for handler in logger.handlers:
        logger.removeHandler(handler)

formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
console = logging.StreamHandler(stream=sys.stdout)
console.setFormatter(formatter)
logger.addHandler(console)


def set_log_level(level):
    """
    Set the minimum logging verbosity in a Python console. Higher verbosity levels
    will include their output and all those of following levels.
    Level options (in descending order of verbosity):

    * 'DEBUG'
    * 'SOLVER' -> Calliope custom level, assigned value of 19,
                  returns solver (e.g. GLPK) stream
    * 'INFO' -> default level
    * 'WARNING'
    * 'ERROR'
    * 'CRITICAL'
    """

    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)

    elif level == 'SOLVER':
        logger.setLevel(SOLVER)

    else:
        logger.setLevel(getattr(logging, level))


def log_time(timings, identifier, comment=None, level='info', time_since_start=False):
    if comment is None:
        comment = identifier

    timings[identifier] = now = datetime.datetime.now()

    if time_since_start:
        time_diff = now - timings['model_creation']
        comment += '. Time since start: {}'.format(time_diff)

    getattr(logger, level)(comment)


class LogWriter:
    def __init__(self, level, strip=False):
        self.level = level
        self.strip = strip

    def write(self, message):
        if message != '\n':
            if self.strip:
                message = message.strip()
            if self.level == 'solver':
                logger.log(SOLVER, message)
            else:
                getattr(logger, self.level)(message)

    def flush(self):
        pass
