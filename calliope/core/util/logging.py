
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

SOLVER = 19
logging.addLevelName(SOLVER, 'SOLVER')
logger = logging.getLogger('calliope')
logger.propagate = False
logger.setLevel(logging.WARNING)


def set_handler(output_format, time_format='%Y-%m-%d %H:%M:%S'):
    """
    Set the logging handler (format and stream) based on whether we're working in
    Python interactively or through the command line interface
    """

    if output_format == 'cli':
        logger.setLevel(logging.WARNING)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s %(message)s', datefmt=time_format
        )
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(formatter)
        logger.addHandler(console)

    elif output_format == 'python' and not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return None


def set_log_level(level):
    """
    Set the minimum logging verbosity in a Python console. Higher verbosity levels
    will include their output and all those of following levels.
    Level options (in descending order of verbosity):

    * 'DEBUG'
    * 'SOLVER' -> Calliope custom level, assigned value of 19
    * 'INFO' -> default level
    * 'WARNING'
    * 'ERROR'
    * 'CRITICAL'
    """

    # FIXME: Propagate logging to backend solvers successfully
    def _set_backend_level(level):
        BACKENDS = ['pyomo']
        for backend in BACKENDS:
            logging.getLogger(backend).setLevel(level)

    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
        #_set_backend_level(logging.INFO)

    elif level == 'SOLVER':
        logger.setLevel(SOLVER)
        #_set_backend_level(logging.INFO)

    else:
        logger.setLevel(getattr(logging, level))


def log_time(timings, identifier, comment=None, level='info', time_since_start=False):
    if comment is None:
        comment = identifier

    timings[identifier] = now = datetime.datetime.now()

    getattr(logger, level)('[{}] {}'.format(now, comment))
    if time_since_start:
        time_diff = now - timings['model_creation']
        getattr(logger, level)('[{}] Time since start: {}'.format(now, time_diff))


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
