__title__ = 'Calliope'
__author__ = 'Stefan Pfenninger'
__copyright__ = 'Copyright 2013-2015 Stefan Pfenninger'

from ._version import __version__

from .core import Model
from .parallel import Parallelizer
from .time_tools import TimeSummarizer
from . import utils
from . import read
from . import time_functions
from . import analysis
