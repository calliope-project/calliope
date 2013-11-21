__title__ = 'Calliope -- a multi-scale energy systems (MUSES) model'
__version__ = '0.1.0'
__author__ = 'Stefan Pfenninger'
__copyright__ = 'Copyright 2013 Stefan Pfenninger'

from .core import Model
from .parallel import Parallelizer
from .time import TimeSummarizer
from . import utils
from . import parallel_tools