"""Public packaging of Calliope."""

from calliope import examples, exceptions
from calliope._version import __version__
from calliope.attrdict import AttrDict
from calliope.model import Model, read_netcdf
from calliope.util.logging import set_log_verbosity

__title__ = "Calliope"
__author__ = "Calliope contributors listed in AUTHORS"
__copyright__ = "Copyright (C) since 2013 Calliope contributors listed in AUTHORS"
