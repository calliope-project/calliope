"""Public packaging of Calliope."""

import importlib

from calliope import examples
from calliope._version import __version__
from calliope.attrdict import AttrDict
from calliope.model import Model, read_dict, read_netcdf, read_yaml
from calliope.util.logging import set_log_verbosity

try:
    from rich import pretty

    pretty.install(max_depth=1)
except ModuleNotFoundError:
    pass

__title__ = "Calliope"
__author__ = "Calliope contributors listed in AUTHORS"
__copyright__ = "Copyright (C) since 2013 Calliope contributors listed in AUTHORS"
