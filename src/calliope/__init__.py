"""Public packaging of Calliope."""

import xarray as xr
from rich import pretty

from calliope import examples, exceptions
from calliope._version import __version__
from calliope.attrdict import AttrDict
from calliope.model import Model, read_netcdf, read_yaml
from calliope.util.logging import set_log_verbosity

xr.set_options(display_expand_attrs=False)
pretty.install(max_depth=1)

__title__ = "Calliope"
__author__ = "Calliope contributors listed in AUTHORS"
__copyright__ = "Copyright (C) since 2013 Calliope contributors listed in AUTHORS"
