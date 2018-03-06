import warnings


from calliope._version import __version__
from calliope.core import AttrDict, Model, read_netcdf
from calliope import examples, exceptions


__title__ = 'Calliope'
__author__ = 'Calliope contributors listed in AUTHORS'
__copyright__ = 'Copyright 2013-2017 Calliope contributors listed in AUTHORS'


_time_format = '%Y-%m-%d %H:%M:%S'


# Enable simple format when printing ModelWarnings
formatwarning_orig = warnings.formatwarning


def _formatwarning(message, category, filename, lineno, line=None):
    """Formats ModelWarnings as "Warning: message" without extra crud"""
    if category == exceptions.ModelWarning:
        return 'Warning: ' + str(message) + '\n'
    else:
        return formatwarning_orig(message, category, filename, lineno, line)


warnings.formatwarning = _formatwarning
