from __future__ import print_function
from __future__ import division

import math


class Technology(object):
    """Base Technology

    If parameters need to be calculated from configuration settings, this
    can be done in the __init__() function of a Technology subclass.

    Technology subclasses must be named like so for core.py to find them:
        `<Techname>Technology`.
    `<Techname>` must always have the first letter capitalized, but no other
    letters, even if it is an acronym (e.g., CspTechnology).

    """
    def __init__(self, model=None, name=None):
        """
        Initialization: perform any calculations that can be done without
        having read in the full (time series) data. By default, this does
        nothing apart from attaching the name property to the Technology
        object.

        Args:
            o : (AttrDict) model settings -- not used for anything
            name : (string) name for the technology

        """
        super(Technology, self).__init__()
        self.name = name

    def _setup(self):
        """Setup: perform any calculations that need the full (time
        series) data already read in. By default, this does nothing."""
        pass

    def __repr__(self):
        if self.name:
            return 'Generic technology ({})'.format(self.name)
        else:
            return 'Generic technology'


class CspTechnology(Technology):
    """Concentrating solar power (CSP)"""
    def __init__(self, model):
        """
        Based on settings in `model`, calculates the maximum storage time if
        needed.

        Args:
            o : (AttrDict) model settings

        """
        super(CspTechnology, self).__init__()
        # If necessary, set storage capacity based on s_time
        # and reference efficiency
        if model.get_option('csp.constraints.use_s_time'):
            s_time = model.get_option('csp.constraints.s_time')
            e_cap_max = model.get_option('csp.constraints.e_cap_max')
            e_eff_ref = model.get_eff_ref('e', 'csp')
            model.set_option('csp.constraints.s_cap_max',
                             s_time * e_cap_max / e_eff_ref)

    def __repr__(self):
        return 'Concentrating solar power (CSP)'
