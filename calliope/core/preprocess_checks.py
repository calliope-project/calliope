"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_checks.py
~~~~~~~~~~~~~~~~~~~~

Checks for model consistency and possible errors during preprocessing.

"""

import os

from .. import utils
from .. import exceptions

def check_initial(config_model, config_run):
    # At this stage, it's probably more useful to have all error messages at once
    err_message = []

    # Only ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3']
    # are allowed as carrier tiers
    for key in config_model.as_dict_flat().keys():
        if ('.carrier_' in key and key.split('.carrier_')[-1].split('.')[0] not
            in ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3', 'ratios']):
            err_message.append("Invalid carrier tier found at {}. Only "
            "'carrier_' + ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3'] "
            "is valid.".format(key))

    # Print all the information in err_message, if there is any
    if err_message:
        raise exceptions.ModelError("\n".join(err_message))

    pass


def check_final(model_run):
    comments = utils.AttrDict()
    return comments
