"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope.core.util.tools import memoize


@memoize
def param_getter(backend_model, var, dims):
    try:
        return backend_model.__calliope_model_data__[var][dims]
    except KeyError:  # Try without time dimension, which is always last
        try:
            return backend_model.__calliope_model_data__[var][dims[:-1]]
        except KeyError:  # Static default value
            return backend_model.__calliope_defaults__[var]


def get_previous_timestep(backend_model, timestep):
    # order_dict starts numbering at zero, timesteps is one-indexed, so we do not need
    # to subtract 1 to get to previous_step -- it happens "automagically"
    return backend_model.timesteps[backend_model.timesteps.order_dict[timestep]]


@memoize
def get_loc_tech_carriers(backend_model, loc_carrier):
    sets = backend_model.__calliope_model_data__['sets']

    loc_tech_carriers = [
        i.split(',') for i in
        backend_model.__calliope_model_data__['data']['lookup_loc_carriers']
    ]

    loc_tech_carriers_prod = [
        i for i in loc_tech_carriers if i in sets['loc_tech_carriers_prod']
    ]
    loc_tech_carriers_con = [
        i for i in loc_tech_carriers if i in sets['loc_tech_carriers_con']
    ]
    loc_tech_carriers_export = [
        i for i in loc_tech_carriers if i in sets['loc_tech_carriers_export']
    ]

    return (
        loc_tech_carriers_prod,
        loc_tech_carriers_con,
        loc_tech_carriers_export
    )
