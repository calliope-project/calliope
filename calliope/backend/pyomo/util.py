"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope.core.util.tools import memoize


#FIXME: change to get_param
@memoize
def param_getter(backend_model, var, dims):
    try:
        return backend_model.__calliope_model_data__['data'][var][dims]
    except KeyError:  # Try without time dimension, which is always last
        try:
            return backend_model.__calliope_model_data__['data'][var][dims[:-1]]
        except KeyError:  # Static default value
            return backend_model.__calliope_defaults__[var]


def get_previous_timestep(backend_model, timestep):
    # order_dict starts numbering at zero, timesteps is one-indexed, so we do not need
    # to subtract 1 to get to previous_step -- it happens "automagically"
    return backend_model.timesteps[backend_model.timesteps.order_dict[timestep]]


@memoize
def get_loc_tech_carriers(backend_model, loc_carrier):

    lookup = backend_model.__calliope_model_data__['data']['lookup_loc_carriers']
    loc_tech_carriers = lookup[loc_carrier].split(',')

    loc_tech_carriers_prod = [
        i for i in loc_tech_carriers if i in backend_model.loc_tech_carriers_prod
    ]
    loc_tech_carriers_con = [
        i for i in loc_tech_carriers if i in backend_model.loc_tech_carriers_con
    ]

    if hasattr(backend_model, 'loc_tech_carriers_export'):
        loc_tech_carriers_export = [
            i for i in loc_tech_carriers
            if i in backend_model.loc_tech_carriers_export
        ]
    else:
        loc_tech_carriers_export = []

    return (
        loc_tech_carriers_prod,
        loc_tech_carriers_con,
        loc_tech_carriers_export
    )


@memoize
def get_loc_tech(loc_tech_carrier):
    return loc_tech_carrier.rsplit(':', 1)[0]


@memoize
def get_timestep_weight(backend_model):
    model_data_dict = backend_model.__calliope_model_data__
    time_res_sum = sum(model_data_dict['data']['timestep_resolution'].values())
    weights_sum = sum(model_data_dict['data']['timestep_weights'].values())
    return (time_res_sum * weights_sum) / 8760
