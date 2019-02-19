"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

modelwide.py
~~~~~~~~~

Model-wide constraints.

"""

import numpy as np
import pyomo.core as po  # pylint: disable=import-error


def load_constraints(backend_model):
    model_data_dict = backend_model.__calliope_model_data__['data']

    if 'modelwide_demand_share_min' in model_data_dict:
        backend_model.modelwide_demand_share_min_constraint = po.Constraint(
            backend_model.modelwide_constraint_groups,
            backend_model.carriers,
            ['min'], rule=demand_share_constraint_rule
        )
    if 'modelwide_demand_share_max' in model_data_dict:
        backend_model.modelwide_demand_share_max_constraint = po.Constraint(
            backend_model.modelwide_constraint_groups,
            backend_model.carriers,
            ['max'], rule=demand_share_constraint_rule
        )


def equalizer(lhs, rhs, sign):
    if sign == 'max':
        return lhs <= rhs
    elif sign == 'min':
        return lhs >= rhs
    elif sign == 'equals':
        return lhs == rhs
    else:
        raise ValueError('Invalid sign: {}'.format(sign))


def demand_share_constraint_rule(backend_model, modelwide_constraint_group, carrier, what):
    """
    TODO write docstring
    """
    model_data_dict = backend_model.__calliope_model_data__['data']
    share = model_data_dict['modelwide_demand_share_{}'.format(what)][(carrier, modelwide_constraint_group)]
    # FIXME uncomment this once Bryn has merged his changes
    # and import again: from calliope.backend.pyomo.util import get_param
    # share = get_param(
    #     backend_model,
    #     'modelwide_demand_share_{}'.format(what), (carrier, modelwide_constraint_group)
    # )

    if np.isnan(share):
        return po.Constraint.NoConstraint
    else:
        lhs_loc_techs = getattr(
            backend_model,
            'modelwide_constraint_loc_techs_{}'.format(modelwide_constraint_group)
        )
        lhs_locs = [loc_tech.split('::')[0] for loc_tech in lhs_loc_techs]
        rhs_loc_techs = [
            i for i in backend_model.loc_techs_demand
            if i.split('::')[0] in lhs_locs
        ]

        lhs = sum(
            backend_model.carrier_prod[loc_tech + '::' + carrier, timestep]
            for loc_tech in lhs_loc_techs
            for timestep in backend_model.timesteps
        )
        rhs = share * -1 * sum(
            backend_model.carrier_con[loc_tech + '::' + carrier, timestep]
            for loc_tech in rhs_loc_techs
            for timestep in backend_model.timesteps
        )

        return equalizer(lhs, rhs, what)
