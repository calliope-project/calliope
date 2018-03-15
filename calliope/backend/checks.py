"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

run_checks.py
~~~~~~~~~~~~~

Checks for model consistency and possible errors when preparing run in the backend.

"""
import ruamel.yaml

import numpy as np
from calliope.core.attrdict import AttrDict


def check_operate_params(model_data):
    """
    if model mode = `operate`, check for clashes in capacity constraints.
    In this mode, all capacity constraints are set to parameters in the backend,
    so can easily lead to model infeasibility if not checked.

    Returns
    -------
    comments : AttrDict
        debug output
    warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    defaults = ruamel.yaml.load(model_data.attrs['defaults'], Loader=ruamel.yaml.Loader)
    warnings, errors = [], []
    comments = AttrDict()

    def get_param(loc_tech, var):
        if is_in(loc_tech, var):
            param = model_data[var].loc[loc_tech].item()
        else:
            param = defaults[var]
        return param

    def is_in(loc_tech, set_or_var):
        if set_or_var in model_data:
            try:
                model_data[set_or_var].loc[loc_tech]
                return True
            except KeyError:
                return False
        else:
            return False

    for loc_tech in model_data.loc_techs.values:
        energy_cap = model_data.energy_cap.loc[loc_tech].item()
        # Must have energy_cap defined for all relevant techs in the model
        if (np.isnan(energy_cap) or np.isinf(energy_cap)) and not is_in(loc_tech, 'force_resource'):
            errors.append(
                'Operate mode: User must define a finite energy_cap (via '
                'energy_cap_equals or energy_cap_max) for {}'.format(loc_tech)
            )
        # Cannot have infinite resource area (physically impossible)
        elif is_in(loc_tech, 'loc_techs_finite_resource'):
            if is_in(loc_tech, 'loc_techs_area'):
                area = model_data.resource_area.loc[loc_tech].item()
                if np.isnan(area) or np.isinf(area):
                    errors.append(
                        'Operate mode: User must define a finite resource_area '
                        '(via resource_area_equals or resource_area_max) for {}, '
                        'as a finite available resource is considered'.format(loc_tech)
                    )
            # Cannot have consumed resource being higher than energy_cap, as
            # constraints will clash. Doesn't affect supply_plus techs with a
            # storage buffer prior to carrier production.
            elif not is_in(loc_tech, 'loc_techs_store'):
                area = 1
                resource_scale = get_param(loc_tech, 'resource_scale')
                energy_cap_scale = get_param(loc_tech, 'energy_cap_scale')
                resource_eff = get_param(loc_tech, 'resource_eff')
                energy_eff = get_param(loc_tech, 'energy_eff')
                energy_cap_scale = get_param(loc_tech, 'energy_cap_scale')
                resource = model_data.resource.loc[loc_tech] * area
                if any(resource.values * resource_scale * resource_eff
                    > energy_cap * energy_cap_scale * energy_eff):
                    errors.append(
                        'Operate mode: resource is forced to be higher than '
                        'fixed energy cap for `{}`'.format(loc_tech)
                    )
            if is_in(loc_tech, 'force_resource'):
                model_data.energy_cap.loc[loc_tech] = np.inf
                warnings.append(
                    'Energy capacity constraint removed from {} as force_resource '
                    'is applied'.format(loc_tech)
                )
                if is_in(loc_tech, 'resource_cap'):
                    model_data.resource_cap.loc[loc_tech] = np.inf
                    warnings.append(
                        'Resource capacity constraint removed from {} as force_resource '
                        'is applied'.format(loc_tech)
                    )
        # FIXME: remove?
        # Cannot have storage capacity be less than energy capacity * charge rate
        if is_in(loc_tech, 'loc_techs_store'):
            if is_in(loc_tech, 'charge_rate'):
                storage_cap = model_data.storage_cap.loc[loc_tech].item()
                energy_cap_scale = get_param(loc_tech, 'energy_cap_scale')
                if storage_cap and energy_cap:
                    charge_rate = model_data['charge_rate'].loc[loc_tech]
                    if storage_cap * charge_rate < energy_cap * energy_cap_scale:
                        errors.append(
                            'fixed storage capacity * charge rate is not larger '
                            'than fixed energy capacity for loc::tech {}'.format(loc_tech)
                        )

    window = model_data.attrs['run.operation.window']
    horizon = model_data.attrs['run.operation.horizon']
    if not window or not horizon:
        errors.append(
            'Operational mode requires a timestep window and horizon to be '
            'defined under run.operation'
        )
    elif horizon < window:
        errors.append(
            'Iteration horizon must be larger than iteration window, '
            'for operational mode'
        )

    return comments, warnings, errors
