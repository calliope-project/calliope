"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

constraint_sets.py
~~~~~~~~~~~~~~~~~~

loc_techs, loc_carriers, and loc_tech_carriers subsets used per constraint, to
reduce constraint complexity

"""

from calliope.core.preprocess.util import constraint_exists

import numpy as np


def generate_constraint_sets(model_run):
    """
    Generate loc-tech sets for a given pre-processed ``model_run``

    Parameters
    ----------
    model_run : AttrDict
    """

    sets = model_run.sets
    ## From here on, everything is a `key=value` pair within a dictionary

    constraint_sets = dict()
    # energy_balance.py
    constraint_sets['loc_carriers_system_balance_constraint'] = sets.loc_carriers
    constraint_sets['loc_techs_balance_supply_constraint'] = sets.loc_techs_finite_resource_supply
    constraint_sets['loc_techs_balance_demand_constraint'] = sets.loc_techs_finite_resource_demand
    constraint_sets['loc_techs_resource_availability_supply_plus_constraint'] = sets.loc_techs_finite_resource_supply_plus
    constraint_sets['loc_techs_balance_transmission_constraint'] = sets.loc_techs_transmission
    constraint_sets['loc_techs_balance_supply_plus_constraint'] = sets.loc_techs_supply_plus
    constraint_sets['loc_techs_balance_storage_constraint'] = sets.loc_techs_storage
    if model_run.run.cyclic_storage is True:
        constraint_sets['loc_techs_storage_initial_constraint'] = [
            i for i in sets.loc_techs_store
            if constraint_exists(model_run, i, 'constraints.storage_initial') is not None
        ]
    constraint_sets['carriers_reserve_margin_constraint'] = [
        i for i in sets.carriers
        if i in model_run.model.get_key('reserve_margin', {}).keys()
    ]
    # clustering-specific balance constraints
    if (model_run.model.get_key('time.function', None) == 'apply_clustering' and
            model_run.model.get_key('time.function_options.storage_inter_cluster', True)):
        set_name = 'loc_techs_balance_storage_inter_cluster_constraint'
        constraint_sets[set_name] = sets.loc_techs_store

    # costs.py
    constraint_sets['loc_techs_cost_constraint'] = sets.loc_techs_cost
    constraint_sets['loc_techs_cost_investment_constraint'] = sets.loc_techs_investment_cost
    constraint_sets['loc_techs_cost_var_constraint'] = [
        i for i in sets.loc_techs_om_cost
        if i not in sets.loc_techs_conversion_plus + sets.loc_techs_conversion
    ]

    # export.py
    constraint_sets['loc_carriers_update_system_balance_constraint'] = [
        i for i in sets.loc_carriers if sets.loc_techs_export
        and any(['{0}::{2}'.format(*j.split('::')) == i
                for j in sets.loc_tech_carriers_export])
    ]
    constraint_sets['loc_tech_carriers_export_balance_constraint'] = (
        sets.loc_tech_carriers_export
    )
    constraint_sets['loc_techs_update_costs_var_constraint'] = [
        i for i in sets.loc_techs_om_cost if i in sets.loc_techs_export
    ]
    constraint_sets['loc_tech_carriers_export_max_constraint'] = [
        i for i in sets.loc_tech_carriers_export
        if constraint_exists(
            model_run, i.rsplit('::', 1)[0], 'constraints.export_cap'
        ) is not None
    ]

    # capacity.py
    constraint_sets['loc_techs_storage_capacity_constraint'] = [
        i for i in sets.loc_techs_store if i not in sets.loc_techs_milp
    ]
    constraint_sets['loc_techs_energy_capacity_storage_constraint'] = [
        i for i in sets.loc_techs_store
        if constraint_exists(model_run, i, 'constraints.charge_rate')
    ]
    constraint_sets['loc_techs_resource_capacity_constraint'] = [
        i for i in sets.loc_techs_finite_resource_supply_plus
        if any([
            constraint_exists(model_run, i, 'constraints.resource_cap_equals'),
            constraint_exists(model_run, i, 'constraints.resource_cap_max'),
            constraint_exists(model_run, i, 'constraints.resource_cap_min')
        ])
    ]
    constraint_sets['loc_techs_resource_capacity_equals_energy_capacity_constraint'] = [
        i for i in sets.loc_techs_finite_resource_supply_plus
        if constraint_exists(model_run, i, 'constraints.resource_cap_equals_energy_cap')
    ]
    constraint_sets['loc_techs_resource_area_constraint'] = sets.loc_techs_area
    constraint_sets['loc_techs_resource_area_per_energy_capacity_constraint'] = [
        i for i in sets.loc_techs_area
        if constraint_exists(model_run, i, 'constraints.resource_area_per_energy_cap')
        is not None
    ]
    constraint_sets['locs_resource_area_capacity_per_loc_constraint'] = [
        i for i in sets.locs
        if model_run.locations[i].get_key('available_area', None) is not None
        and sets.loc_techs_area
    ]
    constraint_sets['loc_techs_energy_capacity_constraint'] = [
        i for i in sets.loc_techs
        if i not in sets.loc_techs_milp + sets.loc_techs_purchase
    ]
    constraint_sets['techs_energy_capacity_systemwide_constraint'] = [
        i for i in sets.techs
        if model_run.get_key('techs.{}.constraints.energy_cap_max_systemwide'.format(i), None)
        or model_run.get_key('techs.{}.constraints.energy_cap_equals_systemwide'.format(i), None)
    ]

    # dispatch.py
    constraint_sets['loc_tech_carriers_carrier_production_max_constraint'] = [
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
    ]
    constraint_sets['loc_tech_carriers_carrier_production_min_constraint'] = [
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i.rsplit('::', 1)[0], 'constraints.energy_cap_min_use')
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
    ]
    constraint_sets['loc_tech_carriers_carrier_consumption_max_constraint'] = [
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
    ]
    constraint_sets['loc_techs_resource_max_constraint'] = sets.loc_techs_supply_plus
    constraint_sets['loc_tech_carriers_ramping_constraint'] = [
        i for i in sets.loc_tech_carriers_prod
        if i.rsplit('::', 1)[0] in sets.loc_techs_ramping
    ]
    # clustering-specific dispatch constraints
    if (model_run.model.get_key('time.function', None) == 'apply_clustering' and
            model_run.model.get_key('time.function_options.storage_inter_cluster', True)):
        constraint_sets['loc_techs_storage_intra_max_constraint'] = sets.loc_techs_store
        constraint_sets['loc_techs_storage_intra_min_constraint'] = sets.loc_techs_store
        constraint_sets['loc_techs_storage_inter_max_constraint'] = sets.loc_techs_store
        constraint_sets['loc_techs_storage_inter_min_constraint'] = sets.loc_techs_store
    else:
        constraint_sets['loc_techs_storage_max_constraint'] = sets.loc_techs_store

    # milp.py
    constraint_sets['loc_techs_unit_commitment_constraint'] = sets.loc_techs_milp
    constraint_sets['loc_techs_unit_capacity_constraint'] = sets.loc_techs_milp
    constraint_sets['loc_tech_carriers_carrier_production_max_milp_constraint'] = [
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
    ]
    constraint_sets['loc_techs_carrier_production_max_conversion_plus_milp_constraint'] = [
        i for i in sets.loc_techs_conversion_plus
        if i in sets.loc_techs_milp
    ]
    constraint_sets['loc_tech_carriers_carrier_production_min_milp_constraint'] = [
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i.rsplit('::', 1)[0], 'constraints.energy_cap_min_use')
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
    ]
    constraint_sets['loc_techs_carrier_production_min_conversion_plus_milp_constraint'] = [
        i for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i in sets.loc_techs_milp
    ]
    constraint_sets['loc_tech_carriers_carrier_consumption_max_milp_constraint'] = [
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
    ]
    constraint_sets['loc_techs_energy_capacity_units_constraint'] = [
        i for i in sets.loc_techs_milp
        if constraint_exists(model_run, i, 'constraints.energy_cap_per_unit')
        is not None
    ]
    constraint_sets['loc_techs_storage_capacity_units_constraint'] = [
        i for i in sets.loc_techs_milp if i in sets.loc_techs_store
    ]
    constraint_sets['loc_techs_energy_capacity_max_purchase_constraint'] = [
        i for i in sets.loc_techs_purchase
        if (constraint_exists(model_run, i, 'constraints.energy_cap_equals') is not None
            or (constraint_exists(model_run, i, 'constraints.energy_cap_max') is not None
                and constraint_exists(model_run, i, 'constraints.energy_cap_max') != np.inf))
    ]
    constraint_sets['loc_techs_energy_capacity_min_purchase_constraint'] = [
        i for i in sets.loc_techs_purchase
        if (not constraint_exists(model_run, i, 'constraints.energy_cap_equals')
            and constraint_exists(model_run, i, 'constraints.energy_cap_min'))
    ]
    constraint_sets['loc_techs_storage_capacity_max_purchase_constraint'] = [
        i for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        if (constraint_exists(model_run, i, 'constraints.storage_cap_equals') is not None
            or (constraint_exists(model_run, i, 'constraints.storage_cap_max') is not None
                and constraint_exists(model_run, i, 'constraints.storage_cap_max') != np.inf))
    ]
    constraint_sets['loc_techs_storage_capacity_min_purchase_constraint'] = [
        i for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        if (not constraint_exists(model_run, i, 'constraints.storage_cap_equals')
            and constraint_exists(model_run, i, 'constraints.storage_cap_min'))
    ]
    constraint_sets['loc_techs_update_costs_investment_units_constraint'] = [
        i for i in sets.loc_techs_milp
        if i in sets.loc_techs_investment_cost and
        any(constraint_exists(model_run, i, 'costs.{}.purchase'.format(j))
            for j in model_run.sets.costs)
    ]
    # loc_techs_purchase technologies only exist because they have defined a purchase cost
    constraint_sets['loc_techs_update_costs_investment_purchase_constraint'] = sets.loc_techs_purchase

    constraint_sets['techs_unit_capacity_systemwide_constraint'] = [
        i for i in sets.techs
        if model_run.get_key('techs.{}.constraints.units_max_systemwide'.format(i), None)
        or model_run.get_key('techs.{}.constraints.units_equals_systemwide'.format(i), None)
    ]

    # conversion.py
    constraint_sets['loc_techs_balance_conversion_constraint'] = sets.loc_techs_conversion
    constraint_sets['loc_techs_cost_var_conversion_constraint'] = sets.loc_techs_om_cost_conversion

    # conversion_plus.py
    constraint_sets['loc_techs_balance_conversion_plus_primary_constraint'] = sets.loc_techs_conversion_plus
    constraint_sets['loc_techs_carrier_production_max_conversion_plus_constraint'] = [
        i for i in sets.loc_techs_conversion_plus
        if i not in sets.loc_techs_milp
    ]
    constraint_sets['loc_techs_carrier_production_min_conversion_plus_constraint'] = [
        i for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i not in sets.loc_techs_milp
    ]
    constraint_sets['loc_techs_cost_var_conversion_plus_constraint'] = sets.loc_techs_om_cost_conversion_plus
    constraint_sets['loc_techs_balance_conversion_plus_in_2_constraint'] = sets.loc_techs_in_2
    constraint_sets['loc_techs_balance_conversion_plus_in_3_constraint'] = sets.loc_techs_in_3
    constraint_sets['loc_techs_balance_conversion_plus_out_2_constraint'] = sets.loc_techs_out_2
    constraint_sets['loc_techs_balance_conversion_plus_out_3_constraint'] = sets.loc_techs_out_3

    # network.py
    constraint_sets['loc_techs_symmetric_transmission_constraint'] = sets.loc_techs_transmission

    # policy.py
    constraint_sets['techlists_group_share_energy_cap_min_constraint'] = [
        i for i in sets.techlists
        if 'energy_cap_min' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
    ]
    constraint_sets['techlists_group_share_energy_cap_max_constraint'] = [
        i for i in sets.techlists
        if 'energy_cap_max' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
    ]
    constraint_sets['techlists_group_share_energy_cap_equals_constraint'] = [
        i for i in sets.techlists
        if 'energy_cap_equals' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
    ]
    constraint_sets['techlists_carrier_group_share_carrier_prod_min_constraint'] = [
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_min' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_min'.format(i), {}).keys()
    ]
    constraint_sets['techlists_carrier_group_share_carrier_prod_max_constraint'] = [
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_max' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_max'.format(i), {}).keys()
    ]
    constraint_sets['techlists_carrier_group_share_carrier_prod_equals_constraint'] = [
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_equals' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_equals'.format(i), {}).keys()
    ]

    return constraint_sets
