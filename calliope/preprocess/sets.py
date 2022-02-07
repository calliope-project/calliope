"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_sets.py
~~~~~~~~~~~~~~~~~~

Sets & sub-sets defined to reduce size of decision variables & constraints.

The markers preceded with `###` are used to auto-include the set description
in the documentation

###PART TO INCLUDE IN DOCUMENTATION STARTS HERE###

Main sets
=========

* timesteps
* techs
* techs_non_transmission
* techs_transmission
* locs
* costs
* resources
* carriers
* carrier_tiers
* techlists

Location-technology subsets
===========================

* loc_techs

Technology groups

* loc_techs_storage
* loc_techs_transmission
* loc_techs_demand
* loc_techs_supply
* loc_techs_supply_plus
* loc_techs_conversion
* loc_techs_conversion_plus

Subsets based on active constraints

* loc_techs_area
* loc_techs_store
* loc_techs_finite_resource
* loc_techs_finite_resource_supply
* loc_techs_finite_resource_demand
* loc_techs_finite_resource_supply_plus
* loc_techs_ramping
* loc_techs_export
* loc_techs_purchase
* loc_techs_milp
* loc_techs_cost
* loc_techs_investment_cost
* loc_techs_om_cost
* loc_techs_out_2
* loc_techs_out_3
* loc_techs_in_2
* loc_techs_in_3


Subsets that include carrier

* loc_tech_carriers_prod
* loc_tech_carriers_con
* loc_tech_carriers_conversion_plus
* loc_tech_carriers_export
* loc_carriers

###PART TO INCLUDE IN DOCUMENTATION ENDS HERE###

"""

from itertools import product

import numpy as np

from calliope.core.attrdict import AttrDict
from calliope.preprocess.util import (
    get_all_carriers,
    split_loc_techs_transmission,
    concat_iterable,
    flatten_list,
)


def generate_simple_sets(model_run):
    """
    Generate basic sets for a given pre-processed ``model_run``.

    Parameters
    ----------
    model_run : AttrDict

    """
    sets = AttrDict()

    flat_techs = model_run.techs.as_dict(flat=True)
    flat_locations = model_run.locations.as_dict(flat=True)

    sets.resources = set(
        flatten_list(v for k, v in flat_techs.items() if ".carrier" in k)
    )

    sets.carriers = sets.resources - set(["resource"])

    sets.carrier_tiers = set(
        key.split(".carrier_")[1] for key in flat_techs.keys() if ".carrier_" in key
    )

    sets.costs = set(
        k.split("costs.")[-1].split(".")[0]
        for k in flat_locations.keys()
        if ".costs." in k
    )

    sets.locs = set(model_run.locations.keys())

    sets.techs_non_transmission = set()
    tech_groups = [
        "demand",
        "supply",
        "supply_plus",
        "conversion",
        "conversion_plus",
        "storage",
    ]
    for tech_group in tech_groups:
        sets["techs_{}".format(tech_group)] = set(
            k for k, v in model_run.techs.items() if v.inheritance[-1] == tech_group
        )
        sets.techs_non_transmission.update(sets["techs_{}".format(tech_group)])

    sets.techs_transmission_names = set(
        k for k, v in model_run.techs.items() if v.inheritance[-1] == "transmission"
    )

    # This builds the "tech:loc" expansion of transmission technologies
    techs_transmission = set()
    for loc_name, loc_config in model_run.locations.items():
        for link_name, link_config in loc_config.get("links", {}).items():
            for tech_name in link_config.techs:
                techs_transmission.add("{}:{}".format(tech_name, link_name))
    sets.techs_transmission = techs_transmission

    sets.techs = sets.techs_non_transmission | sets.techs_transmission_names

    # this extracts location coordinate information
    coordinates = set(
        k.split(".")[-1] for k in flat_locations.keys() if ".coordinates." in k
    )

    if coordinates:
        sets.coordinates = coordinates

    # `timesteps` set is built from the results of timeseries_data processing
    sets.timesteps = list(model_run.timesteps.astype(str))
    model_run.del_key("timesteps")

    # `techlists` are strings with comma-separated techs used for grouping in
    # some model-wide constraints
    sets.techlists = set()
    for k in model_run.model.get_key("group_share", {}).keys():
        sets.techlists.add(k)

    return sets


def generate_loc_tech_sets(model_run, simple_sets):
    """
    Generate loc-tech sets for a given pre-processed ``model_run``

    Parameters
    ----------
    model_run : AttrDict
    simple_sets : AttrDict
        Simple sets returned by ``generate_simple_sets(model_run)``.

    """
    sets = AttrDict()

    ##
    # First deal with transmission techs, which can show up only in
    # loc_techs_transmission, loc_techs_milp, and loc_techs_purchase
    ##

    # All `tech:loc` expanded transmission technologies
    sets.loc_techs_transmission = set(
        concat_iterable(
            [
                (i, u, j)
                for i, j, u in product(  # (loc, loc, tech) product
                    simple_sets.locs,
                    simple_sets.locs,
                    simple_sets.techs_transmission_names,
                )
                if model_run.get_key(
                    "locations.{}.links.{}.techs.{}".format(i, j, u), None
                )
            ],
            ["::", ":"],
        )
    )

    # A dict of transmission tech config objects
    # to make parsing for set membership easier
    loc_techs_transmission_config = {
        k: model_run.get_key(
            "locations.{loc_from}.links.{loc_to}.techs.{tech}".format(
                **split_loc_techs_transmission(k)
            )
        )
        for k in sets.loc_techs_transmission
    }

    ##
    # Now deal with the rest of the techs and other sets
    ##

    # Only loc-tech combinations that actually exist
    sets.loc_techs_non_transmission = set(
        concat_iterable(
            [
                (l, t)
                for l, t in product(
                    simple_sets.locs, simple_sets.techs_non_transmission
                )
                if model_run.get_key("locations.{}.techs.{}".format(l, t), None)
            ],
            ["::"],
        )
    )

    sets.loc_techs = sets.loc_techs_non_transmission | sets.loc_techs_transmission

    # A dict of non-transmission tech config objects
    # to make parsing for set membership easier
    loc_techs_config = {
        k: model_run.get_key("locations.{}.techs.{}".format(*k.split("::")))
        for k in sets.loc_techs_non_transmission
    }

    loc_techs_all_config = {**loc_techs_config, **loc_techs_transmission_config}

    ##
    # Sets based on membership in abstract base technology groups
    ##

    for group in [
        "storage",
        "demand",
        "supply",
        "supply_plus",
        "conversion",
        "conversion_plus",
    ]:
        tech_set = set(
            k
            for k in sets.loc_techs_non_transmission
            if model_run.techs[k.split("::")[1]].inheritance[-1] == group
        )
        sets["loc_techs_{}".format(group)] = tech_set

    sets.loc_techs_non_conversion = (
        set(
            k
            for k in sets.loc_techs_non_transmission
            if k not in sets.loc_techs_conversion
            and k not in sets.loc_techs_conversion_plus
        )
        | sets.loc_techs_transmission
    )

    # Techs that introduce energy into the system
    sets.loc_techs_supply_all = sets.loc_techs_supply | sets.loc_techs_supply_plus

    # Techs that change the energy carrier in the system
    sets.loc_techs_conversion_all = (
        sets.loc_techs_conversion | sets.loc_techs_conversion_plus
    )
    # All techs that can be used to generate a carrier (not just store or move it)
    sets.loc_techs_supply_conversion_all = (
        sets.loc_techs_supply_all | sets.loc_techs_conversion_all
    )

    ##
    # Sets based on specific constraints being active
    ##

    # Technologies that specify resource_area constraints
    sets.loc_techs_area = set(
        k
        for k in sets.loc_techs_non_transmission
        if (
            any("resource_area" in i for i in loc_techs_config[k].keys_nested())
            or loc_techs_config[k].constraints.get("resource_unit", "energy")
            == "energy_per_area"
        )
    )

    # Technologies that define storage, which can include `supply_plus`
    # and `storage` groups.
    sets.loc_techs_store = (
        set(
            k
            for k in sets.loc_techs_supply_plus
            if any(
                "storage_" in i for i in loc_techs_config[k].constraints.keys_nested()
            )
        )
        | sets.loc_techs_storage
    )

    # technologies that specify a finite resource
    sets.loc_techs_finite_resource = set(
        k
        for k in sets.loc_techs_non_transmission
        if loc_techs_config[k].constraints.get("resource")
        and not (
            loc_techs_config[k].constraints.get("resource")
            in ["inf", np.inf, "-inf", -np.inf]
        )
    )

    # `supply` technologies that specify a finite resource
    sets.loc_techs_finite_resource_supply = sets.loc_techs_finite_resource.intersection(
        sets.loc_techs_supply
    )

    # `demand` technologies that specify a finite resource
    sets.loc_techs_finite_resource_demand = sets.loc_techs_finite_resource.intersection(
        sets.loc_techs_demand
    )

    # `supply_plus` technologies that specify a finite resource
    sets.loc_techs_finite_resource_supply_plus = (
        sets.loc_techs_finite_resource.intersection(sets.loc_techs_supply_plus)
    )

    # Technologies that define ramping constraints
    sets.loc_techs_ramping = set(
        k
        for k in sets.loc_techs_non_transmission
        if "energy_ramping" in loc_techs_config[k].constraints
    )

    # Technologies that allow export
    sets.loc_techs_export = set(
        k
        for k in sets.loc_techs_non_transmission
        if "export_carrier" in loc_techs_config[k].constraints
    )

    # Technologies that allow purchasing discrete units
    # NB: includes transmission techs!
    loc_techs_purchase = set(
        k
        for k in sets.loc_techs_non_transmission
        if any(
            ".purchase" in i
            for i in loc_techs_config[k].get("costs", AttrDict()).keys_nested()
        )
        and not any(
            "units_" in i
            for i in loc_techs_config[k].get("constraints", AttrDict()).keys_nested()
        )
    )

    transmission_purchase = set(
        k
        for k in sets.loc_techs_transmission
        if any(
            ".purchase" in i
            for i in loc_techs_transmission_config[k]
            .get("costs", AttrDict())
            .keys_nested()
        )
        and not any(
            "units_" in i
            for i in loc_techs_transmission_config[k]
            .get("constraints", AttrDict())
            .keys_nested()
        )
    )

    sets.loc_techs_purchase = loc_techs_purchase | transmission_purchase

    # Technologies with MILP constraints
    loc_techs_milp = set(
        k
        for k in sets.loc_techs_non_transmission
        if any("units_" in i for i in loc_techs_config[k].constraints.keys_nested())
    )

    transmission_milp = set(
        k
        for k in sets.loc_techs_transmission
        if any(
            "units_" in i
            for i in loc_techs_transmission_config[k].constraints.keys_nested()
        )
    )

    sets.loc_techs_milp = loc_techs_milp | transmission_milp

    # Technologies with forced asynchronous production/consumption of energy
    loc_techs_storage_asynchronous_prod_con = set(
        k
        for k in sets.loc_techs_store
        if "force_asynchronous_prod_con"
        in loc_techs_config[k].constraints.keys_nested()
    )

    loc_techs_transmission_asynchronous_prod_con = set(
        k
        for k in sets.loc_techs_transmission
        if "force_asynchronous_prod_con"
        in loc_techs_transmission_config[k].constraints.keys_nested()
    )
    sets.loc_techs_asynchronous_prod_con = (
        loc_techs_storage_asynchronous_prod_con
        | loc_techs_transmission_asynchronous_prod_con
    )

    ##
    # Sets based on specific costs being active
    # NB includes transmission techs
    ##

    loc_techs_costs = set(
        k
        for k in sets.loc_techs_non_transmission
        if any("costs" in i for i in loc_techs_config[k].keys())
    )

    loc_techs_transmission_costs = set(
        k
        for k in sets.loc_techs_transmission
        if any("costs" in i for i in loc_techs_transmission_config[k].keys())
    )

    # Any capacity or fixed annual costs
    loc_techs_investment_costs = set(
        k
        for k in loc_techs_costs
        if any(
            "_cap" in i or ".purchase" in i or "_area" in i
            for i in loc_techs_config[k].costs.keys_nested()
        )
    )
    loc_techs_transmission_investment_costs = set(
        k
        for k in loc_techs_transmission_costs
        if any(
            "_cap" in i or ".purchase" in i or "_area" in i
            for i in loc_techs_transmission_config[k].costs.keys_nested()
        )
    )

    # Any operation and maintenance
    loc_techs_om_costs = set(
        k
        for k in loc_techs_costs
        if any(
            "om_" in i or "export" in i for i in loc_techs_config[k].costs.keys_nested()
        )
    )
    loc_techs_transmission_om_costs = set(
        k
        for k in loc_techs_transmission_costs
        if any("om_" in i for i in loc_techs_transmission_config[k].costs.keys_nested())
    )

    # Any export costs
    sets.loc_techs_costs_export = set(
        k
        for k in loc_techs_costs
        if any("export" in i for i in loc_techs_config[k].costs.keys_nested())
    )

    sets.loc_techs_cost = loc_techs_costs | loc_techs_transmission_costs
    sets.loc_techs_investment_cost = (
        loc_techs_investment_costs | loc_techs_transmission_investment_costs
    )
    sets.loc_techs_om_cost = loc_techs_om_costs | loc_techs_transmission_om_costs

    ##
    # Subsets of costs for different abstract base technologies
    ##

    sets.loc_techs_om_cost_conversion = loc_techs_om_costs.intersection(
        sets.loc_techs_conversion
    )
    sets.loc_techs_om_cost_conversion_plus = loc_techs_om_costs.intersection(
        sets.loc_techs_conversion_plus
    )
    sets.loc_techs_om_cost_supply = loc_techs_om_costs.intersection(
        sets.loc_techs_supply
    )
    sets.loc_techs_om_cost_supply_plus = loc_techs_om_costs.intersection(
        sets.loc_techs_supply_plus
    )

    ##
    # Subsets of `conversion_plus` technologies
    ##

    # `conversion_plus` technologies with secondary carrier(s) out
    sets.loc_techs_out_2 = set(
        k
        for k in sets.loc_techs_conversion_plus
        if "carrier_out_2" in model_run.techs[k.split("::")[1].split(":")[0]].essentials
    )

    # `conversion_plus` technologies with tertiary carrier(s) out
    sets.loc_techs_out_3 = set(
        k
        for k in sets.loc_techs_conversion_plus
        if "carrier_out_3" in model_run.techs[k.split("::")[1].split(":")[0]].essentials
    )

    # `conversion_plus` technologies with secondary carrier(s) in
    sets.loc_techs_in_2 = set(
        k
        for k in sets.loc_techs_conversion_plus
        if "carrier_in_2" in model_run.techs[k.split("::")[1].split(":")[0]].essentials
    )

    # `conversion_plus` technologies with tertiary carrier(s) in
    sets.loc_techs_in_3 = set(
        k
        for k in sets.loc_techs_conversion_plus
        if "carrier_in_3" in model_run.techs[k.split("::")[1].split(":")[0]].essentials
    )

    ##
    # `loc_tech_carrier` sets
    ##

    # loc_tech_carriers for all technologies that have energy_prod=True
    sets.loc_tech_carriers_prod = set(
        "{}::{}".format(k, carrier)
        for k in sets.loc_techs
        if loc_techs_all_config[k].constraints.get_key("energy_prod", False)
        for carrier in get_all_carriers(
            model_run.techs[k.split("::")[1].split(":")[0]].essentials, direction="out"
        )
    )

    # loc_tech_carriers for all technologies that have energy_con=True
    sets.loc_tech_carriers_con = set(
        "{}::{}".format(k, carrier)
        for k in sets.loc_techs
        if loc_techs_all_config[k].constraints.get_key("energy_con", False)
        for carrier in get_all_carriers(
            model_run.techs[k.split("::")[1].split(":")[0]].essentials, direction="in"
        )
    )

    # loc_tech_carriers for all supply technologies
    sets.loc_tech_carriers_supply_all = set(
        "{}::{}".format(k, carrier)
        for k in sets.loc_techs_supply_all
        for carrier in get_all_carriers(
            model_run.techs[k.split("::")[1].split(":")[0]].essentials, direction="out"
        )
    )

    # loc_tech_carriers for all conversion technologies
    sets.loc_tech_carriers_conversion_all = set(
        "{}::{}".format(k, carrier)
        for k in sets.loc_techs_conversion_all
        for carrier in get_all_carriers(
            model_run.techs[k.split("::")[1].split(":")[0]].essentials, direction="out"
        )
    )

    # loc_tech_carriers for all supply and conversion technologies
    sets.loc_tech_carriers_supply_conversion_all = (
        sets.loc_tech_carriers_supply_all | sets.loc_tech_carriers_conversion_all
    )
    # loc_tech_carriers for all demand technologies
    sets.loc_tech_carriers_demand = set(
        "{}::{}".format(k, carrier)
        for k in sets.loc_techs_demand
        for carrier in get_all_carriers(
            model_run.techs[k.split("::")[1].split(":")[0]].essentials, direction="in"
        )
    )

    # loc_tech_carriers for all technologies that have export
    sets.loc_tech_carriers_export = set(
        "{}::{}".format(k, loc_techs_all_config[k].constraints.export_carrier)
        for k in sets.loc_techs
        if loc_techs_all_config[k].constraints.get_key("export_carrier", False)
    )

    # loc_tech_carriers for `conversion_plus` technologies
    sets.loc_tech_carriers_conversion_plus = set(
        k
        for k in sets.loc_tech_carriers_con | sets.loc_tech_carriers_prod
        if k.rsplit("::", 1)[0] in sets.loc_techs_conversion_plus
    )

    # loc_carrier combinations that exist with either a con or prod tech
    sets.loc_carriers = set(
        "{0}::{2}".format(*k.split("::"))
        for k in sets.loc_tech_carriers_prod | sets.loc_tech_carriers_con
    )

    return sets
