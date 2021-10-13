"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

constraint_sets.py
~~~~~~~~~~~~~~~~~~

loc_techs, loc_carriers, and loc_tech_carriers subsets used per constraint, to
reduce constraint complexity

"""

from itertools import product

import numpy as np

from calliope.preprocess.util import constraint_exists, concat_iterable
from calliope.preprocess import checks
from calliope import exceptions


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
    constraint_sets["loc_carriers_system_balance_constraint"] = sets.loc_carriers
    constraint_sets[
        "loc_techs_balance_supply_constraint"
    ] = sets.loc_techs_finite_resource_supply
    constraint_sets[
        "loc_techs_balance_demand_constraint"
    ] = sets.loc_techs_finite_resource_demand
    constraint_sets[
        "loc_techs_resource_availability_supply_plus_constraint"
    ] = sets.loc_techs_finite_resource_supply_plus
    constraint_sets[
        "loc_techs_balance_transmission_constraint"
    ] = sets.loc_techs_transmission
    constraint_sets[
        "loc_techs_balance_supply_plus_constraint"
    ] = sets.loc_techs_supply_plus
    constraint_sets["loc_techs_balance_storage_constraint"] = sets.loc_techs_storage
    if model_run.run.cyclic_storage is True:
        constraint_sets["loc_techs_storage_initial_constraint"] = [
            i
            for i in sets.loc_techs_store
            if constraint_exists(model_run, i, "constraints.storage_initial")
            is not None
        ]
    constraint_sets["loc_techs_storage_discharge_depth"] = [
        i
        for i in sets.loc_techs_store
        if constraint_exists(model_run, i, "constraints.storage_discharge_depth")
    ]
    constraint_sets["carriers_reserve_margin_constraint"] = [
        i
        for i in sets.carriers
        if i in model_run.model.get_key("reserve_margin", {}).keys()
    ]
    # clustering-specific balance constraints
    if model_run.model.get_key(
        "time.function", None
    ) == "apply_clustering" and model_run.model.get_key(
        "time.function_options.storage_inter_cluster", True
    ):
        set_name = "loc_techs_balance_storage_inter_cluster_constraint"
        constraint_sets[set_name] = sets.loc_techs_store

    # costs.py
    constraint_sets["loc_techs_cost_constraint"] = sets.loc_techs_cost
    constraint_sets[
        "loc_techs_cost_investment_constraint"
    ] = sets.loc_techs_investment_cost
    constraint_sets["loc_techs_cost_var_constraint"] = [
        i
        for i in sets.loc_techs_om_cost
        if i not in sets.loc_techs_conversion_plus + sets.loc_techs_conversion
    ]

    # export.py
    constraint_sets["loc_carriers_update_system_balance_constraint"] = [
        i
        for i in sets.loc_carriers
        if sets.loc_techs_export
        and any(
            [
                "{0}::{2}".format(*j.split("::")) == i
                for j in sets.loc_tech_carriers_export
            ]
        )
    ]
    constraint_sets[
        "loc_tech_carriers_export_balance_constraint"
    ] = sets.loc_tech_carriers_export
    constraint_sets["loc_techs_update_costs_var_constraint"] = [
        i for i in sets.loc_techs_om_cost if i in sets.loc_techs_export
    ]

    constraint_sets["loc_tech_carriers_export_max_constraint"] = [
        i
        for i in sets.loc_tech_carriers_export
        if constraint_exists(model_run, i.rsplit("::", 1)[0], "constraints.export_cap")
        is not None
    ]

    # capacity.py
    constraint_sets["loc_techs_storage_capacity_constraint"] = [
        i for i in sets.loc_techs_store if i not in sets.loc_techs_milp
    ]
    constraint_sets["loc_techs_energy_capacity_storage_constraint_old"] = [
        i
        for i in sets.loc_techs_store
        if constraint_exists(model_run, i, "constraints.charge_rate")
    ]
    constraint_sets["loc_techs_energy_capacity_storage_equals_constraint"] = [
        i
        for i in sets.loc_techs_store
        if constraint_exists(
            model_run, i, "constraints.energy_cap_per_storage_cap_equals"
        )
    ]
    constraint_sets["loc_techs_energy_capacity_storage_min_constraint"] = [
        i
        for i in sets.loc_techs_store
        if constraint_exists(model_run, i, "constraints.energy_cap_per_storage_cap_min")
        and not constraint_exists(
            model_run, i, "constraints.energy_cap_per_storage_cap_equals"
        )
    ]
    constraint_sets["loc_techs_energy_capacity_storage_max_constraint"] = [
        i
        for i in sets.loc_techs_store
        if constraint_exists(model_run, i, "constraints.energy_cap_per_storage_cap_max")
        and not constraint_exists(
            model_run, i, "constraints.energy_cap_per_storage_cap_equals"
        )
    ]
    constraint_sets["loc_techs_resource_capacity_constraint"] = [
        i
        for i in sets.loc_techs_finite_resource_supply_plus
        if any(
            [
                constraint_exists(model_run, i, "constraints.resource_cap_equals"),
                constraint_exists(model_run, i, "constraints.resource_cap_max"),
                constraint_exists(model_run, i, "constraints.resource_cap_min"),
            ]
        )
    ]
    constraint_sets["loc_techs_resource_capacity_equals_energy_capacity_constraint"] = [
        i
        for i in sets.loc_techs_finite_resource_supply_plus
        if constraint_exists(model_run, i, "constraints.resource_cap_equals_energy_cap")
    ]
    constraint_sets["loc_techs_resource_area_constraint"] = sets.loc_techs_area
    constraint_sets["loc_techs_resource_area_per_energy_capacity_constraint"] = [
        i
        for i in sets.loc_techs_area
        if constraint_exists(model_run, i, "constraints.resource_area_per_energy_cap")
        is not None
    ]
    constraint_sets["locs_resource_area_capacity_per_loc_constraint"] = [
        i
        for i in sets.locs
        if model_run.locations[i].get_key("available_area", None) is not None
        and sets.loc_techs_area
    ]
    constraint_sets["loc_techs_energy_capacity_constraint"] = [
        i
        for i in sets.loc_techs
        if i not in sets.loc_techs_milp + sets.loc_techs_purchase
    ]
    constraint_sets["techs_energy_capacity_systemwide_constraint"] = [
        i
        for i in sets.techs
        if model_run.get_key(
            "techs.{}.constraints.energy_cap_max_systemwide".format(i), None
        )
        or model_run.get_key(
            "techs.{}.constraints.energy_cap_equals_systemwide".format(i), None
        )
    ]

    # dispatch.py
    constraint_sets["loc_tech_carriers_carrier_production_max_constraint"] = [
        i
        for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit("::", 1)[0] not in sets.loc_techs_milp
    ]
    constraint_sets["loc_tech_carriers_carrier_production_min_constraint"] = [
        i
        for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(
            model_run, i.rsplit("::", 1)[0], "constraints.energy_cap_min_use"
        )
        and i.rsplit("::", 1)[0] not in sets.loc_techs_milp
    ]
    constraint_sets["loc_tech_carriers_carrier_consumption_max_constraint"] = [
        i
        for i in sets.loc_tech_carriers_con
        if i.rsplit("::", 1)[0]
        in sets.loc_techs_demand + sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit("::", 1)[0] not in sets.loc_techs_milp
    ]
    constraint_sets["loc_techs_resource_max_constraint"] = sets.loc_techs_supply_plus
    constraint_sets["loc_tech_carriers_ramping_constraint"] = [
        i
        for i in sets.loc_tech_carriers_prod
        if i.rsplit("::", 1)[0] in sets.loc_techs_ramping
    ]
    # clustering-specific dispatch constraints
    if model_run.model.get_key(
        "time.function", None
    ) == "apply_clustering" and model_run.model.get_key(
        "time.function_options.storage_inter_cluster", True
    ):
        constraint_sets["loc_techs_storage_intra_max_constraint"] = sets.loc_techs_store
        constraint_sets["loc_techs_storage_intra_min_constraint"] = sets.loc_techs_store
        constraint_sets["loc_techs_storage_inter_max_constraint"] = sets.loc_techs_store
        constraint_sets["loc_techs_storage_inter_min_constraint"] = sets.loc_techs_store
    else:
        constraint_sets["loc_techs_storage_max_constraint"] = sets.loc_techs_store

    # milp.py
    constraint_sets["loc_techs_unit_commitment_milp_constraint"] = sets.loc_techs_milp
    constraint_sets["loc_techs_unit_capacity_milp_constraint"] = sets.loc_techs_milp
    constraint_sets["loc_tech_carriers_carrier_production_max_milp_constraint"] = [
        i
        for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit("::", 1)[0] in sets.loc_techs_milp
    ]
    constraint_sets[
        "loc_techs_carrier_production_max_conversion_plus_milp_constraint"
    ] = [i for i in sets.loc_techs_conversion_plus if i in sets.loc_techs_milp]
    constraint_sets["loc_tech_carriers_carrier_production_min_milp_constraint"] = [
        i
        for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(
            model_run, i.rsplit("::", 1)[0], "constraints.energy_cap_min_use"
        )
        and i.rsplit("::", 1)[0] in sets.loc_techs_milp
    ]
    constraint_sets[
        "loc_techs_carrier_production_min_conversion_plus_milp_constraint"
    ] = [
        i
        for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, "constraints.energy_cap_min_use")
        and i in sets.loc_techs_milp
    ]
    constraint_sets["loc_tech_carriers_carrier_consumption_max_milp_constraint"] = [
        i
        for i in sets.loc_tech_carriers_con
        if i.rsplit("::", 1)[0]
        in sets.loc_techs_demand + sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit("::", 1)[0] in sets.loc_techs_milp
    ]
    constraint_sets["loc_techs_energy_capacity_units_milp_constraint"] = [
        i
        for i in sets.loc_techs_milp
        if constraint_exists(model_run, i, "constraints.energy_cap_per_unit")
        is not None
    ]
    constraint_sets["loc_techs_storage_capacity_units_milp_constraint"] = [
        i for i in sets.loc_techs_milp if i in sets.loc_techs_store
    ]
    constraint_sets["loc_techs_energy_capacity_max_purchase_milp_constraint"] = [
        i
        for i in sets.loc_techs_purchase
        if (
            constraint_exists(model_run, i, "constraints.energy_cap_equals") is not None
            or (
                constraint_exists(model_run, i, "constraints.energy_cap_max")
                is not None
                and constraint_exists(model_run, i, "constraints.energy_cap_max")
                != np.inf
            )
        )
    ]
    constraint_sets["loc_techs_energy_capacity_min_purchase_milp_constraint"] = [
        i
        for i in sets.loc_techs_purchase
        if (
            not constraint_exists(model_run, i, "constraints.energy_cap_equals")
            and constraint_exists(model_run, i, "constraints.energy_cap_min")
        )
    ]
    constraint_sets["loc_techs_storage_capacity_max_purchase_milp_constraint"] = [
        i
        for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        if (
            constraint_exists(model_run, i, "constraints.storage_cap_equals")
            is not None
            or (
                constraint_exists(model_run, i, "constraints.storage_cap_max")
                is not None
                and constraint_exists(model_run, i, "constraints.storage_cap_max")
                != np.inf
            )
        )
    ]
    constraint_sets["loc_techs_storage_capacity_min_purchase_milp_constraint"] = [
        i
        for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        if (
            not constraint_exists(model_run, i, "constraints.storage_cap_equals")
            and constraint_exists(model_run, i, "constraints.storage_cap_min")
        )
    ]
    constraint_sets["loc_techs_update_costs_investment_units_milp_constraint"] = [
        i
        for i in sets.loc_techs_milp
        if i in sets.loc_techs_investment_cost
        and any(
            constraint_exists(model_run, i, "costs.{}.purchase".format(j))
            for j in model_run.sets.costs
        )
    ]

    # loc_techs_purchase technologies only exist because they have defined a purchase cost
    constraint_sets[
        "loc_techs_update_costs_investment_purchase_milp_constraint"
    ] = sets.loc_techs_purchase

    constraint_sets["techs_unit_capacity_systemwide_milp_constraint"] = [
        i
        for i in sets.techs
        if model_run.get_key(
            "techs.{}.constraints.units_max_systemwide".format(i), None
        )
        or model_run.get_key(
            "techs.{}.constraints.units_equals_systemwide".format(i), None
        )
    ]
    constraint_sets[
        "loc_techs_asynchronous_prod_con_milp_constraint"
    ] = sets.loc_techs_asynchronous_prod_con

    # conversion.py
    constraint_sets[
        "loc_techs_balance_conversion_constraint"
    ] = sets.loc_techs_conversion
    constraint_sets[
        "loc_techs_cost_var_conversion_constraint"
    ] = sets.loc_techs_om_cost_conversion

    # conversion_plus.py
    constraint_sets[
        "loc_techs_balance_conversion_plus_primary_constraint"
    ] = sets.loc_techs_conversion_plus
    constraint_sets["loc_techs_carrier_production_max_conversion_plus_constraint"] = [
        i for i in sets.loc_techs_conversion_plus if i not in sets.loc_techs_milp
    ]
    constraint_sets["loc_techs_carrier_production_min_conversion_plus_constraint"] = [
        i
        for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, "constraints.energy_cap_min_use")
        and i not in sets.loc_techs_milp
    ]
    constraint_sets[
        "loc_techs_cost_var_conversion_plus_constraint"
    ] = sets.loc_techs_om_cost_conversion_plus
    constraint_sets[
        "loc_techs_balance_conversion_plus_in_2_constraint"
    ] = sets.loc_techs_in_2
    constraint_sets[
        "loc_techs_balance_conversion_plus_in_3_constraint"
    ] = sets.loc_techs_in_3
    constraint_sets[
        "loc_techs_balance_conversion_plus_out_2_constraint"
    ] = sets.loc_techs_out_2
    constraint_sets[
        "loc_techs_balance_conversion_plus_out_3_constraint"
    ] = sets.loc_techs_out_3

    # network.py
    constraint_sets[
        "loc_techs_symmetric_transmission_constraint"
    ] = sets.loc_techs_transmission

    # policy.py
    for sense in ["min", "max", "equals"]:
        constraint_sets[f"techlists_group_share_energy_cap_{sense}_constraint"] = [
            i
            for i in sets.techlists
            if f"energy_cap_{sense}"
            in model_run.model.get_key("group_share.{}".format(i), {}).keys()
        ]
        constraint_sets[
            f"techlists_carrier_group_share_carrier_prod_{sense}_constraint"
        ] = [
            i + "::" + carrier
            for i in sets.techlists
            if f"carrier_prod_{sense}"
            in model_run.model.get_key("group_share.{}".format(i), {}).keys()
            for carrier in sets.carriers
            if carrier
            in model_run.model.get_key(
                f"group_share.{i}.carrier_prod_{sense}", {}
            ).keys()
        ]

    # group.py
    group_constraints = {
        name: data
        for name, data in model_run["group_constraints"].items()
        if data.get("exists", True)
    }
    constraint_sets["group_constraints"] = set()
    for group_constraint_name, group_constraint in group_constraints.items():
        tech_groups = [
            [
                k
                for k, v in checks.DEFAULTS.tech_groups.items()
                if i in v["allowed_group_constraints"]
            ]
            for i in group_constraint.keys()
            if i not in ["techs", "locs", "exists"]
        ]
        allowed_tech_groups = set(tech_groups[0]).intersection(*tech_groups)
        allowed_techs = sum(
            [sets["techs_{}".format(i)] for i in allowed_tech_groups], []
        )
        techs = group_constraint.get("techs", allowed_techs)

        locs = group_constraint.get("locs", sets["locs"])
        trans_techs = set(techs).intersection(sets["techs_transmission_names"])
        for i in trans_techs:
            techs += [i + ":" + j for j in sets["locs"]]
            techs.remove(i)
        # If the group constraint defines its own techs, remove those that are not allowed
        techs = list(set(techs).intersection(allowed_techs))

        # All possible loc_techs for this constraint
        loc_techs_all = list(
            set(concat_iterable([(l, t) for l, t in product(locs, techs)], ["::"]))
        )

        # Some loc_techs may not actually exist in the model,
        # so we must filter with actually exising loc_techs
        loc_techs = [i for i in loc_techs_all if i in sets.loc_techs]
        default_group_config = checks.DEFAULTS.group_constraints.default_group
        _constraints = {
            k: v
            for k, v in group_constraint.items()
            if k not in ["locs", "techs", "exists"]
        }
        if any(
            isinstance(default_group_config.get(_constraint, None), dict)
            and "default_carrier" in default_group_config[_constraint].keys()
            for _constraint in _constraints.keys()
        ):
            if len(_constraints) > 1:
                raise exceptions.ModelError(
                    "Can only handle one constraint in a group constraint if one of them is carrier-based"
                )
            _name, _config = list(_constraints.items())[0]
            loc_tech_carrier_dict = _get_carrier_group_constraint_loc_techs(
                loc_techs, locs, _config, _name, sets, constraint_sets
            )
            if any(len(val) == 0 for val in loc_tech_carrier_dict.values()):
                exceptions.warn(
                    f"Constraint group `{group_constraint_name}` will be completely ignored since there are no valid location::technology::carrier combinations"
                )
                continue
            else:
                for key, loc_tech_carriers in loc_tech_carrier_dict.items():
                    constraint_sets[
                        key.format(group_constraint_name)
                    ] = loc_tech_carriers

        else:
            if len(loc_techs) == 0:
                exceptions.warn(
                    f"Constraint group `{group_constraint_name}` will be completely ignored since there are no valid location::technology combinations"
                )
                break
            constraint_sets[
                "group_constraint_loc_techs_{}".format(group_constraint_name)
            ] = loc_techs
        _add_to_group_constraint_mapping(
            constraint_sets, group_constraint_name, list(_constraints.keys())
        )
    constraint_sets["group_constraints"] = list(constraint_sets["group_constraints"])
    return constraint_sets


def _get_carrier_group_constraint_loc_techs(
    loc_techs, locs, config, constraint_name, sets, constraint_sets
):
    flow = "con" if "_con_" in constraint_name else "prod"
    if len(config.keys()) > 1:
        raise exceptions.ModelError(
            "Can only handle one carrier per group constraint that is carrier-based"
        )
    carrier = list(config.keys())[0]
    if "net_import" in constraint_name:
        _loc_tech_carriers = _get_net_import_loc_tech_carrier_subset(
            loc_techs,
            sets["loc_tech_carriers_con"] + sets["loc_tech_carriers_prod"],
            carrier,
            locs,
        )
    else:
        _loc_tech_carriers = _get_loc_tech_carrier_subset(
            loc_techs, sets[f"loc_tech_carriers_{flow}"], carrier, locs
        )

    if "share" in constraint_name:
        lhs_loc_tech_carriers = _loc_tech_carriers
        if "demand" in constraint_name or "import" in constraint_name:
            rhs_loc_tech_carriers = _get_loc_tech_carrier_subset(
                sets["loc_techs_demand"], sets["loc_tech_carriers_con"], carrier, locs
            )
        elif flow == "con":
            rhs_loc_tech_carriers = _get_loc_tech_carrier_subset(
                sets["loc_techs_demand"] + sets["loc_techs_conversion_all"],
                sets["loc_tech_carriers_con"],
                carrier,
                locs,
            )
        elif flow == "prod":
            rhs_loc_tech_carriers = _get_loc_tech_carrier_subset(
                sets["loc_techs_supply_conversion_all"],
                sets["loc_tech_carriers_prod"],
                carrier,
                locs,
            )
        return {
            "group_constraint_loc_tech_carriers_{}_lhs": list(lhs_loc_tech_carriers),
            "group_constraint_loc_tech_carriers_{}_rhs": list(rhs_loc_tech_carriers),
        }
    else:
        return {"group_constraint_loc_tech_carriers_{}": list(_loc_tech_carriers)}


def _get_loc_tech_carrier_subset(loc_techs, loc_tech_carriers, carrier, locs):
    return set(
        f"{loc_tech}::{carrier}"
        for loc_tech in loc_techs
        if f"{loc_tech}::{carrier}" in loc_tech_carriers
        and loc_tech.split("::")[0] in locs
    )


def _get_net_import_loc_tech_carrier_subset(
    loc_techs, loc_tech_carriers, carrier, locs
):
    return set(
        f"{loc_tech}::{carrier}"
        for loc_tech in loc_techs
        if f"{loc_tech}::{carrier}" in loc_tech_carriers
        and loc_tech.split(":")[-1] not in locs
    )


def _add_to_group_constraint_mapping(constraint_sets, group_name, constraint_names):
    for constraint_name in constraint_names:
        if (
            f"group_names_{constraint_name}" in constraint_sets.keys()
            and group_name not in constraint_sets[f"group_names_{constraint_name}"]
        ):
            constraint_sets[f"group_names_{constraint_name}"] += [group_name]
        else:
            constraint_sets[f"group_names_{constraint_name}"] = [group_name]
        constraint_sets["group_constraints"].update([constraint_name])
