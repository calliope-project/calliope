"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model_data.py
~~~~~~~~~~~~~~~~~~

Functionality to build the model-internal data array and process
time-varying parameters.

"""

import collections

import ruamel.yaml
import xarray as xr
import numpy as np
import pandas as pd

from calliope.core.attrdict import AttrDict
from calliope._version import __version__
from calliope.preprocess import checks
from calliope.preprocess.util import split_loc_techs_transmission, concat_iterable
from calliope.preprocess.time import add_time_dimension
from calliope.preprocess.lookup import add_lookup_arrays


def build_model_data(model_run, debug=False):
    """
    Take a Calliope model_run and convert it into an xarray Dataset, ready for
    constraint generation. Timeseries data is also extracted from file at this
    point, and the time dimension added to the data

    Parameters
    ----------
    model_run : AttrDict
        preprocessed model_run dictionary, as produced by
        Calliope.preprocess.preprocess_model
    debug : bool, default = False
        Used to debug steps within build_model_data, particularly before/after
        time dimension addition. If True, more information is returned

    Returns
    -------
    data : xarray Dataset
        Dataset with optimisation parameters as variables, optimisation sets as
        coordinates, and other information in attributes.
    data_dict : dict, only returned if debug = True
        dictionary of parameters, prior to time dimension addition. Used here to
        populate the Dataset (using `from_dict()`)
    data_pre_time : xarray Dataset, only returned if debug = True
        Dataset, prior to time dimension addition, with optimisation parameters
        as variables, optimisation sets as coordinates, and other information
        in attributes.
    """
    # We build up a dictionary of the data, then convert it to an xarray Dataset
    # before applying time dimensions
    data = xr.Dataset(coords=add_sets(model_run), attrs=add_attributes(model_run))

    data_dict = dict()
    data_dict.update(constraints_to_dataset(model_run))
    data_dict.update(costs_to_dataset(model_run))
    data_dict.update(location_specific_to_dataset(model_run))
    data_dict.update(tech_specific_to_dataset(model_run))
    data_dict.update(carrier_specific_to_dataset(model_run))
    data_dict.update(group_constraints_to_dataset(model_run))

    data = data.merge(xr.Dataset.from_dict(data_dict))

    data = add_lookup_arrays(data, model_run)

    if debug:
        data_pre_time = data.copy(deep=True)

    data = add_time_dimension(data, model_run)

    data = update_dtypes(data)
    # Carrier information uses DataArray indexing in the function, so we merge
    # these directly into the main xarray Dataset

    if debug:
        return data, data_dict, data_pre_time
    else:
        return data


def add_sets(model_run):
    coords = dict()
    for key, value in model_run.sets.items():
        if value:
            coords[key] = value
    for key, value in model_run.constraint_sets.items():
        if value:
            coords[key] = value
    return coords


def constraints_to_dataset(model_run):
    """
    Extract all constraints from the processed dictionary (model.model_run) and
    return an xarray Dataset with all the constraints as DataArray variables and
    model sets as Dataset dimensions.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    data_dict = dict()

    # FIXME: hardcoding == bad
    def _get_set(constraint):
        """
        return the set of loc_techs over which the given constraint should be
        built
        """
        if "_area" in constraint:
            return "loc_techs_area"
        elif any(
            i in constraint for i in ["resource_cap", "parasitic", "resource_min_use"]
        ):
            return "loc_techs_supply_plus"
        elif (
            "resource" in constraint
        ):  # i.e. everything with 'resource' in the name that isn't resource_cap
            return "loc_techs_finite_resource"
        elif (
            "storage" in constraint
            or "charge_rate" in constraint
            or "energy_cap_per_storage_cap" in constraint
        ):
            return "loc_techs_store"
        elif "purchase" in constraint:
            return "loc_techs_purchase"
        elif "units_" in constraint:
            return "loc_techs_milp"
        elif "export" in constraint:
            return "loc_techs_export"
        else:
            return "loc_techs"

    # find all constraints which are actually defined in the yaml file
    relevant_constraints = set(
        i.split(".constraints.")[1]
        for i in model_run.locations.as_dict_flat().keys()
        if ".constraints." in i and ".carrier_ratios." not in i
    )
    for constraint in relevant_constraints:
        data_dict[constraint] = dict(dims=_get_set(constraint), data=[])
        for loc_tech in model_run.sets[_get_set(constraint)]:
            loc, tech = loc_tech.split("::", 1)
            # for transmission technologies, we also need to go into link nesting
            if ":" in tech:  # i.e. transmission technologies
                tech, link = tech.split(":")
                loc_tech_dict = model_run.locations[loc].links[link].techs[tech]
            else:  # all other technologies
                loc_tech_dict = model_run.locations[loc].techs[tech]
            constraint_value = loc_tech_dict.constraints.get(constraint, np.nan)
            # inf is assumed to be string on import, so we need to np.inf it
            if constraint_value == "inf":
                constraint_value = np.inf
            # add the value for the particular location & technology combination to the list
            data_dict[constraint]["data"].append(constraint_value)
        # once we've looped through all technology & location combinations, add the array to the dataset

    group_share_data = {}
    group_constraints = ["energy_cap_min", "energy_cap_max", "energy_cap_equals"]
    group_constraints_carrier = [
        "carrier_prod_min",
        "carrier_prod_max",
        "carrier_prod_equals",
        "carrier_con_min",
        "carrier_con_max",
        "carrier_con_equals",
    ]

    for constraint in [  # Only process constraints that are defined
        c
        for c in group_constraints
        if c
        in "".join(model_run.model.get_key("group_share", AttrDict()).keys_nested())
    ]:
        group_share_data[constraint] = [
            model_run.model.get_key(
                "group_share.{}.{}".format(techlist, constraint), np.nan
            )
            for techlist in model_run.sets["techlists"]
        ]

    for constraint in [  # Only process constraints that are defined
        c
        for c in group_constraints_carrier
        if c
        in "".join(model_run.model.get_key("group_share", AttrDict()).keys_nested())
    ]:
        group_share_data[constraint] = [
            [
                model_run.model.get_key(
                    "group_share.{}.{}.{}".format(techlist, constraint, carrier), np.nan
                )
                for techlist in model_run.sets["techlists"]
            ]
            for carrier in model_run.sets["carriers"]
        ]

    # Add to data_dict and set dims correctly
    for k in group_share_data:
        data_dict["group_share_" + k] = {
            "data": group_share_data[k],
            "dims": "techlists"
            if k in group_constraints
            else ("carriers", "techlists"),
        }

    return data_dict


def costs_to_dataset(model_run):
    """
    Extract all costs from the processed dictionary (model.model_run) and
    return an xarray Dataset with all the costs as DataArray variables. Variable
    names will be prepended with `cost_` to differentiate from other constraints

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    data_dict = dict()

    # FIXME: hardcoding == bad
    def _get_set(cost):
        """
        return the set of loc_techs over which the given cost should be built
        """
        if any(i in cost for i in ["_cap", "depreciation_rate", "purchase", "area"]):
            return "loc_techs_investment_cost"
        elif any(i in cost for i in ["om_", "export"]):
            return "loc_techs_om_cost"
        else:
            return "loc_techs"

    # find all cost classes and associated costs which are actually defined in the model_run
    costs = set(
        i.split(".costs.")[1].split(".")[1]
        for i in model_run.locations.as_dict_flat().keys()
        if ".costs." in i
    )
    cost_classes = model_run.sets["costs"]

    # loop over unique costs, cost classes and technology & location combinations
    for cost in costs:
        data_dict["cost_" + cost] = dict(dims=["costs", _get_set(cost)], data=[])
        for cost_class in cost_classes:
            cost_class_array = []
            for loc_tech in model_run.sets[_get_set(cost)]:
                loc, tech = loc_tech.split("::", 1)
                # for transmission technologies, we also need to go into link nesting
                if ":" in tech:  # i.e. transmission technologies
                    tech, link = tech.split(":")
                    loc_tech_dict = model_run.locations[loc].links[link].techs[tech]
                else:  # all other technologies
                    loc_tech_dict = model_run.locations[loc].techs[tech]
                cost_dict = loc_tech_dict.get_key("costs." + cost_class, None)

                # inf is assumed to be string on import, so need to np.inf it
                cost_value = np.nan if not cost_dict else cost_dict.get(cost, np.nan)
                # add the value for the particular location & technology combination to the correct cost class list
                cost_class_array.append(cost_value)
            data_dict["cost_" + cost]["data"].append(cost_class_array)

    return data_dict


def carrier_specific_to_dataset(model_run):
    """
    Extract carrier information from the processed dictionary (model.model_run)
    and return an xarray Dataset with DataArray variables describing carrier_in,
    carrier_out, and carrier_ratio (for conversion plus technologies) information.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    carrier_tiers = model_run.sets["carrier_tiers"]
    loc_tech_dict = {k: [] for k in model_run.sets["loc_techs_conversion_plus"]}
    data_dict = dict()
    # Set information per carrier tier ('out', 'out_2', 'in', etc.)
    # for conversion-plus technologies
    if model_run.sets["loc_techs_conversion_plus"]:
        # carrier ratios are the floating point numbers used to compare one
        # carrier_in/_out value with another carrier_in/_out value
        data_dict["carrier_ratios"] = dict(
            dims=["carrier_tiers", "loc_tech_carriers_conversion_plus"], data=[]
        )
        for carrier_tier in carrier_tiers:
            data = []
            for loc_tech_carrier in model_run.sets["loc_tech_carriers_conversion_plus"]:
                loc, tech, carrier = loc_tech_carrier.split("::")
                carrier_ratio = (
                    model_run.locations[loc]
                    .techs[tech]
                    .constraints.get_key(
                        "carrier_ratios.carrier_" + carrier_tier + "." + carrier, 1
                    )
                )
                data.append(carrier_ratio)
                loc_tech_dict[loc + "::" + tech].append(carrier_ratio)
            data_dict["carrier_ratios"]["data"].append(data)

    # Additional system-wide constraints from model_run.model
    if model_run.model.get("reserve_margin", {}) != {}:
        data_dict["reserve_margin"] = {
            "data": [
                model_run.model.reserve_margin.get(c, np.nan)
                for c in model_run.sets["carriers"]
            ],
            "dims": "carriers",
        }

    return data_dict


def location_specific_to_dataset(model_run):
    """
    Extract location specific information from the processed dictionary
    (model.model_run) and return an xarray Dataset with DataArray variables
    describing distance, coordinate and available area information.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    # for every transmission technology, we extract distance information, if it
    # is available
    data_dict = dict()

    data_dict["distance"] = dict(
        dims="loc_techs_transmission",
        data=[
            model_run.get_key(
                "locations.{loc_from}.links.{loc_to}.techs.{tech}.distance".format(
                    **split_loc_techs_transmission(loc_tech)
                ),
                np.nan,
            )
            for loc_tech in model_run.sets["loc_techs_transmission"]
        ],
    )
    # If there is no distance information stored, distance array is deleted
    if data_dict["distance"]["data"].count(np.nan) == len(
        data_dict["distance"]["data"]
    ):
        del data_dict["distance"]

    data_dict["lookup_remotes"] = dict(
        dims="loc_techs_transmission",
        data=concat_iterable(
            [
                (k["loc_to"], k["tech"], k["loc_from"])
                for k in [
                    split_loc_techs_transmission(loc_tech)
                    for loc_tech in model_run.sets["loc_techs_transmission"]
                ]
            ],
            ["::", ":"],
        ),
    )
    # If there are no remote locations stored, lookup_remotes array is deleted
    if data_dict["lookup_remotes"]["data"].count(np.nan) == len(
        data_dict["lookup_remotes"]["data"]
    ):
        del data_dict["lookup_remotes"]

    data_dict["available_area"] = dict(
        dims="locs",
        data=[
            model_run.locations[loc].get("available_area", np.nan)
            for loc in model_run.sets["locs"]
        ],
    )

    # remove this dictionary element if nothing is defined in it
    if set(data_dict["available_area"]["data"]) == {np.nan}:
        del data_dict["available_area"]

    # Coordinates are defined per location, but may not be defined at all for
    # the model
    if "coordinates" in model_run.sets:
        data_dict["loc_coordinates"] = dict(dims=["locs", "coordinates"], data=[])
        for loc in model_run.sets["locs"]:
            data_dict["loc_coordinates"]["data"].append(
                [
                    model_run.locations[loc].coordinates[coordinate]
                    for coordinate in model_run.sets.coordinates
                ]
            )

    return data_dict


def tech_specific_to_dataset(model_run):
    """
    Extract technology (location inspecific) information from the processed
    dictionary (model.model_run) and return an xarray Dataset with DataArray
    variables describing color and inheritance chain information.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    data_dict = collections.defaultdict(lambda: {"dims": ["techs"], "data": []})

    systemwide_constraints = set(
        [
            k.split(".")[-1]
            for k in model_run.techs.keys_nested()
            if ".constraints." in k and k.endswith("_systemwide")
        ]
    )

    for tech in model_run.sets["techs"]:
        if tech in model_run.sets["techs_transmission"]:
            tech = tech.split(":")[0]
        data_dict["colors"]["data"].append(
            model_run.techs[tech].get_key("essentials.color")
        )
        data_dict["inheritance"]["data"].append(
            ".".join(model_run.techs[tech].get_key("inheritance"))
        )
        data_dict["names"]["data"].append(
            # Default to tech ID if no name is set
            model_run.techs[tech].get_key("essentials.name", tech)
        )
        for k in systemwide_constraints:
            data_dict[k]["data"].append(
                model_run.techs[tech].constraints.get_key(k, np.nan)
            )

    return data_dict


def group_constraints_to_dataset(model_run):
    data_dict = {}

    group_constraints = model_run["group_constraints"]

    for constr_name in model_run.constraint_sets["group_constraints"]:
        constr_group_name = f"group_names_{constr_name}"
        dims = [constr_group_name]
        constr = checks.DEFAULTS.group_constraints.default_group.get(constr_name, None)
        group_names = model_run.constraint_sets.get(constr_group_name, None)
        if group_names is not None:
            if isinstance(constr, dict):
                if "default_carrier" in constr.keys():
                    data = [
                        list(group_constraints[i][constr_name].as_dict_flat().values())[
                            0
                        ]
                        for i in group_names
                    ]
                elif "default_cost" in constr.keys():
                    dims.append("costs")
                    data = [
                        [
                            group_constraints[i][constr_name].get(cost, np.nan)
                            for cost in model_run.sets["costs"]
                        ]
                        for i in group_names
                    ]
            else:
                data = [group_constraints[i][constr_name] for i in group_names]
        else:
            continue

        data_dict[f"group_{constr_name}"] = {"dims": dims, "data": data}

    return data_dict


def add_attributes(model_run):
    attr_dict = AttrDict()

    attr_dict["calliope_version"] = __version__
    attr_dict["applied_overrides"] = model_run["applied_overrides"]
    attr_dict["scenario"] = model_run["scenario"]

    ##
    # Build the `defaults` attribute that holds all default settings
    # used in get_param() lookups inside the backend
    ##

    default_tech_dict = checks.DEFAULTS.techs.default_tech.as_dict()
    default_location_dict = checks.DEFAULTS.locations.default_location.as_dict()

    # Group constraint defaults are a little bit more involved
    default_group_constraint_keys = [
        i
        for i in checks.DEFAULTS.group_constraints.default_group.keys()
        if i not in ["locs", "techs", "exists"]
    ]
    default_group_constraint_dict = {}
    for k in default_group_constraint_keys:
        k_default = checks.DEFAULTS.group_constraints.default_group[k]
        if isinstance(k_default, dict):
            assert len(k_default.keys()) == 1
            default_group_constraint_dict["group_" + k] = k_default[
                list(k_default.keys())[0]
            ]
        else:
            default_group_constraint_dict["group_" + k] = k_default

    attr_dict["defaults"] = ruamel.yaml.dump(
        {
            **default_tech_dict["constraints"],
            **{
                "cost_{}".format(k): v
                for k, v in default_tech_dict["costs"]["default_cost"].items()
            },
            **default_location_dict,
            **default_group_constraint_dict,
        }
    )

    return attr_dict


def update_dtypes(model_data):
    """
    Update dtypes to not be 'Object', if possible.
    Order of preference is: bool, int, float
    """
    for var_name, var in model_data.data_vars.items():
        if var.dtype.kind in ["O", "U"]:
            no_nans = var.where(var != "nan", drop=True)
            model_data[var_name] = var.where(var != "nan")

            if (
                no_nans.isin(["True", "False", "0", "1"])
                | no_nans.isin([0, 1])
                | no_nans.isin([True, False])
            ).all():
                # Turn to bool
                model_data[var_name] = (
                    var.isin(["True", "1"]) | var.isin([1]) | var.isin([True])
                )
            else:
                try:
                    model_data[var_name] = var.astype(int, copy=False)
                except (ValueError, TypeError):
                    try:
                        model_data[var_name] = var.astype(float, copy=False)
                    except ValueError:
                        continue
    return model_data
