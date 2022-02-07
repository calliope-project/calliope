"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_checks.py
~~~~~~~~~~~~~~~~~~~~

Checks for model consistency and possible errors during preprocessing.

"""

import os
import logging

import numpy as np
import pandas as pd

from inspect import signature

import calliope
from calliope._version import __version__
from calliope.core.attrdict import AttrDict
from calliope.preprocess.util import get_all_carriers
from calliope.core.util.tools import load_function

logger = logging.getLogger(__name__)

DEFAULTS = AttrDict.from_yaml(
    os.path.join(os.path.dirname(calliope.__file__), "config", "defaults.yaml")
)
POSSIBLE_COSTS = [i for i in DEFAULTS.techs.default_tech.costs.default_cost.keys()]


def check_overrides(config_model, override):
    """
    Perform checks on the override dict and override file inputs to ensure they
    are not doing something silly.
    """
    model_warnings = []
    info = []
    for key in override.as_dict_flat().keys():
        if key in config_model.as_dict_flat().keys():
            info.append(
                "Override applied to {}: {} -> {}".format(
                    key, config_model.get_key(key), override.get_key(key)
                )
            )
        else:
            info.append(
                "`{}`:{} applied from override as new configuration".format(
                    key, override.get_key(key)
                )
            )

    # Check if overriding coordinates are in the same coordinate system. If not,
    # delete all incumbent coordinates, ready for the new coordinates to come in
    if any(["coordinates" in k for k in config_model.as_dict_flat().keys()]) and any(
        ["coordinates" in k for k in override.as_dict_flat().keys()]
    ):

        # get keys that might be deleted and incumbent coordinate system
        config_keys = [
            k for k in config_model.as_dict_flat().keys() if "coordinates." in k
        ]
        config_coordinates = set([k.split("coordinates.")[-1] for k in config_keys])

        # get overriding coordinate system
        override_coordinates = set(
            k.split("coordinates.")[-1]
            for k in override.as_dict_flat().keys()
            if "coordinates." in k
        )

        # compare overriding and incumbent, deleting incumbent if overriding is different
        if config_coordinates != override_coordinates:
            for key in config_keys:
                config_model.del_key(key)
            model_warnings.append(
                "Updated from coordinate system {} to {}, using overrides".format(
                    config_coordinates, override_coordinates
                )
            )

    if info:
        logger.info("\n".join(info))

    return model_warnings


def check_initial(config_model):
    """
    Perform initial checks of model and run config dicts.

    Returns
    -------
    model_warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    errors = []
    model_warnings = []

    # Check for version mismatch
    model_version = config_model.model.get("calliope_version", False)
    if model_version:
        if not str(model_version) in __version__:
            model_warnings.append(
                "Model configuration specifies calliope_version={}, "
                "but you are running {}. Proceed with caution!".format(
                    model_version, __version__
                )
            )

    # Check top-level keys
    for k in config_model.keys():
        if k not in [
            "model",
            "run",
            "locations",
            "tech_groups",
            "techs",
            "links",
            "overrides",
            "scenarios",
            "config_path",
            "group_constraints",
        ]:
            model_warnings.append(
                "Unrecognised top-level configuration item: {}".format(k)
            )

    # Check that all required top-level keys are specified
    for k in ["model", "run", "locations", "techs"]:
        if k not in config_model.keys():
            errors.append(
                "Model is missing required top-level configuration item: {}".format(k)
            )

    # Check run configuration
    # Exclude solver_options and objective_options.cost_class from checks,
    # as we don't know all possible options for all solvers
    for k in config_model["run"].keys_nested():
        if (
            k not in DEFAULTS["run"].keys_nested()
            and "solver_options" not in k
            and "objective_options.cost_class" not in k
        ):
            model_warnings.append(
                "Unrecognised setting in run configuration: {}".format(k)
            )

    # Check model configuration, but top-level keys only
    for k in config_model["model"].keys():
        if k not in DEFAULTS["model"].keys():
            model_warnings.append(
                "Unrecognised setting in model configuration: {}".format(k)
            )
    # If spores run mode is selected, check the correct definition of all needed parameters
    if config_model.run.mode == "spores":
        # Check that spores number is greater than 0, otherwise raise warning
        if config_model.run.spores_options.spores_number == 0:
            model_warnings.append(
                "spores run mode is selected, but a number of 0 spores is requested"
            )
        # Check that slack cost is greater than 0, otherwise set to default (0.1) and raise warning
        if config_model.run.spores_options.slack <= 0:
            config_model.run.spores_options.slack = 0.1
            model_warnings.append(
                "Slack must be greater than zero, setting slack to default value of 0.1 "
            )
        # Check that score_cost_class is a string
        _spores_cost_class = config_model.run.spores_options.get("score_cost_class", {})
        if not isinstance(_spores_cost_class, str):
            errors.append("`run.spores_options.score_cost_class` must be a string")
        # Check that slack_cost_group is one of the defined group contraints
        if (
            config_model.run.spores_options.slack_cost_group
            not in config_model.get("group_constraints", {}).keys()
        ):
            errors.append(
                "`run.spores_options.slack_cost_group` must correspond to "
                "one of the group constraints defined in the model"
            )

    # Only ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3']
    # are allowed as carrier tiers
    for key in config_model.as_dict_flat().keys():
        if "essentials.carrier_" in key and key.split(".carrier_")[-1].split(".")[
            0
        ] not in ["in", "out", "in_2", "out_2", "in_3", "out_3"]:
            errors.append(
                "Invalid carrier tier found at {}. Only "
                "'carrier_' + ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3'] "
                "is valid.".format(key)
            )

    # Warn if any unknown group constraints are defined
    permitted_group_constraints = DEFAULTS.group_constraints.default_group.keys()

    for group in config_model.get("group_constraints", {}).keys():
        for key in config_model.group_constraints[group].keys():
            if key not in permitted_group_constraints:
                model_warnings.append(
                    "Unrecognised group constraint `{}` in group `{}` "
                    "will be ignored - possibly a misspelling?".format(key, group)
                )

    # No techs may have the same identifier as a tech_group
    name_overlap = set(config_model.tech_groups.keys()) & set(config_model.techs.keys())
    if name_overlap:
        errors.append(
            "tech_groups and techs with " "the same name exist: {}".format(name_overlap)
        )

    # Checks for techs and tech_groups:
    # * All user-defined tech and tech_groups must specify a parent
    # * techs cannot be parents, only tech groups can
    # * No carrier may be called 'resource'
    default_tech_groups = list(DEFAULTS.tech_groups.keys())
    for tg_name, tg_config in config_model.tech_groups.items():
        if tg_name in default_tech_groups:
            continue
        if not tg_config.get_key("essentials.parent"):
            errors.append(
                "tech_group {} does not define " "`essentials.parent`".format(tg_name)
            )
        elif tg_config.get_key("essentials.parent") in config_model.techs.keys():
            errors.append(
                "tech_group `{}` has a tech as a parent, only another tech_group "
                "is allowed".format(tg_name)
            )
        if "resource" in get_all_carriers(tg_config.essentials):
            errors.append(
                "No carrier called `resource` may "
                "be defined (tech_group: {})".format(tg_name)
            )

    for t_name, t_config in config_model.techs.items():
        for key in t_config.keys():
            if key not in DEFAULTS.techs.default_tech.keys():
                model_warnings.append(
                    "Unknown key `{}` defined for tech {}.".format(key, t_name)
                )
        if not t_config.get_key("essentials.parent"):
            errors.append(
                "tech {} does not define " "`essentials.parent`".format(t_name)
            )
        elif t_config.get_key("essentials.parent") in config_model.techs.keys():
            errors.append(
                "tech `{}` has another tech as a parent, only a tech_group "
                "is allowed".format(tg_name)
            )
        if "resource" in get_all_carriers(t_config.essentials):
            errors.append(
                "No carrier called `resource` may "
                "be defined (tech: {})".format(t_name)
            )

    # Check whether any unrecognised mid-level keys are defined in techs, locations, or links
    for k, v in config_model.get("locations", {}).items():
        unrecognised_keys = [
            i for i in v.keys() if i not in DEFAULTS.locations.default_location.keys()
        ]
        if len(unrecognised_keys) > 0:
            errors.append(
                "Location `{}` contains unrecognised keys {}. "
                "These could be mispellings or a technology not defined "
                "under the `techs` key.".format(k, unrecognised_keys)
            )
        for loc_tech_key, loc_tech_val in v.get("techs", {}).items():
            if loc_tech_val is None:
                continue
            unrecognised_keys = [
                i
                for i in loc_tech_val.keys()
                if i not in DEFAULTS.techs.default_tech.keys()
            ]
            if len(unrecognised_keys) > 0:
                errors.append(
                    "Technology `{}` in location `{}` contains unrecognised keys {}; "
                    "these are most likely mispellings".format(
                        loc_tech_key, k, unrecognised_keys
                    )
                )

    default_link = DEFAULTS.links["default_location_from,default_location_to"]
    for k, v in config_model.get("links", {}).items():
        unrecognised_keys = [i for i in v.keys() if i not in default_link.keys()]
        if len(unrecognised_keys) > 0:
            errors.append(
                "Link `{}` contains unrecognised keys {}. "
                "These could be mispellings or a technology not defined "
                "under the `techs` key.".format(k, unrecognised_keys)
            )
        for link_tech_key, link_tech_val in v.get("techs", {}).items():
            if link_tech_val is None:
                continue
            unrecognised_keys = [
                i
                for i in link_tech_val.keys()
                if i not in default_link.techs.default_tech.keys()
                and i not in DEFAULTS.techs.default_tech.keys()
            ]
            if len(unrecognised_keys) > 0:
                errors.append(
                    "Technology `{}` in link `{}` contains unrecognised keys {}; "
                    "these are most likely mispellings".format(
                        link_tech_key, k, unrecognised_keys
                    )
                )

    # Error if a technology is defined twice, in opposite directions
    link_techs = [
        tuple(sorted(j.strip() for j in k.split(","))) + (i,)
        for k, v in config_model.get("links", {}).items()
        for i in v.get("techs", {}).keys()
    ]
    if len(link_techs) != len(set(link_techs)):
        duplicated_techs = np.array(link_techs)[
            pd.Series(link_techs).duplicated().values
        ]
        duplicated_techs = set([i[-1] for i in duplicated_techs])
        tech_end = "y" if len(duplicated_techs) == 1 else "ies"
        errors.append(
            "Technolog{} {} defined twice on a link defined in both directions "
            "(e.g. `A,B` and `B,A`). A technology can only be defined on one link "
            "even if it allows unidirectional flow in each direction "
            "(i.e. `one_way: true`).".format(tech_end, ", ".join(duplicated_techs))
        )

    # Error if a constraint is loaded from file that must not be
    allowed_from_file = DEFAULTS.model.file_allowed
    for k, v in config_model.as_dict_flat().items():
        if "file=" in str(v):

            possible_identifiers = k.split(".")
            is_time_varying = any("_time_varying" in i for i in possible_identifiers)
            if is_time_varying:
                model_warnings.append(
                    "Using custom constraint `{}` with time-varying data.".format(k)
                )
            elif (
                not set(possible_identifiers).intersection(allowed_from_file)
                and not is_time_varying
            ):
                errors.append(
                    "Cannot load data from file for configuration `{}`.".format(k)
                )

    # We no longer allow cost_class in objective_obtions to be a string
    _cost_class = config_model.run.objective_options.get("cost_class", {})

    if not isinstance(_cost_class, dict):
        errors.append(
            "`run.objective_options.cost_class` must be a dictionary."
            "If you want to minimise or maximise with a single cost class, "
            'use e.g. "{monetary: 1}", which gives the monetary cost class a weight '
            "of 1 in the objective, and ignores any other cost classes."
        )
    else:
        # This next check is only run if we have confirmed that cost_class is
        # a dict, as it errors otherwise

        # For cost minimisation objective, check for cost_class: None and set to one
        for k, v in _cost_class.items():
            if v is None:
                config_model.run.objective_options.cost_class[k] = 1
                model_warnings.append(
                    "cost class {} has weight = None, setting weight to 1".format(k)
                )

    if (
        isinstance(_cost_class, dict)
        and _cost_class.get("monetary", 0) == 1
        and len(_cost_class.keys()) > 1
    ):
        # Warn that {monetary: 1} is still in the objective, since it is not
        # automatically overidden on setting another objective.
        model_warnings.append(
            "Monetary cost class with a weight of 1 is still included "
            "in the objective. If you want to remove the monetary cost class, "
            'add `{"monetary": 0}` to the dictionary nested under '
            " `run.objective_options.cost_class`."
        )

    # Don't allow time clustering with cyclic storage if not also using
    # storage_inter_cluster
    storage_inter_cluster = "model.time.function_options.storage_inter_cluster"
    if (
        config_model.get_key("model.time.function", None) == "apply_clustering"
        and config_model.get_key("run.cyclic_storage", True)
        and not config_model.get_key(storage_inter_cluster, True)
    ):
        errors.append(
            "When time clustering, cannot have cyclic storage constraints if "
            "`storage_inter_cluster` decision variable is not activated."
        )

    return model_warnings, errors


def _check_tech_final(
    model_run, tech_id, tech_config, loc_id, model_warnings, errors, comments
):
    """
    Checks individual tech/tech groups at specific locations.
    NOTE: Updates `model_warnings` and `errors` lists in-place.
    """
    if tech_id not in model_run.techs:
        model_warnings.append(
            "Tech {} was removed by setting ``exists: False`` - not checking "
            "the consistency of its constraints at location {}.".format(tech_id, loc_id)
        )
        return model_warnings, errors

    required = model_run.techs[tech_id].required_constraints
    allowed = model_run.techs[tech_id].allowed_constraints
    allowed_costs = model_run.techs[tech_id].allowed_costs

    # Error if required constraints are not defined
    for r in required:
        # If it's a string, it must be defined
        single_ok = isinstance(r, str) and r in tech_config.constraints
        # If it's a list of strings, one of them must be defined
        multiple_ok = isinstance(r, list) and any(
            [i in tech_config.constraints for i in r]
        )
        if not single_ok and not multiple_ok:
            errors.append(
                "`{}` at `{}` fails to define "
                "all required constraints: {}".format(tech_id, loc_id, required)
            )

    # Warn if defining a carrier ratio for a conversion_plus tech,
    # but applying it to a carrier that isn't one of the carriers specified by that tech
    # e.g. carrier_ratios.carrier_in_2.cooling when cooling isn't a carrier`
    defined_carriers = get_all_carriers(model_run.techs[tech_id].essentials)
    carriers_in_ratios = [
        i.split(".")[-1]
        for i in tech_config.constraints.get_key("carrier_ratios", AttrDict())
        .as_dict_flat()
        .keys()
    ]
    for carrier in carriers_in_ratios:
        if carrier not in defined_carriers:
            model_warnings.append(
                "Tech `{t}` gives a carrier ratio for `{c}`, but does not actually "
                "configure `{c}` as a carrier.".format(t=tech_id, c=carrier)
            )

    # If the technology involves storage, warn when energy_cap and storage_cap aren't connected
    energy_cap_per_storage_cap_params = [
        "charge_rate",
        "energy_cap_per_storage_cap_min",
        "energy_cap_per_storage_cap_max",
        "energy_cap_per_storage_cap_equals",
    ]
    if loc_id + "::" + tech_id in model_run.sets.loc_techs_store and not any(
        i in tech_config.constraints.keys() for i in energy_cap_per_storage_cap_params
    ):
        logger.info(
            "`{}` at `{}` has no constraint to explicitly connect `energy_cap` to "
            "`storage_cap`, consider defining a `energy_cap_per_storage_cap_min/max/equals` "
            "constraint".format(tech_id, loc_id)
        )

    # If a technology is defined by units (i.e. integer decision variable), it must define energy_cap_per_unit
    if (
        any(["units_" in k for k in tech_config.constraints.keys()])
        and "energy_cap_per_unit" not in tech_config.constraints.keys()
    ):
        errors.append(
            "`{}` at `{}` fails to define energy_cap_per_unit when specifying "
            "technology in units_max/min/equals".format(tech_id, loc_id, required)
        )

    # If a technology is defined by units & is a storage tech, it must define storage_cap_per_unit
    if (
        any(["units_" in k for k in tech_config.constraints.keys()])
        and model_run.techs[tech_id].essentials.parent in ["storage", "supply_plus"]
        and any(["storage" in k for k in tech_config.constraints.keys()])
        and "storage_cap_per_unit" not in tech_config.constraints.keys()
    ):
        errors.append(
            "`{}` at `{}` fails to define storage_cap_per_unit when specifying "
            "technology in units_max/min/equals".format(tech_id, loc_id, required)
        )

    # If a technology defines force_resource but is not in loc_techs_finite_resource
    if (
        "force_resource" in tech_config.constraints.keys()
        and loc_id + "::" + tech_id not in model_run.sets.loc_techs_finite_resource
    ):

        model_warnings.append(
            "`{}` at `{}` defines force_resource but not a finite resource, so "
            "force_resource will not be applied".format(tech_id, loc_id)
        )

    # Gather remaining unallowed constraints
    remaining = set(tech_config.constraints) - set(allowed)

    # Error if something is defined that's not allowed, but is in defaults
    # Warn if something is defined that's not allowed, but is not in defaults
    # (it could be a misspelling)
    for k in remaining:
        if k in DEFAULTS.techs.default_tech.constraints.keys():
            errors.append(
                "`{}` at `{}` defines non-allowed "
                "constraint `{}`".format(tech_id, loc_id, k)
            )
        else:
            model_warnings.append(
                "`{}` at `{}` defines unrecognised "
                "constraint `{}` - possibly a misspelling?".format(tech_id, loc_id, k)
            )

    # Error if an `export` statement does not match the given carrier_outs
    if "export_carrier" in tech_config.constraints:
        essentials = model_run.techs[tech_id].essentials
        export = tech_config.constraints.export_carrier
        if export and export not in [
            essentials.get_key(k, "")
            for k in ["carrier_out", "carrier_out_2", "carrier_out_3"]
        ]:
            errors.append(
                "`{}` at `{}` is attempting to export a carrier "
                "not given as an output carrier: `{}`".format(tech_id, loc_id, export)
            )

    # Error if non-allowed costs are defined
    for cost_class in tech_config.get_key("costs", {}):
        for k in tech_config.costs[cost_class]:
            if k not in allowed_costs:
                errors.append(
                    "`{}` at `{}` defines non-allowed "
                    "{} cost: `{}`".format(tech_id, loc_id, cost_class, k)
                )

    # Error if non-allowed `resource_unit` is defined
    if tech_config.constraints.get("resource_unit", "energy") not in [
        "energy",
        "energy_per_cap",
        "energy_per_area",
    ]:
        errors.append(
            "`{}` is an unknown resource unit for `{}` at `{}`. "
            "Only `energy`, `energy_per_cap`, or `energy_per_area` is allowed.".format(
                tech_config.constraints.resource_unit, tech_id, loc_id
            )
        )

    return model_warnings, errors


def check_final(model_run):
    """
    Perform final checks of the completely built model_run.

    Returns
    -------
    comments : AttrDict
        debug output
    model_warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    model_warnings, errors = [], []
    comments = AttrDict()

    # Go through all loc-tech combinations and check validity
    for loc_id, loc_config in model_run.locations.items():
        if "techs" in loc_config:
            for tech_id, tech_config in loc_config.techs.items():
                _check_tech_final(
                    model_run,
                    tech_id,
                    tech_config,
                    loc_id,
                    model_warnings,
                    errors,
                    comments,
                )

        if "links" in loc_config:
            for link_id, link_config in loc_config.links.items():
                for tech_id, tech_config in link_config.techs.items():
                    _check_tech_final(
                        model_run,
                        tech_id,
                        tech_config,
                        "link {}:{}".format(loc_id, link_id),
                        model_warnings,
                        errors,
                        comments,
                    )

    # Either all locations or no locations must have coordinates
    all_locs = list(model_run.locations.keys())
    locs_with_coords = [
        k for k in model_run.locations.keys() if "coordinates" in model_run.locations[k]
    ]
    if len(locs_with_coords) != 0 and len(all_locs) != len(locs_with_coords):
        errors.append(
            "Either all or no locations must have `coordinates` defined. "
            "Locations defined: {} - Locations with coordinates: {}".format(
                all_locs, locs_with_coords
            )
        )

    # If locations have coordinates, they must all be either lat/lon or x/y
    elif len(locs_with_coords) != 0:
        first_loc = list(model_run.locations.keys())[0]
        try:
            coord_keys = sorted(list(model_run.locations[first_loc].coordinates.keys()))
            if coord_keys != ["lat", "lon"] and coord_keys != ["x", "y"]:
                errors.append(
                    "Unidentified coordinate system. All locations must either"
                    "use the format {lat: N, lon: M} or {x: N, y: M}."
                )
        except AttributeError:
            errors.append(
                "Coordinates must be given in the format {lat: N, lon: M} or "
                "{x: N, y: M}, not " + str(model_run.locations[first_loc].coordinates)
            )

        for loc_id, loc_config in model_run.locations.items():
            try:
                if sorted(list(loc_config.coordinates.keys())) != coord_keys:
                    errors.append("All locations must use the same coordinate format.")
                    break
            except AttributeError:
                errors.append(
                    "Coordinates must be given in the format {lat: N, lon: M} or "
                    "{x: N, y: M}, not "
                    + str(model_run.locations[first_loc].coordinates)
                )
                break

    # Ensure that timeseries have no non-unique index values
    for k, df in model_run["timeseries_data"].items():
        if df.index.duplicated().any():
            errors.append(
                "Time series `{}` contains non-unique timestamp values.".format(k)
            )

    # Warn if objective cost class is not defined elsewhere in the model
    objective_cost_class = set(model_run.run.objective_options.cost_class.keys())
    cost_classes = model_run.sets.costs
    cost_classes_mismatch = objective_cost_class.difference(cost_classes)
    if cost_classes_mismatch:
        model_warnings.append(
            "Cost classes `{}` are defined in the objective options but not "
            "defined elsewhere in the model. They will be ignored in the "
            "objective function.".format(cost_classes_mismatch)
        )

    # FIXME:
    # make sure `comments` is at the the base level:
    # i.e. comments.model_run.xxxxx....

    return comments, model_warnings, errors


def check_model_data(model_data):
    """
    Perform final checks of the completely built xarray Dataset `model_data`.

    Returns
    -------
    comments : AttrDict
        debug output
    model_warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    model_warnings, errors = [], []
    comments = AttrDict()

    # Ensure that no loc-tech specifies infinite resource and force_resource=True
    if "force_resource" in model_data.data_vars:
        relevant_loc_techs = [
            i.loc_techs_finite_resource.item()
            for i in model_data.force_resource
            if i.item() is True
        ]
        forced_resource = model_data.resource.loc[
            dict(loc_techs_finite_resource=relevant_loc_techs)
        ]
        conflict = forced_resource.where(forced_resource == np.inf).to_pandas().dropna()
        if not conflict.empty:
            errors.append(
                "loc_tech(s) {} cannot have `force_resource` set as infinite "
                "resource values are given".format(", ".join(conflict.index))
            )

    # Ensure that if a tech has negative costs, there is a max cap defined
    # FIXME: doesn't consider capacity being set by a linked constraint e.g.
    # `resource_cap_per_energy_cap`.
    relevant_caps = [
        i
        for i in ["energy_cap", "storage_cap", "resource_cap", "resource_area"]
        if "cost_" + i in model_data.data_vars.keys()
    ]
    for cap in relevant_caps:
        relevant_loc_techs = (
            model_data["cost_" + cap]
            .where(model_data["cost_" + cap] < 0, drop=True)
            .to_pandas()
        )
        cap_max = cap + "_max"
        cap_equals = cap + "_equals"
        for loc_tech in relevant_loc_techs.columns:
            try:
                cap_val = model_data[cap_max][loc_tech].item()
            except KeyError:
                try:
                    cap_val = model_data[cap_equals][loc_tech].item()
                except KeyError:
                    cap_val = np.nan
            if np.isinf(cap_val) or np.isnan(cap_val):
                errors.append(
                    "loc_tech {} cannot have a negative cost_{} as the "
                    "corresponding capacity constraint is not set".format(loc_tech, cap)
                )

    for loc_tech in set(model_data.loc_techs_demand.values).intersection(
        model_data.loc_techs_finite_resource.values
    ):
        if any(model_data.resource.sel(loc_techs_finite_resource=loc_tech).values > 0):
            errors.append(
                "Positive resource given for demand loc_tech {}. All demands "
                "must have negative resource".format(loc_tech)
            )

    # Delete all empty dimensions & the variables associated with them
    for dim_name, dim_length in model_data.dims.items():
        if dim_length == 0:
            if dim_name in model_data.coords.keys():
                del model_data[dim_name]
            associated_vars = [
                var
                for var, data in model_data.data_vars.items()
                if dim_name in data.dims
            ]
            model_data = model_data.drop(associated_vars)
            model_warnings.append(
                "dimension {} and associated variables {} were empty, so have "
                "been deleted".format(dim_name, ", ".join(associated_vars))
            )

    # Check if we're allowed to use operate mode
    if "allow_operate_mode" not in model_data.attrs.keys():
        daily_timesteps = [
            model_data.timestep_resolution.loc[i].values
            for i in np.unique(model_data.timesteps.to_index().strftime("%Y-%m-%d"))
        ]
        if not np.all(daily_timesteps == daily_timesteps[0]):
            model_data.attrs["allow_operate_mode"] = 0
            model_warnings.append(
                "Operational mode requires the same timestep resolution profile "
                "to be emulated on each date"
            )
        else:
            model_data.attrs["allow_operate_mode"] = 1

    # Check for any milp constraints, and warn that the problem contains binary /
    # integer decision variables
    if any("_milp_constraint" in i for i in model_data.dims):
        model_warnings.append(
            "Integer and / or binary decision variables are included in this model. "
            "This may adversely affect solution time, particularly if you are "
            "using a non-commercial solver. To improve solution time, consider "
            "changing MILP related solver options (e.g. `mipgap`) or removing "
            "MILP constraints."
        )

    # Check for storage_initial being a fractional value

    if hasattr(model_data, "loc_techs_store"):
        for loc_tech in model_data.loc_techs_store.values:
            if hasattr(model_data, "storage_initial"):
                if (
                    model_data.storage_initial.loc[{"loc_techs_store": loc_tech}].values
                    > 1
                ):
                    errors.append(
                        "storage_initial values larger than 1 are not allowed."
                    )
                if (
                    model_data.storage_initial.loc[{"loc_techs_store": loc_tech}].values
                    == 0
                ):
                    model_data.storage_initial.loc[{"loc_techs_store": loc_tech}] = 0.0

    # Check for storage_initial being greater than or equal to the storage_discharge_depth

    if hasattr(model_data, "loc_techs_store"):
        for loc_tech in model_data.loc_techs_store.values:
            if hasattr(model_data, "storage_initial") and hasattr(
                model_data, "storage_discharge_depth"
            ):
                if (
                    model_data.storage_initial.loc[{"loc_techs_store": loc_tech}].values
                    < model_data.storage_discharge_depth.loc[
                        {"loc_techs_store": loc_tech}
                    ].values
                ):
                    errors.append(
                        "storage_initial is smaller than storage_discharge_depth."
                        " Please change the model configuration to ensure that"
                        " storage initial is greater than or equal to storage_discharge_depth"
                    )

    # Check for storage_inter_cluster not being used together with storage_discharge_depth
    if hasattr(model_data, "clusters") and hasattr(
        model_data, "storage_discharge_depth"
    ):
        errors.append(
            "storage_discharge_depth is currently not allowed when time clustering is active."
        )

    return model_data, comments, model_warnings, errors
