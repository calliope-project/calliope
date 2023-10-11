# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
preprocess_checks.py
~~~~~~~~~~~~~~~~~~~~

Checks for model consistency and possible errors during preprocessing.

"""

import logging
import os
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import calliope
from calliope._version import __version__
from calliope.core.attrdict import AttrDict
from calliope.preprocess.util import get_all_carriers

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


def check_initial(config_model: AttrDict):
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
    model_version = config_model.config.init["calliope_version"]
    if model_version is not None:
        if str(model_version) not in __version__:
            model_warnings.append(
                "Model configuration specifies calliope_version={}, "
                "but you are running {}. Proceed with caution!".format(
                    model_version, __version__
                )
            )

    # Check top-level keys
    for k in config_model.keys():
        if k not in [
            "config",
            "parameters",
            "nodes",
            "locations",  # TODO: remove in v0.7.1
            "tech_groups",
            "techs",
            "links",
            "overrides",
            "scenarios",
            "_model_def_path",
        ]:
            model_warnings.append(
                "Unrecognised top-level configuration item: {}".format(k)
            )

    # Check that all required top-level keys are specified
    for k in ["config", "nodes", "techs"]:
        if k not in config_model.keys():
            if k == "nodes" and "locations" in config_model.keys():
                # TODO: remove in v0.7.1
                warnings.warn(
                    "`locations` has been renamed to `nodes` and will stop working "
                    "in v0.7.1. Please update your model configuration accordingly.",
                    DeprecationWarning,
                )
            else:
                errors.append(
                    "Model is missing required top-level configuration item: {}".format(
                        k
                    )
                )

    # Check run configuration
    # Exclude solver_options and objective_options.cost_class from checks,
    # as we don't know all possible options for all solvers
    for config_group in ["init", "build", "solve"]:
        for k in config_model.config[config_group].keys():
            if k not in DEFAULTS.config[config_group].keys():
                model_warnings.append(
                    f"Unrecognised setting in `{config_group}` configuration: {k}"
                )

    # If spores run mode is selected, check the correct definition of all needed parameters
    if config_model.config.build.mode == "spores":
        # Check that spores number is greater than 0, otherwise raise warning
        if config_model.config.solve.spores_number == 0:
            model_warnings.append(
                "spores run mode is selected, but a number of 0 spores is requested"
            )
        # Check that slack cost is greater than 0
        if config_model.parameters.spores_slack <= 0:
            errors.append("`parameters.spores_slack` must be > 0")

        # Check that score_cost_class is a string
        _spores_cost_class = config_model.config.solve.spores_score_cost_class
        if not isinstance(_spores_cost_class, str):
            errors.append("`config.solve.spores_score_cost_class` must be a string")

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

    # No techs may have the same identifier as a tech_group
    name_overlap = set(config_model.tech_groups.keys()) & set(config_model.techs.keys())
    if name_overlap:
        errors.append(
            "tech_groups and techs with " "the same name exist: {}".format(name_overlap)
        )

    # Checks for techs and tech_groups:
    # * All user-defined tech and tech_groups must specify a parent
    # * techs cannot be parents, only tech groups can
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

    for t_name, t_config in config_model.techs.items():
        for key in t_config.keys():
            if key not in DEFAULTS.techs.default_tech.keys():
                model_warnings.append(
                    "Unknown key `{}` defined for tech {}.".format(key, t_name)
                )
        if not t_config.get_key("essentials.parent", False):
            errors.append(
                "tech {} does not define " "`essentials.parent`".format(t_name)
            )
        elif t_config.get_key("essentials.parent", None) in config_model.techs.keys():
            errors.append(
                "tech `{}` has another tech as a parent, only a tech_group "
                "is allowed".format(tg_name)
            )

    # Check whether any unrecognised mid-level keys are defined in techs, nodes, or links
    for k, v in config_model.get("nodes", {}).items():
        unrecognised_keys = [
            i for i in v.keys() if i not in DEFAULTS.nodes.default_node.keys()
        ]
        if len(unrecognised_keys) > 0:
            errors.append(
                "Node `{}` contains unrecognised keys {}. "
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
                    "Technology `{}` in node `{}` contains unrecognised keys {}; "
                    "these are most likely mispellings".format(
                        loc_tech_key, k, unrecognised_keys
                    )
                )

    default_link = DEFAULTS.links["default_node_from,default_node_to"]
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

    # We no longer allow cost_class in objective_options to be a string
    _cost_class = config_model.parameters.objective_cost_class

    if not isinstance(_cost_class.data, list):
        _data = [_cost_class.data]
    else:
        _data = _cost_class.data
    for _val in _data:
        if not isinstance(_val, (int, float)):
            errors.append(
                f"Objective cost class weights must be numeric, received:{_val}"
            )

    return model_warnings, errors


def _check_tech_final(
    model_run, tech_id, tech_config, loc_id, model_warnings, errors, comments
):
    """
    Checks individual tech/tech groups at specific nodes.
    NOTE: Updates `model_warnings` and `errors` lists in-place.
    """
    if tech_id not in model_run.techs:
        model_warnings.append(
            "Tech {} was removed by setting ``exists: False`` - not checking "
            "the consistency of its constraints at node {}.".format(tech_id, loc_id)
        )
        return model_warnings, errors

    required = model_run.techs[tech_id].required_constraints
    allowed = model_run.techs[tech_id].allowed_constraints
    allowed_costs = model_run.techs[tech_id].allowed_costs

    # Error if required constraints are not defined
    for r in required:
        # If it's a string, it must be defined
        single_ok = isinstance(r, str) and r in tech_config.get("constraints", {})
        # If it's a list of strings, one of them must be defined
        multiple_ok = isinstance(r, list) and any(
            [i in tech_config.get("constraints", {}) for i in r]
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
        for i in tech_config.get_key("constraints.carrier_ratios", AttrDict())
        .as_dict_flat()
        .keys()
    ]
    for carrier in carriers_in_ratios:
        if carrier not in defined_carriers:
            model_warnings.append(
                "Tech `{t}` gives a carrier ratio for `{c}`, but does not actually "
                "configure `{c}` as a carrier.".format(t=tech_id, c=carrier)
            )

    # If a technology is defined by units (i.e. integer decision variable), it must define flow_cap_per_unit
    if (
        any(["units_" in k for k in tech_config.get("constraints", {}).keys()])
        and "flow_cap_per_unit" not in tech_config.get("constraints", {}).keys()
    ):
        errors.append(
            f"`{tech_id}` at `{loc_id}` fails to define flow_cap_per_unit when "
            "specifying technology in units_max/min/equals"
        )

    # If a technology is defined by units & is a storage tech, it must define storage_cap_per_unit
    if (
        any(["units_" in k for k in tech_config.get("constraints", {}).keys()])
        and model_run.techs[tech_id].essentials.parent in ["storage", "supply_plus"]
        and any(["storage" in k for k in tech_config.get("constraints", {}).keys()])
        and "storage_cap_per_unit" not in tech_config.get("constraints", {}).keys()
    ):
        errors.append(
            f"`{tech_id}` at `{loc_id}` fails to define storage_cap_per_unit when "
            "specifying technology in units_max/min/equals"
        )

    # Gather remaining unallowed constraints
    remaining = set(tech_config.get("constraints", {})) - set(allowed)

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
    if "export_carrier" in tech_config.get("constraints", {}):
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

    # Error if non-allowed `source_unit` is defined
    if tech_config.switches.get("source_unit", "absolute") not in [
        "absolute",
        "per_cap",
        "per_area",
    ]:
        errors.append(
            "`{}` is an unknown source unit for `{}` at `{}`. "
            "Only `absolute`, `per_cap`, or `per_area` is allowed.".format(
                tech_config.switches.source_unit, tech_id, loc_id
            )
        )
    # Error if non-allowed `sink_unit` is defined
    if tech_config.switches.get("sink_unit", "absolute") not in [
        "absolute",
        "per_area",
    ]:
        errors.append(
            "`{}` is an unknown sink unit for `{}` at `{}`. "
            "Only `absolute`, or `per_area` is allowed.".format(
                tech_config.switches.sink_unit, tech_id, loc_id
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
    for loc_id, loc_config in model_run.nodes.items():
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

    # Either all nodes or no nodes must have coordinates
    all_nodes = list(model_run.nodes.keys())
    nodes_with_coords = [
        k for k in model_run.nodes.keys() if "coordinates" in model_run.nodes[k]
    ]
    if len(nodes_with_coords) != 0 and len(all_nodes) != len(nodes_with_coords):
        errors.append(
            "Either all or no nodes must have `coordinates` defined. "
            "nodes defined: {} - nodes with coordinates: {}".format(
                all_nodes, nodes_with_coords
            )
        )

    # If nodes have coordinates, they must all be either lat/lon or x/y
    elif len(nodes_with_coords) != 0:
        first_loc = list(model_run.nodes.keys())[0]
        try:
            coord_keys = sorted(list(model_run.nodes[first_loc].coordinates.keys()))
            if coord_keys != ["lat", "lon"] and coord_keys != ["x", "y"]:
                errors.append(
                    "Unidentified coordinate system. All nodes must either"
                    "use the format {lat: N, lon: M} or {x: N, y: M}."
                )
        except AttributeError:
            errors.append(
                "Coordinates must be given in the format {lat: N, lon: M} or "
                "{x: N, y: M}, not " + str(model_run.nodes[first_loc].coordinates)
            )

        for _loc_id, loc_config in model_run.nodes.items():
            try:
                if sorted(list(loc_config.coordinates.keys())) != coord_keys:
                    errors.append("All nodes must use the same coordinate format.")
                    break
            except AttributeError:
                errors.append(
                    "Coordinates must be given in the format {lat: N, lon: M} or "
                    "{x: N, y: M}, not " + str(model_run.nodes[first_loc].coordinates)
                )
                break

    # Ensure that timeseries have no non-unique index values
    for k, df in model_run["timeseries_data"].items():
        if df.index.duplicated().any():
            errors.append(
                "Time series `{}` contains non-unique timestamp values.".format(k)
            )

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
    if any(
        np.any(np.isinf(model_data.get(f"{x}_equals", False)))
        for x in ["source", "sink"]
    ):
        errors.append("Cannot include infinite values in `source_equals`/`sink_equals`")

    # Ensure that if a tech has negative costs, there is a max cap defined
    # FIXME: doesn't consider capacity being set by a linked constraint e.g.
    # `source_cap_per_flow_cap`.
    relevant_caps = set(
        [re.search(r"cost_(\w+_cap)", i) for i in model_data.data_vars.keys()]
    ).difference([None])
    for cap in relevant_caps:
        if (
            (model_data[cap.group(0)] < 0)
            & model_data.get(f"{cap.group(1)}_max", xr.DataArray(np.nan)).isnull()
            & model_data.get(f"{cap.group(1)}_equals", xr.DataArray(np.nan)).isnull()
        ).any():
            errors.append(
                f"Cannot have a negative {cap.group(0)} as there is an unset "
                "corresponding capacity constraint"
            )

    # TODO: fix operate mode by implementing windowsteps, etc., which should make this
    # issue of resolution changes redundant
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

    return model_data, comments, model_warnings, errors
