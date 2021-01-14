"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model_data.py
~~~~~~~~~~~~~~~~~~

Functionality to build the model-internal data array and process
time-varying param_dict.

"""
import ast
import collections

import ruamel.yaml
import xarray as xr
import numpy as np
import pandas as pd

from calliope.core.attrdict import AttrDict
from calliope._version import __version__
from calliope.preprocess import checks
from calliope.preprocess.time import add_time_dimension, update_dtypes


def build_model_data(model_run, debug=False):
    """
    Take a Calliope model_run and convert it into an xarray Dataset, ready for
    constraint generation. Timeseries data is also extracted from file at this
    point, and the time dimension added to the data

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
        Dataset with optimisation param_dict as variables, optimisation sets as
        coordinates, and other information in attributes.
    data_dict : dict, only returned if debug = True
        dictionary of param_dict, prior to time dimension addition. Used here to
        populate the Dataset (using `from_dict()`)
    data_pre_time : xarray Dataset, only returned if debug = True
        Dataset, prior to time dimension addition, with optimisation param_dict
        as variables, optimisation sets as coordinates, and other information
        in attributes.
    """
    # We build up a dictionary of the data, then convert it to an xarray Dataset
    # before applying time dimensions
    data = xr.Dataset(
        coords={"timesteps": model_run.timeseries_data.index},
        attrs=add_attributes(model_run),
    )

    # param_dict is going to be of the form (*dim_names): {(*dims): {**relevant_parms}}
    # E.g. "('nodes', 'techs')": {"('N1', 'heat_pipes:X1')": {{'energy_cap_max': 1, 'energy_eff': 0.99, etc.}}
    param_dict = AttrDict({})
    # Get parameters that are node-specific
    get_node_params(param_dict, model_run)
    # Get parameters that are tech-specific, and node-invariant
    get_tech_params(param_dict, model_run)

    for set_names, data_over_sets in param_dict.items():
        _df = pd.DataFrame.from_dict(data_over_sets.as_dict())
        # Column values go from e.g. "('N1', 'heat_pipes:X1')" to ('N1', 'heat_pipes:X1')
        # Column name goes from e.g. "('nodes', 'techs')" to ('nodes', 'techs')
        _df.columns = _df.columns.map(ast.literal_eval).rename(
            ast.literal_eval(set_names)
        )
        _ds = xr.Dataset(_df.T).unstack()
        data = data.merge(_ds)

    for param_data in data.data_vars.values():
        param_data.attrs["parameters"] = 1

    if debug:
        data_pre_time = data.copy(deep=True)
    # Remove techs not assigned to nodes, nodes with no associated techs, and carriers associated with removed techs
    for dim in ["nodes", "techs"]:
        data = data.dropna(dim, how="all", subset=["node_tech"])
    for dim in ["carriers", "carrier_tiers"]:
        data = data.dropna(dim, how="all")
    data = data.drop_vars(
        [var_name for var_name, var in data.data_vars.items() if var.isnull().all()]
    )

    data = add_time_dimension(data, model_run)

    data = update_dtypes(data)
    # Carrier information uses DataArray indexing in the function, so we merge
    # these directly into the main xarray Dataset

    if debug:
        return data, param_dict, data_pre_time
    else:
        return data


def get_node_params(param_dict, model_run):
    """
    For all nodes, get tech and link data
    """
    model_locations = (
        model_run.locations.copy()
    )  # TODO: "locations" -> "nodes" in YAMLs
    for node, node_info in model_locations.items():
        # Techs in node, pop them out
        techs = node_info.pop("techs", {})
        for tech, tech_info in techs.items():
            if tech_info is not None:
                set_tech_at_node_info(param_dict, node, tech, tech_info)
        # Links in node
        links = node_info.pop("links", {})
        for link, link_info in links.items():
            techs = link_info.pop("techs", {})
            for tech, tech_info in techs.items():
                link_tech = f"{tech}:{link}"
                if tech_info is not None:
                    set_tech_at_node_info(param_dict, node, link_tech, tech_info)
                    param_dict.set_key(
                        set_idx(
                            param="link_remote_techs",
                            keydict={"nodes": link, "techs": f"{tech}:{node}"},
                        ),
                        link_tech,
                    )
                    param_dict.set_key(
                        set_idx(
                            param="link_remote_nodes",
                            keydict={"nodes": link, "techs": f"{tech}:{node}"},
                        ),
                        node,
                    )
        # node info (e.g. coordinates)
        for node_param, node_param_info in node_info.items():
            if node_param == "coordinates":
                for k, v in node_param_info.items():
                    coord_key, coord = k, v
                    param_dict.set_key(
                        set_idx(
                            param="node_coordinates",
                            keydict={"nodes": node, "coordinates": coord_key},
                        ),
                        coord,
                    )
            else:
                param_dict.set_key(
                    set_idx(param=node_param, keydict={"nodes": node}), node_param_info
                )


def get_tech_params(param_dict, model_run):
    """
    For a given tech, get all 'essentials' information
    """
    model_techs = model_run.techs.copy()
    for tech, tech_dict in model_techs.items():
        # Transmission techs are referred to as "tech_name:node_name", meaning there are
        # now several of them, which we'll need to iterate through
        if tech_dict.inheritance[-1] == "transmission":
            techs = set(
                [
                    "{}:{}".format(i.split(".")[4], i.split(".")[2])
                    for i in model_run.locations.as_dict_flat().keys()
                    if tech in i
                ]
            )
        else:
            techs = [tech]
        for tech_param, tech_param_info in tech_dict.as_dict_flat().items():
            if tech_param.startswith("inheritance"):
                set_tech_info(
                    param_dict, techs, "inheritance", ".".join(tech_param_info)
                )
            elif tech_param.startswith("essentials"):
                if "carrier" in tech_param:
                    if "primary" in tech_param:
                        for _tech in techs:
                            param_dict.set_key(
                                set_idx(
                                    param=tech_param.split(".")[-1],
                                    keydict={
                                        "techs": _tech,
                                        "carriers": tech_param_info,
                                    },
                                ),
                                1,
                            )
                    else:
                        carrier_tier = tech_param.split(".")[-1].replace("carrier_", "")
                        if tech_param_info == "resource" or carrier_tier == "carrier":
                            continue
                        for _tech in techs:
                            if not isinstance(tech_param_info, list):
                                tech_param_info = [tech_param_info]
                            for _tech_param_info in tech_param_info:
                                param_dict.set_key(
                                    set_idx(
                                        param="carrier",
                                        keydict={
                                            "techs": _tech,
                                            "carriers": _tech_param_info,
                                            "carrier_tiers": carrier_tier,
                                        },
                                    ),
                                    1,
                                )
                else:
                    param_name = tech_param.split(".")[-1]
                    set_tech_info(param_dict, techs, param_name, tech_param_info)
            elif tech_param.startswith("constraints") and "systemwide" in tech_param:
                param_name = tech_param.split(".")[-1]
                set_tech_info(param_dict, techs, param_name, tech_param_info)


def set_idx(param, keydict):
    """
    Set a key:value pair in the re-processed model data dictionary

    Parameters
    ----------
    param : string
        name of parameter being set
    keydict : Dictionary
        Dimensions being set for that parameter

    Returns
    -------
    AttrDict key : string
        e,g, "(tech, node).(pv, X1).energy_cap_max"

    """
    return "{}.{}.{}".format(
        tuple(k for k in keydict.keys()), tuple(v for v in keydict.values()), param
    )


def set_tech_at_node_info(param_dict, node, tech, tech_info):
    """
    Get all data from 'constraints', 'costs', and 'switches' for a tech at a node.

    Parameters
    ----------
    param_dict : Dictionary
        Dictionary of re-processed model data
    node : str
        Name of node
    tech : str
        Name of tech
    node : str
        Name of node
    tech_info : Dictionary
        Dictionary of tech parameters at a particular node
    """
    constraints = tech_info.pop("constraints", {})
    costs = tech_info.pop("costs", {})
    switches = tech_info.pop("switches", {})
    for constraint, constraint_info in constraints.items():
        # carrier-based data is reformatted to be indexed by carrier
        # FIXME: this should not be hardcoded...
        if constraint == "carrier_ratios":
            for k, v in constraint_info.as_dict_flat().items():
                carrier_tier, carrier = k.split(".")
                if carrier == "resource":
                    continue
                param_dict.set_key(
                    set_idx(
                        param=constraint,
                        keydict={
                            "nodes": node,
                            "techs": tech,
                            "carriers": carrier,
                            "carrier_tiers": carrier_tier.replace("carrier_", ""),
                        },
                    ),
                    v,
                )
        elif constraint == "export_carrier":
            param_dict.set_key(
                set_idx(
                    param=constraint,
                    keydict={
                        "nodes": node,
                        "techs": tech,
                        "carriers": constraint_info,
                    },
                ),
                1,
            )
        else:
            param_dict.set_key(
                set_idx(param=constraint, keydict={"nodes": node, "techs": tech}),
                constraint_info,
            )
    for cost_class, cost_info in costs.items():
        for cost_param, cost in cost_info.items():
            param_dict.set_key(
                set_idx(
                    param=f"cost_{cost_param}",
                    keydict={"nodes": node, "techs": tech, "costs": cost_class},
                ),
                cost,
            )
    for switch, switch_info in switches.items():
        param_dict.set_key(
            set_idx(param=switch, keydict={"nodes": node, "techs": tech}), switch_info,
        )

    for other_param, other_info in tech_info.items():
        param_dict.set_key(
            set_idx(param=other_param, keydict={"nodes": node, "techs": tech}),
            other_info,
        )
    # Also set the parameter that the tech exists at this node
    param_dict.set_key(
        set_idx(param="node_tech", keydict={"nodes": node, "techs": tech}), 1,
    )


def set_tech_info(param_dict, techs, param_name, param_info):
    """
    Set a key:value pair for a tech in the re-processed model data dictionary

    Parameters
    ----------
    param_dict : Dictionary
        Dictionary of re-processed model data
    techs : list
        tech names (mostly a single length list, except for transmission techs)
    param_name : str
        Name of parameter being set
    node : str
        Name of node
    tech_info : Dictionary
        Dictionary of tech parameters at a particular node
    """
    for tech in techs:
        param_dict.set_key(
            set_idx(param=param_name, keydict={"techs": tech}), param_info
        )


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

    attr_dict["defaults"] = ruamel.yaml.dump(
        {
            **default_tech_dict["constraints"],
            **{
                "cost_{}".format(k): v
                for k, v in default_tech_dict["costs"]["default_cost"].items()
            },
            **default_location_dict,
        }
    )

    return attr_dict
