"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
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
from calliope.preprocess.util import split_loc_techs_transmission, concat_iterable
from calliope.preprocess.time import add_time_dimension
from calliope.preprocess.lookup import add_lookup_arrays


def build_model_data(model_run, debug=False):
    """
    Take a Calliope model_run and convert it into an xarray Dataset, ready for
    constraint generation. Timeseries data is also extracted from file at this
    point, and the time dimension added to the data

    Param_dict
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
    data = xr.Dataset(coords={'timesteps': model_run.timesteps}, attrs=add_attributes(model_run))

    param_dict = AttrDict({})
    get_node_params(param_dict, model_run)
    get_tech_params(param_dict, model_run)

    for k, v in param_dict.items():
        _df = pd.DataFrame.from_dict(v.as_dict())
        _df.columns = _df.columns.map(ast.literal_eval).rename(ast.literal_eval(k))
        _ds = xr.Dataset(_df.T).unstack()
        data = data.merge(_ds)

    if debug:
        data_pre_time = data.copy(deep=True)

    data = add_time_dimension(data, model_run)

    # Carrier information uses DataArray indexing in the function, so we merge
    # these directly into the main xarray Dataset

    if debug:
        return data, param_dict, data_pre_time
    else:
        return data


def add_sets(model_run):
    coords = dict()
    for key, value in model_run.sets.items():
        if value:
            coords[key] = value
    return coords


def get_node_params(param_dict, model_run):
    model_locations = model_run.locations.copy()
    for node, node_info in model_locations.items():
        techs = node_info.pop("techs", {})
        for tech, tech_info in techs.items():
            _get_tech_info(param_dict, node, tech, tech_info)
        links = node_info.pop("links", {})
        for link, link_info in links.items():
            techs = link_info.pop("techs", {})
            for tech, tech_info in techs.items():
                link_tech = f"{tech}:{link}"
                _get_tech_info(param_dict, node, link_tech, tech_info)
                param_dict.set_key(
                    _set_idx(
                        param="link_remotes",
                        keydict={"nodes": link, "techs": f"{tech}:{node}"},
                    ),
                    link_tech,
                )

        for node_param, node_param_info in node_info.items():
            if node_param == "coordinates":
                for k, v in node_param_info.items():
                    coord_key, coord = k, v
                    param_dict.set_key(
                        _set_idx(
                            param="node_coordinates",
                            keydict={"nodes": node, "coordinates": coord_key},
                        ),
                        coord,
                    )
            else:
                param_dict.set_key(
                    _set_idx(param=node_param, keydict={"nodes": node}), node_param_info
                )


def get_tech_params(param_dict, model_run):
    model_techs = model_run.techs.copy()
    for tech, tech_dict in model_techs.items():
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
                _set_tech_info(
                    param_dict, techs, "inheritance", ".".join(tech_param_info)
                )
            elif tech_param.startswith("essentials"):
                if "carrier" in tech_param and tech_param_info != "resource":
                    carrier_tier = tech_param.split(".")[-1].replace("carrier_", "")
                    for _tech in techs:
                        param_dict.set_key(
                            _set_idx(
                                param="carrier",
                                keydict={
                                    "techs": _tech,
                                    "carriers": tech_param_info,
                                    "carrier_tier": carrier_tier,
                                },
                            ),
                            1,
                        )
                else:
                    param_name = tech_param.split(".")[-1]
                    _set_tech_info(param_dict, techs, param_name, tech_param_info)



def _set_idx(param, keydict):
    return "{}.{}.{}".format(
        tuple(k for k in keydict.keys()), tuple(v for v in keydict.values()), param
    )


def _get_tech_info(param_dict, node, tech, tech_info):
    constraints = tech_info.pop("constraints", {})
    costs = tech_info.pop("costs", {})
    for constraint, constraint_info in constraints.items():
        if constraint == "carrier_ratios":
            for k, v in constraint_info.as_dict_flat().items():
                carrier_tier, carrier = k.split(".")
                if carrier == "resource":
                    print(tech)
                param_dict.set_key(
                    _set_idx(
                        param=constraint,
                        keydict={
                            "nodes": node,
                            "techs": tech,
                            "carriers": carrier,
                            "carrier_tier": carrier_tier.replace("carrier_", ""),
                        },
                    ),
                    v,
                )
        else:
            param_dict.set_key(
                _set_idx(param=constraint, keydict={"nodes": node, "techs": tech}),
                constraint_info,
            )
    for cost_class, cost_info in costs.items():
        for cost_param, cost in cost_info.items():
            param_dict.set_key(
                _set_idx(
                    param=f"cost_{cost_param}",
                    keydict={"nodes": node, "techs": tech, "costs": cost_class},
                ),
                cost,
            )

    for other_param, other_info in tech_info.items():
        param_dict.set_key(
            _set_idx(param=other_param, keydict={"nodes": node, "techs": tech}),
            other_info,
        )


def _set_tech_info(param_dict, techs, param_name, param_info):
    for tech in techs:
        param_dict.set_key(
            _set_idx(param=param_name, keydict={"techs": tech}), param_info
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
