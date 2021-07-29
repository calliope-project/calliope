"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

lookup.py
~~~~~~~~~~~~~~~~~~

Functionality to create DataArrays for looking up string values between loc_techs
and loc_tech_carriers, to avoid string operations during backend operations.

"""


import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

from calliope import exceptions


def add_lookup_arrays(data, model_run):
    """
    Take partially completed Calliope Model model_data and add lookup DataArrays
    to it.
    """
    data_dict = dict(
        lookup_loc_carriers=lookup_loc_carriers(model_run),
        lookup_loc_techs=lookup_loc_techs_non_conversion(model_run),
    )

    data = data.merge(xr.Dataset.from_dict(data_dict))

    if model_run.sets["loc_techs_conversion"]:
        data = lookup_loc_techs_conversion(data, model_run)

    if model_run.sets["loc_techs_conversion_plus"]:
        data = lookup_loc_techs_conversion_plus(data, model_run)

    if model_run.sets["loc_techs_export"]:
        data = lookup_loc_techs_export(data)

    if model_run.sets["loc_techs_area"]:
        data = lookup_loc_techs_area(data)

    return data


def lookup_loc_carriers(model_run):
    """
    loc_carriers, used in system_wide balance, are linked to loc_tech_carriers
    e.g. `X1::power` will be linked to `X1::chp::power` and `X1::battery::power`
    in a comma delimited string, e.g. `X1::chp::power,X1::battery::power`
    """
    # get the technologies associated with a certain loc_carrier
    lookup_loc_carriers_dict = dict(dims=["loc_carriers"])
    data = []
    for loc_carrier in model_run.sets["loc_carriers"]:
        loc_tech_carrier = list(
            set(
                i
                for i in model_run.sets["loc_tech_carriers_prod"]
                + model_run.sets["loc_tech_carriers_con"]
                if loc_carrier == "{0}::{2}".format(*i.split("::"))
            )
        )
        data.append(",".join(loc_tech_carrier))
    lookup_loc_carriers_dict["data"] = data

    return lookup_loc_carriers_dict


def lookup_loc_techs_non_conversion(model_run):
    """
    loc_techs be linked to their loc_tech_carriers, based on their carrier_in or
    carrier_out attribute. E.g. `X1::ccgt` will be linked to `X1::ccgt::power`
    as carrier_out for the ccgt is `power`.
    """
    lookup_loc_techs_dict = dict(dims=["loc_techs_non_conversion"])

    data = []
    for loc_tech in model_run.sets["loc_techs_non_conversion"]:
        # For any non-conversion technology, there is only one carrier (either
        # produced or consumed)
        loc_tech_carrier = list(
            set(
                i
                for i in model_run.sets["loc_tech_carriers_prod"]
                + model_run.sets["loc_tech_carriers_con"]
                if loc_tech == i.rsplit("::", 1)[0]
            )
        )
        if len(loc_tech_carrier) > 1:
            raise exceptions.ModelError(
                "More than one carrier associated with "
                "non-conversion location:technology `{}`".format(loc_tech)
            )
        else:
            data.append(loc_tech_carrier[0])
    lookup_loc_techs_dict["data"] = data

    return lookup_loc_techs_dict


def lookup_loc_techs_conversion(dataset, model_run):
    """
    Conversion technologies are seperated from other non-conversion technologies
    as there is more than one carrier associated with a single loc_tech. Here,
    the link is made per carrier tier (`out` and `in` are the two primary carrier
    tiers)
    """
    # Get the string name for a loc_tech which includes the carriers in and out
    # associated with that technology (for conversion technologies)
    carrier_tiers = model_run.sets["carrier_tiers"]

    loc_techs_conversion_array = xr.DataArray(
        data=np.empty(
            (len(model_run.sets["loc_techs_conversion"]), len(carrier_tiers)),
            dtype=object,
        ),
        dims=["loc_techs_conversion", "carrier_tiers"],
        coords={
            "loc_techs_conversion": list(model_run.sets["loc_techs_conversion"]),
            "carrier_tiers": list(carrier_tiers),
        },
    )
    for loc_tech in model_run.sets["loc_techs_conversion"]:
        # For any non-conversion technology, there are only two carriers
        # (one produced and one consumed)
        loc_tech_carrier_in = [
            i
            for i in model_run.sets["loc_tech_carriers_con"]
            if loc_tech == i.rsplit("::", 1)[0]
        ]

        loc_tech_carrier_out = [
            i
            for i in model_run.sets["loc_tech_carriers_prod"]
            if loc_tech == i.rsplit("::", 1)[0]
        ]
        if len(loc_tech_carrier_in) > 1 or len(loc_tech_carrier_out) > 1:
            raise exceptions.ModelError(
                "More than one carrier in or out associated with "
                "conversion location:technology `{}`".format(loc_tech)
            )
        else:
            loc_techs_conversion_array.loc[
                dict(loc_techs_conversion=loc_tech, carrier_tiers=["in", "out"])
            ] = [loc_tech_carrier_in[0], loc_tech_carrier_out[0]]

    dataset = dataset.merge(
        loc_techs_conversion_array.to_dataset(name="lookup_loc_techs_conversion")
    )

    return dataset


def lookup_loc_techs_conversion_plus(dataset, model_run):
    """
    Conversion plus technologies are seperated from other technologies
    as there is more than one carrier associated with a single loc_tech. Here,
    the link is made per carrier tier (`out`, `in`, `out_2`, `in_2`, `out_3`,
    `in_3` are the possible carrier tiers). Multiple carriers may be associated
    with a single loc_tech tier, so a comma delimited string will be created.
    """
    # Get the string name for a loc_tech which includes all the carriers in
    # and out associated with that technology (for conversion_plus technologies)
    carrier_tiers = model_run.sets["carrier_tiers"]
    loc_techs_conversion_plus = model_run.sets["loc_techs_conversion_plus"]

    loc_techs_conversion_plus_array = np.empty(
        (len(loc_techs_conversion_plus), len(carrier_tiers)), dtype=object
    )
    primary_carrier_data = {"_in": [], "_out": []}
    for loc_tech_idx, loc_tech in enumerate(loc_techs_conversion_plus):
        _tech = loc_tech.split("::", 1)[1]
        for k, v in primary_carrier_data.items():
            primary_carrier = model_run.techs[_tech].essentials.get(
                "primary_carrier" + k, ""
            )
            v.append(loc_tech + "::" + primary_carrier)
        for carrier_tier_idx, carrier_tier in enumerate(carrier_tiers):
            # create a list of carriers for the given technology that fits
            # the current carrier_tier.
            relevant_carriers = model_run.techs[_tech].essentials.get(
                "carrier_" + carrier_tier, None
            )

            if relevant_carriers and isinstance(relevant_carriers, list):
                loc_tech_carriers = ",".join(
                    [loc_tech + "::" + i for i in relevant_carriers]
                )
            elif relevant_carriers:
                loc_tech_carriers = loc_tech + "::" + relevant_carriers
            else:
                continue
            loc_techs_conversion_plus_array[
                loc_tech_idx, carrier_tier_idx
            ] = loc_tech_carriers
    for k, v in primary_carrier_data.items():
        primary_carrier_data_array = xr.DataArray.from_dict(
            {"data": v, "dims": ["loc_techs_conversion_plus"]}
        )
        dataset["lookup_primary_loc_tech_carriers" + k] = primary_carrier_data_array

    dataset["lookup_loc_techs_conversion_plus"] = xr.DataArray(
        data=loc_techs_conversion_plus_array,
        dims=["loc_techs_conversion_plus", "carrier_tiers"],
        coords={
            "loc_techs_conversion_plus": loc_techs_conversion_plus,
            "carrier_tiers": carrier_tiers,
        },
    )

    return dataset


def lookup_loc_techs_export(dataset):
    """
    For a given loc_tech, return loc_tech_carrier where the carrier is the export
    carrier of that loc_tech
    """
    data_dict = dict(dims=["loc_techs_export"], data=[])

    for i in dataset.export_carrier:
        data_dict["data"].append(i.loc_techs_export.item() + "::" + i.item())

    dataset["lookup_loc_techs_export"] = xr.DataArray.from_dict(data_dict)

    return dataset


def lookup_loc_techs_area(dataset):
    """
    For a given loc, return loc_techs where the tech is any technology at that
    location which defines a resource_area, if it isn't a demand technology.
    If there are multiple loc_techs, the result is a comma delimited string of
    loc_techs
    """
    data_dict = dict(dims=["locs"], data=[])

    for loc in dataset.locs:
        relevant_loc_techs = [
            loc_tech
            for loc_tech in dataset.loc_techs_area.values
            if loc == loc_tech.split("::")[0]
            and loc_tech not in dataset.loc_techs_demand.values
        ]
        data_dict["data"].append(",".join(relevant_loc_techs))

    dataset["lookup_loc_techs_area"] = xr.DataArray.from_dict(data_dict)

    return dataset


def lookup_clusters(dataset):
    """
    For any given timestep in a time clustered model, get:
    1. the first and last timestep of the cluster,
    2. the last timestep of the cluster corresponding to a date in the original timeseries
    """

    data_dict_first = dict(dims=["timesteps"], data=[])
    data_dict_last = dict(dims=["timesteps"], data=[])
    for timestep in dataset.timesteps:
        t = pd.to_datetime(timestep.item()).date().strftime("%Y-%m-%d")
        timestep_first = dataset.timesteps.loc[t][0]
        timestep_last = dataset.timesteps.loc[t][-1]
        if timestep == timestep_first:
            data_dict_first["data"].append(1)
            data_dict_last["data"].append(timestep_last.values)
        else:
            data_dict_first["data"].append(0)
            data_dict_last["data"].append(None)
    dataset["lookup_cluster_first_timestep"] = xr.DataArray.from_dict(data_dict_first)
    dataset["lookup_cluster_last_timestep"] = xr.DataArray.from_dict(data_dict_last)

    if "datesteps" in dataset.dims:
        last_timesteps = dict(dims=["datesteps"], data=[])
        cluster_date = dataset.timestep_cluster.to_pandas().resample("1D").mean()
        for datestep in dataset.datesteps.to_index():
            cluster = dataset.lookup_datestep_cluster.loc[
                datestep.strftime("%Y-%m-%d")
            ].item()
            last_timesteps["data"].append(
                datetime.combine(
                    cluster_date[cluster_date == cluster].index[0].date(),
                    dataset.timesteps.to_index().time[-1],
                )
            )
        dataset["lookup_datestep_last_cluster_timestep"] = xr.DataArray.from_dict(
            last_timesteps
        )

    return dataset
