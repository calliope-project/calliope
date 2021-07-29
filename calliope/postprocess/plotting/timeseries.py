"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

timeseries.py
~~~~~~~~~~~~~

Plot timeseries data.

"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from calliope.postprocess.util import subset_sum_squeeze
from calliope.postprocess.plotting.util import (
    get_data_layout,
    hex_to_rgba,
    break_name,
    get_clustered_layout,
)


def _get_relevant_vars(model, dataset, array):

    carriers = list(dataset.carriers.values)

    allowed_input_vars = [
        k
        for k, v in model.inputs.data_vars.items()
        if "timesteps" in v.dims and len(v.dims) > 1 and k != "carrier_ratios"
    ]
    allowed_result_vars = [
        "results",
        "inputs",
        "all",
        "storage",
        "storage_inter_cluster",
        "resource_con",
        "cost_var",
    ]

    if (
        isinstance(array, list)
        and not set(array).intersection(
            allowed_input_vars + allowed_result_vars + carriers
        )
    ) or (
        isinstance(array, str)
        and array not in allowed_input_vars + allowed_result_vars + carriers
    ):
        raise ValueError(
            "Cannot plot array={}. If you want carrier flow (_prod, _con, _export) "
            "then specify the name of the energy carrier as array".format(array)
        )

    # relevant_vars are all variables relevant to this plotting instance
    relevant_vars = []

    # Ensure carriers are at the top of the list
    if array == "results":
        relevant_vars += sorted(carriers) + sorted(allowed_result_vars)
    elif array == "inputs" or (array == "all" and not hasattr(model, "results")):
        relevant_vars += sorted(allowed_input_vars)
    elif array == "all":
        relevant_vars += sorted(carriers) + sorted(
            allowed_result_vars + allowed_input_vars
        )
    elif isinstance(array, list):
        relevant_vars = array
    elif isinstance(array, str):
        relevant_vars = [array]

    relevant_vars = [i for i in relevant_vars if i in dataset or i in carriers]
    return relevant_vars


def _get_var_data(var, model, dataset, visible, subset, sum_dims, squeeze):
    """
    Get variable data from model_data and use it to populate a list with Plotly plots
    """
    carriers = list(dataset.carriers.values)

    # list to populate
    data = []

    # Clustered data does not get plotted on a datetime axis
    if "clusters" in dataset.dims:
        clusters = dataset.timestep_cluster.to_pandas()
        cluster_groups = clusters.groupby(clusters).groups
        # Add an extra timestep at the end of each cluster, so we can break
        # scatter lines with a NaN between each cluster
        dummy_timestep = lambda x: pd.to_datetime(x[-1] + pd.Timedelta(100, unit="ns"))
        reindexing_timesteps = np.concatenate(
            [
                group_timesteps.append(
                    pd.Index([dummy_timestep(group_timesteps)])
                ).values
                for group_timesteps in cluster_groups.values()
            ]
        )
        timesteps = pd.Index(reindexing_timesteps).values.astype(str)
        # replace dummy timestep with NaN
        timesteps[pd.Index(reindexing_timesteps).nanosecond == 100] = np.nan
        timesteps = pd.to_datetime(timesteps, errors="ignore")
        if var == "storage_inter_cluster":
            datesteps = pd.to_datetime(model._model_data.datesteps.values)
    else:
        timesteps = pd.to_datetime(model._model_data.timesteps.values)

    def _get_reindexed_array(array, index=["locs", "techs"], fillna=None):
        # reindexing data means that DataArrays have the same values in locs and techs
        reindexer = {k: sorted(dataset[k].values) for k in index}
        formatted_array = model.get_formatted_array(array)
        if fillna is not None:
            return formatted_array.reindex(**reindexer).fillna(fillna)
        else:
            return formatted_array.reindex(**reindexer)

    def _get_y_vals(array, ignore_clustering=False):
        if "clusters" in dataset.dims and not ignore_clustering:
            return array.to_pandas().reindex(reindexing_timesteps).values
        else:
            return array.values

    if hasattr(model, "results") and len(model.results.data_vars) > 0:
        array_prod = _get_reindexed_array(
            "carrier_prod", index=["locs", "techs", "carriers"], fillna=0
        )
        array_con = _get_reindexed_array(
            "carrier_con", index=["locs", "techs", "carriers"], fillna=0
        )
        if "resource_con" in dataset.data_vars:
            resource_con = _get_reindexed_array("resource_con", fillna=0)
        else:
            resource_con = 0

    # carrier flow is a combination of carrier_prod, carrier_con and
    # carrier_export for a given energy carrier
    if var in carriers:
        array_flow = (
            array_prod.loc[dict(carriers=var)] + array_con.loc[dict(carriers=var)]
        )
        if "carrier_export" in dataset:
            export_flow = subset_sum_squeeze(
                _get_reindexed_array(
                    "carrier_export", index=["locs", "techs", "carriers"], fillna=0
                ).loc[dict(carriers=var)],
                subset,
                sum_dims,
                squeeze,
            )
        if "unmet_demand" in dataset:
            unmet_flow = subset_sum_squeeze(
                _get_reindexed_array(
                    "unmet_demand", index=["locs", "carriers"], fillna=0
                ).loc[dict(carriers=var)],
                subset,
                sum_dims,
                squeeze=False,
            )

    # array flow for storage tracks stored energy. carrier_flow is
    # charge/discharge (including resource consumed for supply_plus techs)
    elif var == "storage":
        array_flow = _get_reindexed_array("storage")

        if "resource_eff" in dataset.data_vars:
            resource_eff = _get_reindexed_array("resource_eff", fillna=1)
        else:
            resource_eff = 1

        charge = subset_sum_squeeze(
            -array_con.sum("carriers") + resource_con * resource_eff,
            subset,
            sum_dims,
            squeeze=False,
        )
        discharge = -subset_sum_squeeze(
            array_prod.sum("carriers"), subset, sum_dims, squeeze=False
        )

    elif var == "resource_con":
        array_flow = resource_con

    else:
        array_flow = _get_reindexed_array(var)

    array_flow = subset_sum_squeeze(array_flow, subset, sum_dims, squeeze)

    err_details = " (variable: `{}`, subset: `{}`, sum_dims: `{}`.".format(
        var, subset, sum_dims
    )

    if "timesteps" not in array_flow.dims and "datesteps" not in array_flow.dims:
        raise ValueError(
            "`timesteps` not in plotting data, cannot proceed" + err_details
        )

    if len(array_flow.dims) > 2:
        if "costs" in array_flow.dims:
            err_details = err_details + (
                " Try subsetting to select a cost class, "
                "e.g. subset={'costs': ['monetary']}."
            )
        raise ValueError(
            "Too many dimensions to plot: `{}`".format(array_flow.dims) + err_details
        )

    # If techs not given as a subset (implying a desire to order manually),
    # sort techs such that those varying output least are at the bottom of the stack
    if subset.get("techs", None) or var == "storage_inter_cluster":
        sorted_techs = array_flow.techs.values
    else:
        sorted_techs = [
            array_flow.techs.values[i]
            for i in array_flow.var(dim="timesteps").argsort()
        ]

    # for tech in array_flow.techs.values:
    for tech in sorted_techs:
        tech_dict = {"techs": tech}
        if not array_flow.loc[tech_dict].sum():
            continue
        # We allow transmission tech information to show up in some cases
        if (
            "techs_transmission" in dataset
            and tech in dataset.techs_transmission.values
        ):
            base_tech = "transmission"
            color = dataset.colors.loc[{"techs": tech.split(":")[0]}].item()
            name = break_name(
                dataset.names.loc[{"techs": tech.split(":")[0]}].item(), 30
            )
            if var in carriers:
                continue  # no transmission in carrier flow
        else:
            base_tech = dataset.inheritance.loc[tech_dict].item().split(".")[0]
            color = dataset.colors.loc[tech_dict].item()
            name = break_name(dataset.names.loc[tech_dict].item(), 30)

        if base_tech == "demand":
            # Always insert demand at position 0 in the list, to make
            # sure it appears on top in the legend
            data.insert(
                0,
                go.Scatter(
                    x=timesteps,
                    y=_get_y_vals(-array_flow.loc[tech_dict]),
                    visible=visible,
                    line=dict(color=color),
                    name=name,
                    legendgroup=tech,
                ),
            )

        elif var == "storage":
            # stored energy as scatter, carrier/resource prod/con as stacked bar
            data.insert(
                0,
                go.Scatter(
                    x=timesteps,
                    y=_get_y_vals(array_flow.loc[tech_dict]),
                    visible=visible,
                    line=dict(color=color),
                    mode="lines",
                    name=name + " stored energy",
                    showlegend=False,
                    text=tech + " stored energy",
                    hoverinfo="x+y+text",
                    legendgroup=tech,
                ),
            )
            data.append(
                go.Bar(
                    x=timesteps,
                    y=_get_y_vals(charge.loc[tech_dict]),
                    visible=visible,
                    name=name,
                    marker=dict(color=color),
                    legendgroup=tech,
                    text=tech + " charge",
                    hoverinfo="x+y+text",
                )
            )
            data.append(
                go.Bar(
                    x=timesteps,
                    y=_get_y_vals(discharge.loc[tech_dict]),
                    visible=visible,
                    name=name,
                    marker=dict(color=color, opacity=0.8),
                    legendgroup=tech,
                    text=tech + " discharge",
                    hoverinfo="x+y+text",
                    showlegend=False,
                )
            )

        elif var == "storage_inter_cluster":
            data.append(
                go.Scatter(
                    x=datesteps,
                    y=_get_y_vals(array_flow.loc[tech_dict], ignore_clustering=True),
                    visible=visible,
                    line=dict(color=color),
                    mode="lines",
                    name=name,
                    showlegend=False,
                    text=tech,
                    hoverinfo="x+y+text",
                    legendgroup=tech,
                )
            )

        else:
            data.append(
                go.Bar(
                    x=timesteps,
                    y=_get_y_vals(array_flow.loc[tech_dict]),
                    visible=visible,
                    name=name,
                    legendgroup=tech,
                    marker=dict(color=color),
                )
            )

        if (
            var in carriers
            and "carrier_export" in dataset
            and export_flow.loc[tech_dict].sum()
        ):
            data.append(
                go.Bar(
                    x=timesteps,
                    y=_get_y_vals(-export_flow.loc[tech_dict]),
                    visible=visible,
                    name=name + " export",
                    legendgroup=tech,
                    marker=dict(color=hex_to_rgba(color, 0.5)),
                )
            )

    if var in carriers and "unmet_demand" in dataset:
        data.append(
            go.Bar(
                x=timesteps,
                y=_get_y_vals(unmet_flow),
                visible=visible,
                name="Unmet " + var + " demand",
                legendgroup="unmet_demand",
                marker=dict(color="grey"),
            )
        )

    return data


def _get_var_layout(var, dataset):
    """
    Variable-specific layout. Increases axis verbosity for some known variables.
    `visible` used in dropdown, not if only one array is shown.

    """
    args = {}
    if var in dataset.carriers.values:
        title = "Carrier flow: {}".format(var)
        y_axis_title = "Energy produced(+) / consumed(-)"
    elif var == "resource":
        title = "Available resource"
        y_axis_title = "Energy (per unit of area)"
    elif var == "resource_con":
        title = "Consumed resource"
        y_axis_title = "Energy"
    elif var == "cost_var":
        title = "Variable costs"
        y_axis_title = "Cost"
    elif var == "storage_inter_cluster":
        title = "Storage at start of each day in unclustered timeseries"
        y_axis_title = "Stored energy"
        args.update({"xaxis": {"type": "date"}})
    else:
        title = y_axis_title = "{}".format(var).capitalize()

    if "clusters" in dataset.dims and var != "storage_inter_cluster":
        args.update(get_clustered_layout(dataset))

    args.update({"yaxis": dict(title=y_axis_title), "title": title})

    return args


def plot_timeseries(
    model,
    array="all",
    timesteps_zoom=None,
    rangeslider=False,
    subset={},
    sum_dims="locs",
    squeeze=True,
    **kwargs,
):
    """
    Parameters
    ----------
    array : str or list; default = 'all'
        options: 'all', 'results', 'inputs', the name/list of any energy carrier(s)
        (e.g. 'power'), the name/list of any input/output DataArray(s).

        User can specify 'all' for all input/results timeseries plots, 'inputs'
        for just input timeseries, 'results' for just results timeseries, or the
        name of any data array to plot (in either inputs or results).
        In all but the last case, arrays can be picked from dropdown in visualisation.
        In the last case, output can be saved to SVG and a rangeslider can be used.

    timesteps_zoom : int, optional
        Number of timesteps to show initially on the x-axis (if not
        given, the full time range is shown by default).
    rangeslider : bool, optional
        If True, displays a range slider underneath the plot for navigating
        (helpful primarily in interactive use).
    subset : dict, optional
        Dictionary by which data is subset (uses xarray `loc` indexing). Keys
        any of ['timeseries', 'locs', 'techs', 'carriers', 'costs'].
    sum_dims : str, optional
        List of dimension names to sum plot variable over.
    squeeze : bool, optional
        Whether to squeeze out dimensions of length = 1.

    """

    dataset = model._model_data.copy()

    timesteps = pd.to_datetime(model._model_data.timesteps.values)

    layout = dict(
        barmode="relative",
        xaxis={},
        autosize=True,
        legend=(dict(traceorder="reversed", xanchor="left")),
        hovermode="x",
    )
    # Clustered data does not get plotted on a datetime axis
    if "clusters" in dataset.dims:
        layout.update(get_clustered_layout(dataset))

    relevant_vars = _get_relevant_vars(model, dataset, array)

    data, layout = get_data_layout(
        _get_var_data,
        _get_var_layout,
        relevant_vars,
        layout,
        model,
        dataset,
        subset,
        sum_dims,
        squeeze,
    )

    if rangeslider:
        layout["xaxis"]["rangeslider"] = {}

    if timesteps_zoom:
        layout["xaxis"]["range"] = [timesteps[0], timesteps[timesteps_zoom]]

    return data, layout
