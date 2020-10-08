"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

capacity.py
~~~~~~~~~~~

Plot capacity data.

"""

from itertools import product

import plotly.graph_objs as go
from natsort import natsorted

from calliope.postprocess.util import subset_sum_squeeze
from calliope.postprocess.plotting.util import get_data_layout, break_name


def _get_relevant_vars(dataset, array):
    allowed_input_vars = [
        i + j
        for i, j in product(
            ["resource_area", "energy_cap", "resource_cap", "storage_cap", "units"],
            ["_max", "_min", "_equals"],
        )
    ]
    allowed_result_vars = [
        "results",
        "inputs",
        "all",
        "resource_area",
        "energy_cap",
        "resource_cap",
        "storage_cap",
        "units",
        "systemwide_levelised_cost",
        "systemwide_capacity_factor",
    ]

    if (
        isinstance(array, list)
        and not set(array) != set(allowed_input_vars + allowed_result_vars)
    ) or (
        isinstance(array, str) and array not in allowed_input_vars + allowed_result_vars
    ):
        raise ValueError(
            "Cannot plot array={}. as one or more of the elements is not considered "
            "to be a capacity".format(array)
        )

    # relevant_vars are all variables relevant to this plotting instance
    if array == "results":
        relevant_vars = natsorted(allowed_result_vars)
    elif array == "inputs":
        relevant_vars = natsorted(allowed_input_vars)
    elif array == "all":
        relevant_vars = natsorted(allowed_result_vars + allowed_input_vars)
    elif isinstance(array, list):
        relevant_vars = array
    elif isinstance(array, str):
        relevant_vars = [array]

    relevant_vars = [i for i in relevant_vars if i in dataset]

    # Remove all vars that don't actually turn up in the dataset, which is relevant
    # ony really for results vars
    return sorted(list(set(relevant_vars).intersection(dataset.data_vars.keys())))


def _get_var_data(
    cap, model, dataset, visible, subset, sum_dims, squeeze, locations, orientation
):
    if "systemwide" in cap:
        array_cap = subset_sum_squeeze(dataset[cap], subset)
        if "costs" in array_cap.dims and len(array_cap["costs"]) == 1:
            array_cap = array_cap.squeeze("costs")
        elif "costs" in array_cap.dims and len(array_cap["costs"]) > 1:
            raise ValueError(
                "Cannot plot {} without subsetting to pick one cost type "
                "of interest".format(cap)
            )
        if "carriers" not in subset.keys():
            array_cap = array_cap.sortby("carriers")

    else:
        array_cap = model.get_formatted_array(cap).reindex(locs=locations)
        array_cap = subset_sum_squeeze(array_cap, subset, sum_dims, squeeze)

    if len(array_cap.dims) > 2:
        raise ValueError(
            "Maximum two dimensions allowed for plotting capacity, but {} "
            "given as dimensions for {}".format(array_cap.dims, cap)
        )

    if "techs" not in array_cap.dims:
        raise ValueError("Cannot plot capacity without `techs` in dimensions")

    elif "techs" not in subset.keys():
        array_cap = array_cap.sortby("techs")

    data = []

    for tech in array_cap.techs.values:
        if tech not in dataset.techs.values:
            continue

        if (
            "techs_transmission" in dataset
            and tech in dataset.techs_transmission.values
        ):
            continue
        else:
            base_tech = dataset.inheritance.loc[{"techs": tech}].item().split(".")[0]

        if base_tech in "demand":
            continue

        if array_cap.loc[{"techs": tech}].sum() > 0:
            x = array_cap.loc[{"techs": tech}].values
            name = break_name(model._model_data.names.loc[{"techs": tech}].item(), 30)
            if "systemwide" in cap:
                y = natsorted(array_cap.carriers.values)
            else:
                if "locs" in array_cap.dims:
                    y = natsorted(array_cap.locs.values)
                else:  # Single location
                    y = [array_cap.locs.values]

            if orientation == "v":
                x, y = y, x  # Flip axes
                hoverinfo = "y+name"
            else:
                hoverinfo = "x+name"
                x, y = x[::-1], y[::-1]  # Make sure that sorting is from bottom down

            data.append(
                go.Bar(
                    x=x,
                    y=y,
                    visible=visible,
                    name=name,
                    legendgroup=tech,
                    text=tech,
                    hoverinfo=hoverinfo,
                    marker=dict(
                        color=model._model_data.colors.loc[{"techs": tech}].item()
                    ),
                    orientation=orientation,
                )
            )

    return data


def _get_var_layout(cap, dataset, location_axis, value_axis):
    args = {}
    if "area" in cap:
        value_axis_title = "Installed area"
    elif "units" in cap:
        value_axis_title = "Installed units"
    elif "storage" in cap:
        value_axis_title = "Installed storage capacity"
    elif "energy" in cap:
        value_axis_title = "Installed energy capacity"
    elif "systemwide" in cap:
        value_axis_title = cap.replace("_", " ").capitalize()
        args.update({location_axis: {"title": "Carrier"}})
    else:
        value_axis_title = "Installed capacity"

    if "_max" in cap:
        title = value_axis_title.replace("Installed", "Maximum allowed")
    elif "_min" in cap:
        title = value_axis_title.replace("Installed", "Minimum allowed")
    elif "_equal" in cap:
        title = value_axis_title.replace("Installed", "Allowed")
    else:
        title = value_axis_title

    if "systemwide" not in cap:
        args.update({location_axis: {"title": "Location"}})

    # Grouped, not stacked, barcharts for the systemwide variables
    if "systemwide" in cap:
        args["barmode"] = "group"
    else:
        args["barmode"] = "relative"

    args.update({value_axis: dict(title=value_axis_title), "title": title})

    return args


def plot_capacity(
    model, orient="h", array="all", subset={}, sum_dims=None, squeeze=True, **kwargs
):
    """
    Parameters
    ----------
    array : str or list; default = 'all'
        options: 'all', 'results', 'inputs', the name/list of any energy capacity
        DataArray(s) from inputs/results.
        User can specify 'all' for all input/results capacities, 'inputs'
        for just input capacities, 'results' for just results capacities, or the
        name(s) of any data array(s) to plot (in either inputs or results).
        In all but the last case, arrays can be picked from dropdown in visualisation.
        In the last case, output can be saved to SVG.
    orient : str, optional
        'h' for horizontal or 'v' for vertical barchart
    subset : dict, optional
        Dictionary by which data is selected (using xarray indexing `loc[]`).
        Keys any of ['timeseries', 'locs', 'techs', 'carriers', 'costs']).
    sum_dims : str, optional
        List of dimension names to sum plot variable over.
    squeeze : bool, optional
        Whether to squeeze out dimensions containing only single values.

    """
    dataset = model._model_data.copy()
    locations = natsorted(list(dataset.locs.values))

    if orient in ["horizontal", "h"]:
        orientation = "h"
        location_axis = "yaxis"
        value_axis = "xaxis"
    elif orient in ["vertical", "v"]:
        orientation = "v"
        location_axis = "xaxis"
        value_axis = "yaxis"
    else:
        raise ValueError("Orient must be `v`/`vertical` or `h`/`horizontal`")

    layout = {
        location_axis: dict(
            title="Location",
            showticklabels=True,  # FIXME: Ensure labels do not get hidden if little vertical space
        ),
        "legend": (dict(traceorder="reversed")),
        "autosize": True,
        "hovermode": "closest",
    }

    relevant_vars = _get_relevant_vars(dataset, array)

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
        get_var_data_kwargs={"locations": locations, "orientation": orientation},
        get_var_layout_kwargs={
            "location_axis": location_axis,
            "value_axis": value_axis,
        },
    )

    return data, layout
