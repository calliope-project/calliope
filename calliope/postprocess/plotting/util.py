"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

util.py
~~~~~~~~

Plotting util functions.

"""

import numpy as np
from IPython import get_ipython


def type_of_script():
    if get_ipython() is not None:
        ipy_str = str(type(get_ipython()))
        if "zmqshell" in ipy_str:
            return "jupyter"
        if "terminal" in ipy_str:
            return "ipython"
    else:
        return "terminal"


def get_data_layout(
    get_var_data,
    get_var_layout,
    relevant_vars,
    layout,
    model,
    dataset,
    subset,
    sum_dims,
    squeeze,
    get_var_data_kwargs={},
    get_var_layout_kwargs={},
):
    """
    For each dropdown dataset in capacity and timeseries plotting, build the
    plotly data dictionary and the plotly layout dictionary.

    """
    # data_len is used to populate visibility of traces, for dropdown
    data_len = [0]
    data = []
    buttons = []
    # fill trace data and add number of traces per var to 'data_len' for use with
    # visibility. first var in loop has visibility == True by default
    visible = True

    for var in relevant_vars:
        data += get_var_data(
            var,
            model,
            dataset,
            visible,
            subset,
            sum_dims,
            squeeze,
            **get_var_data_kwargs,
        )
        data_len.append(len(data))
        visible = False

    # Initialise all visibility to False for dropdown updates
    total_data_arrays = np.array([False for i in range(data_len[-1])])
    var_num = 0
    for var in relevant_vars:
        # update visibility to True for all traces linked to this variable `var`
        visible_data = total_data_arrays.copy()
        visible_data[data_len[var_num] : data_len[var_num + 1]] = True

        # Get variable-specific layout
        var_layout = get_var_layout(var, dataset, **get_var_layout_kwargs)

        if var_num == 0:
            layout["title"] = var_layout["title"]
            if "barmode" in var_layout:
                layout["barmode"] = var_layout["barmode"]

        if len(relevant_vars) > 1:
            var_layout = [{"visible": list(visible_data)}, var_layout]
            buttons.append(dict(label=var, method="update", args=var_layout))

        var_num += 1

    # If there are multiple vars to plot, use dropdowns via 'updatemenus'
    if len(relevant_vars) > 1:
        updatemenus = list(
            [
                dict(
                    active=0,
                    buttons=buttons,
                    type="dropdown",
                    xanchor="left",
                    x=0,
                    y=1.13,
                    pad=dict(t=0.05, b=0.05, l=0.05, r=0.05),
                )
            ]
        )
        layout["updatemenus"] = updatemenus
    else:
        layout.update(var_layout)
    return data, layout


def hex_to_rgba(hex_color, opacity):
    """Embed opacity in a colour by converting calliope HEX colours to an RGBA"""
    _NUMERALS = "0123456789abcdefABCDEF"
    _HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
    hex_color = hex_color.lstrip("#")
    rgb = [_HEXDEC[hex_color[0:2]], _HEXDEC[hex_color[2:4]], _HEXDEC[hex_color[4:6]]]
    return "rgba({1}, {2}, {3}, {0})".format(opacity, *rgb)


def break_name(name, length):
    """Take a long technology name string and break it across multiple lines"""

    name_breaks = len(name) / length
    if name_breaks > 1:
        initial_name = name
        break_buffer = 0
        for name_break in range(1, int(name_breaks) + 1):
            # preferably break at a space
            breakpoint = initial_name.rfind(
                " ", name_break * length - 10, name_break * length
            )
            # -1 means rfind failed to find any space
            if breakpoint != -1:
                breakpoint += break_buffer
                name = name[:breakpoint].rstrip() + "<br>" + name[breakpoint:].lstrip()
                break_buffer += 4
            else:
                breakpoint = length * name_break + break_buffer
                name = (
                    name[:breakpoint].rstrip() + "...<br>" + name[breakpoint:].lstrip()
                )
                # '...' is len 3
                break_buffer += 7

    return name


def get_clustered_layout(dataset):
    layout = {}
    timestep_cluster = dataset.timestep_cluster.to_pandas()
    clusters = timestep_cluster.groupby(timestep_cluster).groups
    layout["xaxis"] = {}
    layout["xaxis"]["type"] = "category"
    layout["xaxis"]["tickvals"] = [
        (2 * (k - min(clusters.keys())) + 1) * len(v) / 2 - 0.5
        for k, v in clusters.items()
    ]
    layout["xaxis"]["ticktext"] = [k for k in clusters.keys()]
    layout["xaxis"]["title"] = "Clusters"

    # Make rectangles to fit in the background over every other cluster,
    # to distinguish them
    layout["shapes"] = []
    shape_template = {
        "type": "rect",
        "xref": "x",
        "yref": "paper",
        "y0": 0,
        "y1": 1,
        "line": {"width": 0},
        "layer": "below",
    }

    for cluster in clusters.keys():
        x0 = clusters[cluster][0]
        x1 = clusters[cluster][-1]
        opacity = 0.3 * (cluster % 2)
        day_shape = {"x0": x0, "x1": x1, "fillcolor": "grey", "opacity": opacity}

        shape_template.update(day_shape)
        layout["shapes"].append(shape_template.copy())

    return layout


def get_range(coordinates, axis, buffer):
    """
    Based on lat/lon of the underlying data, get a range which defines the
    bounding box for the map

    Parameters
    ----------
    coordinates: xarray DataArray
        Coordinate data of the underlying data
    axis: str
        axis of the data (lat/lon) by which to slice the coordinates DataArray
    buffer: float
        value, expressed as a fraction of the range of the underlying data,
        to add as a buffer to the bounding box

    Returns
    -------
    list, defining the range as input_range +/- input_range * buffer
    """
    _range = [
        coordinates.loc[dict(coordinates=axis)].min().item(),
        coordinates.loc[dict(coordinates=axis)].max().item(),
    ]
    _offset = abs(_range[1] - _range[0]) * 0.1
    return [_range[0] - _offset, _range[1] + _offset]
