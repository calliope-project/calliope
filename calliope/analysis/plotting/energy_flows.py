"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

energy_flows.py
~~~~~~~~~~~~~~~

Plot energy flows data.

"""

import pandas as pd


def _format_date(date, timeseries_format):
    ts = pd.to_datetime(date)
    return ts.strftime(timeseries_format)


def _scale_prod_factor(carrier_prod):
    return 100 / abs(carrier_prod.values.max() - carrier_prod.values.min())


def _line(loc_coordinates, location, carrier, tech, prod,
          scale_factor, techs_colors, is_initial_timestep):
    [transmission_type, from_location] = tech.split(":")
    hover_info = "%s>%s: %.2f by %s (%s)" % \
        (from_location, location, prod, transmission_type, carrier)
    # ex: "Region1>Region2: 256.54 by gas_transmission (gas)"

    line = dict(
        visible=False,
        mode="lines",
        hoverinfo="text",
        text="",
        line=dict(
            width=prod * scale_factor + 1,
            color=techs_colors[transmission_type]
        ),
        legendgroup=transmission_type,
        name=tech,
        opacity=0.6,
    )

    line_info_marker = dict(
        visible=False,
        mode="markers",
        hoverinfo="text",
        text=hover_info,
        marker=dict(
            symbol="square",
            opacity=0,
            color=techs_colors[transmission_type]
        ),
        legendgroup=transmission_type,
        name=tech,
        showlegend=False
    )

    h_coord, v_coord = "lon", "lat"  # by default
    if set(loc_coordinates.index) == set(["x", "y"]):
        h_coord, v_coord = "x", "y"
        line["type"] = "scatter"
        line_info_marker["type"] = "scatter"
    elif set(loc_coordinates.index) == set(["lon", "lat"]):
        h_coord, v_coord = "lon", "lat"
        line["type"] = "scattergeo"
        line_info_marker["type"] = "scattergeo"

    line[h_coord] = [
        loc_coordinates[from_location][h_coord],
        loc_coordinates[location][h_coord]
    ]
    line[v_coord] = [
        loc_coordinates[from_location][v_coord],
        loc_coordinates[location][v_coord]
    ]
    line_info_marker[h_coord] = [(1 / 2) * (loc_coordinates[from_location][h_coord] +
                                 loc_coordinates[location][h_coord])]
    line_info_marker[v_coord] = [(1 / 2) * (loc_coordinates[from_location][v_coord] +
                                 loc_coordinates[location][v_coord])]

    if is_initial_timestep:
        # plot only the first timestep data when the chart is initialized
        line["visible"] = True
        line_info_marker["visible"] = True

    return [line, line_info_marker]


def _marker(loc_coordinates, location, carrier, tech, prod, scale_factor,
            techs_colors, is_initial_timestep):
    hover_info = "%s: %.2f of %s (%s)" % \
        (location, prod, tech, carrier)
    # ex: "Region1: 3552.65 of pipe_import (gas)"

    marker_dict = dict(
        visible=False,
        hoverinfo="text",
        text=hover_info,
        mode="markers",
        marker=dict(
            symbol="dot",
            opacity=0.6,
            size=prod * scale_factor + 1,
            color=techs_colors[tech],
        ),
        legendgroup=tech,
        name=tech
    )

    h_coord, v_coord = "lon", "lat"  # by default
    if set(loc_coordinates.index) == set(["x", "y"]):
        h_coord, v_coord = "x", "y"
        marker_dict["type"] = "scatter"
    elif set(loc_coordinates.index) == set(["lon", "lat"]):
        h_coord, v_coord = "lon", "lat"
        marker_dict["type"] = "scattergeo"

    marker_dict[h_coord] = [loc_coordinates[location][h_coord]]
    marker_dict[v_coord] = [loc_coordinates[location][v_coord]]

    if is_initial_timestep:
        # plot only the first timestep data when the chart is initialized
        marker_dict["visible"] = True

    return marker_dict


def _production_data(model, timesteps, timestep):
    """
    returns a list of dicts, each dict is a plotly marker (location production)
    or line (transmission) on the map.
    """
    loc_coordinates = model.inputs.loc_coordinates.to_pandas()
    locs_techs_carriers_production = model.get_formatted_array("carrier_prod")
    techs_colors = model._model_data.colors.to_pandas()
    scale_factor = _scale_prod_factor(model.results.carrier_prod)

    production_data = []
    # we iterate through each dimension for one timestep in order to
    # add the different production sources of the location and line
    # transmissions toward it
    # Complexity: O(len(carriers)*len(techs)*len(locations)). Considering
    # len(carriers) is suppose to be small (<10), we have a large margin in
    # terms of techs and locations numbers (len(techs)*len(locations) <= 10^9,
    # equivalent to one second processing time for python)
    for location in locs_techs_carriers_production.locs.values:
        for carrier in locs_techs_carriers_production.carriers.values:
            techs_production = locs_techs_carriers_production.sel(
                carriers=carrier,
                locs=location
            ).to_pandas()
            for tech, prod in techs_production.loc[:, timestep].iteritems():
                if prod and prod > 0:
                    # if some energy is at stake
                    if len(tech.split(":")) > 1:
                        # "transmission_type:location"
                        # if it gets energy from another location
                        production_data.extend(
                            # "extend" because _line() also returns a
                            # transparent marker dict at the middle of the line
                            # to display info on hover
                            _line(
                                loc_coordinates, location,
                                carrier, tech, prod,
                                scale_factor, techs_colors,
                                timestep == timesteps[0]
                            )
                        )
                    else:
                        # if the energy comes from this location
                        production_data.append(
                            _marker(
                                loc_coordinates, location,
                                carrier, tech, prod,
                                scale_factor, techs_colors,
                                timestep == timesteps[0]
                            )
                        )
    return production_data


def plot_energy_flows(model, timestep_cycle=1,
                      timestep_index_subset=[], **kwargs):
    """
    Parameters
    ----------
    timestep_cycle : int, optional
        Shows one of every timestep_cycle timesteps. Default is 1 (all timesteps
        are shown).
    timestep_index_subset : list of int, optional
        Only the timesteps between those two indexes are shown. Default is []
        (all timesteps are shown).
    """

    if len(timestep_index_subset) == 2:
        timestep_start = timestep_index_subset[0]
        timestep_end = timestep_index_subset[1]
    else:
        timestep_start, timestep_end = 0, len(model._model_data.timesteps.values)

    try:
        model._model_data.loc_coordinates
    except AttributeError:
        raise ValueError(
            'Model does not define location coordinates '
            '- no energy flow plotting possible.'
        )

    timesteps = model._model_data.timesteps.values[
        timestep_start:timestep_end:timestep_cycle
    ]  # slicing the desired timesteps
    timeseries_dateformat = model._model_data.attrs["model.timeseries_dateformat"]

    steps_length = []
    data = []
    for timestep in timesteps:
        data_by_timestep = _production_data(model, timesteps, timestep)
        steps_length.append(len(data_by_timestep))
        data.extend(data_by_timestep)

    steps = []
    for i, timestep in enumerate(timesteps):
        step = dict(
            # active="label of first show timestep data",
            method="restyle",
            args=["visible", [False] * len(data)],
            label=_format_date(timestep, timeseries_dateformat),
        )
        i_start = sum(steps_length[:i])  # visible start index
        i_end = i_start + steps_length[i]  # visible end index
        step["args"][1][i_start: i_end] = [True] * steps_length[i]
        # we set visible to True for all the points of one timestep
        steps.append(step)

    sliders = [dict(
        # active="start sliding",True
        currentvalue=dict(
            visible=True,
            prefix="Timestep: ",
        ),
        pad={"t": 50},
        activebgcolor="black",
        bgcolor="grey",
        steps=steps
    )]

    # define the map general layout here
    layout = dict(
        title="Energy Flow",
        showlegend=True,
        width="900",
        height="700",
        hovermode="closest",
        sliders=sliders,
    )

    def get_range(axis):
        # returns range of the graph
        _range = [
            model._model_data.loc_coordinates.loc[dict(coordinates=axis)].min().item(),
            model._model_data.loc_coordinates.loc[dict(coordinates=axis)].max().item()
        ]
        _offset = abs(_range[1] - _range[0]) * 0.1
        return [_range[0] - _offset, _range[1] + _offset]

    # change the range of the plot whether its x,y or lat,lon coords
    if set(model.inputs.loc_coordinates.coordinates.values) == set(["x", "y"]):
        layout["xaxis"] = dict(range=get_range("x"))
        layout["yaxis"] = dict(range=get_range("y"))
    elif set(model.inputs.loc_coordinates.coordinates.values) == set(["lon", "lat"]):
        layout["geo"] = dict(
            scope="world",
            showland=True,
            showcountries=True,
            showsubunits=True,
            showocean=True,
            oceancolor="#aec6cf",
            subunitcolor="blue",
            countrycolor="green",
            lonaxis=dict(range=get_range("lon")),
            lataxis=dict(range=get_range("lat")),
            countrywidth=0.5,
            subunitwidth=0.5,
            landcolor="rgb(255,255,255)",
        )

    return data, layout
