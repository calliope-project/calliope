"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

flows.py
~~~~~~~~~~~~~~~

Plot energy flows data.
"""

import pandas as pd

from calliope.postprocess.plotting.util import break_name, get_range


def _line(
    locs_coordinates,
    transmission_type,
    to_location,
    from_location,
    carrier,
    tech,
    prod,
    scale_factor,
    techs_colors,
    is_initial_timestep,
    add_legend,
    name,
):
    # e.g. "Region1->Region2: 256.54 by gas_transmission (gas)"
    hover_info = "%s->%s: %.2f by %s (%s)" % (
        from_location,
        to_location,
        prod,
        transmission_type,
        carrier,
    )

    line = dict(
        visible=False,
        mode="lines",
        hoverinfo="text",
        text="",
        line=dict(width=prod * scale_factor + 1, color=techs_colors[transmission_type]),
        legendgroup=transmission_type,
        opacity=0.6,
        showlegend=False,
    )

    line_legend = dict(
        visible=False,
        mode="lines",
        hoverinfo="text",
        text="",
        line=dict(width=10, color=techs_colors[transmission_type]),
        legendgroup=transmission_type,
        name=break_name(name, 18),
        opacity=0.6,
    )

    line_info_marker = dict(
        visible=False,
        mode="markers",
        hoverinfo="text",
        text=hover_info,
        marker=dict(symbol="square", opacity=0, color=techs_colors[transmission_type]),
        legendgroup=transmission_type,
        name=tech,
        showlegend=False,
    )

    if set(locs_coordinates.index) == set(["x", "y"]):
        h_coord, v_coord = "x", "y"
    elif set(locs_coordinates.index) == set(["lon", "lat"]):
        h_coord, v_coord = "lon", "lat"

    line[h_coord] = [
        locs_coordinates[from_location][h_coord],
        locs_coordinates[to_location][h_coord],
    ]
    line[v_coord] = [
        locs_coordinates[from_location][v_coord],
        locs_coordinates[to_location][v_coord],
    ]
    line_legend[h_coord] = [None]
    line_legend[v_coord] = [None]
    line_info_marker[h_coord] = [
        (1 / 2)
        * (
            locs_coordinates[from_location][h_coord]
            + locs_coordinates[to_location][h_coord]
        )
    ]
    line_info_marker[v_coord] = [
        (1 / 2)
        * (
            locs_coordinates[from_location][v_coord]
            + locs_coordinates[to_location][v_coord]
        )
    ]

    if is_initial_timestep:
        # plot only the first timestep data when the chart is initialized
        line["visible"] = True
        line_legend["visible"] = True
        line_info_marker["visible"] = True
    if add_legend:
        return [line, line_legend, line_info_marker]
    else:
        return [line, line_info_marker]


def _marker(
    locs_coordinates,
    location,
    carrier,
    tech,
    prod,
    scale_factor,
    techs_colors,
    is_initial_timestep,
    add_legend,
    name,
):
    # Example: "Region1: 3552.65 of pipe_import (gas)"
    hover_info = "%s: %.2f of %s (%s)" % (location, prod, tech, carrier)

    marker_dict = dict(
        visible=False,
        hoverinfo="text",
        text=hover_info,
        mode="markers",
        marker=dict(
            symbol="circle-dot",
            opacity=0.6,
            size=prod * scale_factor + 1,
            color=techs_colors[tech],
        ),
        legendgroup=tech,
        showlegend=False,
    )

    marker_legend = dict(
        visible=False,
        hoverinfo="text",
        text=hover_info,
        mode="markers",
        marker=dict(
            symbol="circle-dot",
            opacity=0.6,
            size=10,
            color=techs_colors[tech],
        ),
        legendgroup=tech,
        name=break_name(name, 18),
    )

    if set(locs_coordinates.index) == set(["x", "y"]):
        h_coord, v_coord = "x", "y"
    elif set(locs_coordinates.index) == set(["lon", "lat"]):
        h_coord, v_coord = "lon", "lat"

    marker_dict[h_coord] = [locs_coordinates[location][h_coord]]
    marker_dict[v_coord] = [locs_coordinates[location][v_coord]]
    marker_legend[h_coord] = [None]
    marker_legend[v_coord] = [None]

    if is_initial_timestep:
        # plot only the first timestep data when the chart is initialized
        marker_dict["visible"] = True
        marker_legend["visible"] = True
    if add_legend:
        return [marker_dict, marker_legend]
    else:
        return [marker_dict]


def _production_data(model, timesteps, timestep):
    """
    returns a list of dicts, each dict is a plotly marker (location production)
    or line (transmission) on the map.
    """
    locs_coordinates = model._model_data.loc_coordinates.to_pandas()
    locs_techs_carriers_production = model.get_formatted_array("carrier_prod")
    techs_colors = model._model_data.colors.to_pandas()
    scale_factor = 100 / abs(
        model.results.carrier_prod.values.max()
        - model.results.carrier_prod.values.min()
    )
    tech_names = set(model._model_data.techs.values)

    production_data = []
    # we iterate through each dimension for one timestep in order to
    # add the different production sources of the location and line
    # transmissions toward it
    # Complexity: O(len(carriers)*len(techs)*len(locations)). Considering
    # len(carriers) is supposed to be small (<10), we have a large margin in
    # terms of techs and locations numbers (len(techs)*len(locations) <= 10^9,
    # equivalent to one second processing time for python)
    links = []
    # list of sets { tech, from_location, to_location }
    links_data = []
    # links associated data, like prod, carrier, transmission_type
    # [ [prod, carrier, transmission_type], [] ..]
    for location in locs_techs_carriers_production.locs.values:

        for carrier in locs_techs_carriers_production.carriers.values:
            techs_production = locs_techs_carriers_production.sel(
                carriers=carrier, locs=location
            ).to_pandas()
            for tech, prod in techs_production.loc[:, timestep].iteritems():
                if prod and prod > 0:
                    # if some energy is at stake

                    tech_name = tech.split(":")[0]
                    if tech_name in tech_names:
                        add_legend = True
                        tech_names.discard(tech_name)
                        name = model._model_data.names.loc[tech_name].item()
                    else:
                        # only add legend information once for a tech
                        add_legend = False
                        name = ""

                    if len(tech.split(":")) > 1:
                        # "transmission_type:location"
                        # if it gets energy from another location
                        [transmission_type, from_location] = tech.split(":")
                        links.append({tech_name, from_location, location})
                        links_data.append(
                            {
                                "transmission_type": transmission_type,
                                "from_location": from_location,
                                "to_location": location,
                                "prod": prod,
                                "carrier": carrier,
                                "tech": tech,
                                "add_legend": add_legend,
                                "name": name,
                            }
                        )
                    else:
                        # if the energy comes from this location
                        production_data.extend(
                            _marker(
                                locs_coordinates,
                                location,
                                carrier,
                                tech,
                                prod,
                                scale_factor,
                                techs_colors,
                                timestep == timesteps[0],
                                add_legend,
                                name,
                            )
                        )

    def merge(first_link, second_link):
        if first_link["prod"] > second_link["prod"]:
            first_link["prod"] -= second_link["prod"]
            return first_link
        elif first_link["prod"] < second_link["prod"]:
            second_link["prod"] -= first_link["prod"]
            return second_link
        else:
            # the two transmission links are equal,
            # thus, no representation of it is return
            return {}

    # merge the links data
    links_merged = []
    while len(links) > 0:
        data = links_data[0]
        # we check if there is a transmission
        # link in the opposite direction and merge if so
        j = 1
        while j < len(links):
            if links[j] == links[0]:
                data = merge(links_data[0], links_data[j])
                links.remove(links[j])
                links_data.remove(links_data[j])
                j -= 1
            j += 1
        links_merged.append(data)
        links.remove(links[0])
        links_data.remove(links_data[0])

    # add merged links to production_data
    for link in links_merged:
        if link:
            params_list = [
                locs_coordinates,
                link["transmission_type"],
                link["to_location"],
                link["from_location"],
                link["carrier"],
                link["tech"],
                link["prod"],
                scale_factor,
                techs_colors,
                timestep == timesteps[0],
                link["add_legend"],
                link["name"],
            ]
            production_data.extend(_line(*params_list))

    return production_data


def plot_flows(model, timestep_cycle=1, timestep_index_subset=[], **kwargs):
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
        locs_coordinates = model._model_data.loc_coordinates
    except AttributeError:
        raise ValueError(
            "Model does not define location coordinates "
            "- no energy flow plotting possible."
        )

    timesteps = model._model_data.timesteps.values[
        timestep_start:timestep_end:timestep_cycle
    ]  # slicing the desired timesteps
    timeseries_dateformat = model.model_config["timeseries_dateformat"]

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
            label=pd.to_datetime(timestep).strftime(timeseries_dateformat),
        )
        i_start = sum(steps_length[:i])  # visible start index
        i_end = i_start + steps_length[i]  # visible end index
        step["args"][1][i_start:i_end] = [True] * steps_length[i]
        # we set visible to True for all the points of one timestep
        steps.append(step)

    sliders = [
        dict(
            # active="start sliding",True
            currentvalue=dict(
                visible=True,
                prefix="Timestep: ",
            ),
            pad={"t": 50},
            activebgcolor="black",
            bgcolor="grey",
            steps=steps,
        )
    ]

    # define the map general layout here
    layout = dict(
        title="Energy Flow",
        showlegend=True,
        width=900,
        height=700,
        hovermode="closest",
        sliders=sliders,
        margin={"autoexpand": False, "b": 150, "r": 180},
    )

    # change the range of the plot whether its x,y or lat,lon coords
    if sorted(locs_coordinates.coordinates.values) == ["x", "y"]:
        layout["xaxis"] = dict(range=get_range(locs_coordinates, "x", 0.2))
        layout["yaxis"] = dict(range=get_range(locs_coordinates, "y", 0.2))
        for trace in data:
            trace["type"] = "scatter"
    elif sorted(locs_coordinates.coordinates.values) == ["lat", "lon"]:
        layout["geo"] = dict(
            scope="world",
            showland=True,
            showcountries=True,
            showsubunits=True,
            showocean=True,
            oceancolor="#aec6cf",
            subunitcolor="blue",
            countrycolor="green",
            lonaxis=dict(range=get_range(locs_coordinates, "lon", 0.2)),
            lataxis=dict(range=get_range(locs_coordinates, "lat", 0.2)),
            countrywidth=0.5,
            subunitwidth=0.5,
            landcolor="rgb(255,255,255)",
        )
        for trace in data:
            trace["type"] = "scattergeo"

    return data, layout
