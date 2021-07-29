"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

transmission.py
~~~~~~~~~~~~~~~

Plot transmission data.

"""

import pandas as pd
import xarray as xr
import numpy as np

from calliope.preprocess.util import vincenty
from calliope.postprocess.plotting.util import break_name, get_range


def _get_zoom(coordinate_array, width):
    """
    If mapbox is being used for tranmission plotting, get the zoom based on the
    bounding area of the input data and the width (in pixels) of the map
    """

    # Keys are zoom levels, values are m/pixel at that zoom level
    zoom_dict = {
        0: 156412,
        1: 78206,
        2: 39103,
        3: 19551,
        4: 9776,
        5: 4888,
        6: 2444,
        7: 1222,
        8: 610.984,
        9: 305.492,
        10: 152.746,
        11: 76.373,
        12: 38.187,
        13: 19.093,
        14: 9.547,
        15: 4.773,
        16: 2.387,
        17: 1.193,
        18: 0.596,
        19: 0.298,
    }

    bounds = [
        coordinate_array.max(dim="locs").values,
        coordinate_array.min(dim="locs").values,
    ]

    max_distance = vincenty(*bounds)

    metres_per_pixel = max_distance / width

    for k, v in zoom_dict.items():
        if v > metres_per_pixel:
            continue
        else:
            zoom = k - 4
            break

    return zoom


def _get_data(model, var, sum_dims=None):
    var_da = model.get_formatted_array(var).rename({"locs": "locs_to"})

    if sum_dims:
        var_da = var_da.sum(dim=sum_dims)
    techs = list(
        set(model._model_data.techs_transmission.values).intersection(
            var_da.techs.values
        )
    )
    var_df = var_da.loc[dict(techs=techs)].to_pandas()

    clean_var = var_df[
        (var_df != 0)
        & (var_df.columns.isin(model._model_data.techs_transmission.values))
    ]

    clean_var.columns = pd.MultiIndex.from_tuples(
        clean_var.columns.str.split(":").tolist(), names=["techs", "locs_from"]
    )

    return xr.DataArray.from_series(clean_var.stack().stack())


def _fill_scatter(coordinates, energy_cap, scatter_dict, dict_entry, tech):
    mid_edge = (
        lambda _from, _to: (
            coordinates.loc[{"locs": _from, "coordinates": dict_entry}]
            + coordinates.loc[{"locs": _to, "coordinates": dict_entry}]
        ).item()
        / 2
    )

    edge = lambda _from, _to: [
        coordinates.loc[{"locs": _from, "coordinates": dict_entry}].item(),
        coordinates.loc[{"locs": _to, "coordinates": dict_entry}].item(),
        None,
    ]

    links = []
    filled_list = []

    for loc_from in energy_cap.loc[dict(techs=tech)].locs_from:
        for loc_to in energy_cap.loc[dict(techs=tech)].locs_to:
            if [loc_to, loc_from] in links:
                continue
            e_cap = energy_cap.loc[
                dict(techs=tech, locs_to=loc_to, locs_from=loc_from)
            ].fillna(0)
            if e_cap:
                links.append([loc_from, loc_to])
                if dict_entry == "text":
                    e_cap = "inf" if np.isinf(e_cap.item()) else int(e_cap.item())
                    filled_list.append("{} capacity: {}".format(tech, e_cap))
                else:
                    filled_list.append(
                        edge(loc_from, loc_to)
                        if scatter_dict == "edge"
                        else mid_edge(loc_from, loc_to)
                    )

    return filled_list


def _get_centre(coordinates):
    """
    Get centre of a map based on given lat and lon coordinates
    """
    centre = (coordinates.max(dim="locs") + coordinates.min(dim="locs")) / 2

    return dict(
        lat=centre.loc[dict(coordinates="lat")].item(),
        lon=centre.loc[dict(coordinates="lon")].item(),
    )


def plot_transmission(model, mapbox_access_token=None, **kwargs):
    """
    Parameters
    ----------
    mapbox_access_token : str, optional
        If given and a valid Mapbox API key, a Mapbox map is drawn
        for lat-lon coordinates, else (by default), a more simple
        built-in map.

    """
    try:
        coordinates = model._model_data.loc_coordinates.sortby("locs")
    except AttributeError:
        raise ValueError(
            "Model does not define location coordinates "
            "- no transmission plotting possible."
        )

    colors = model._model_data.colors
    names = model._model_data.names

    plot_width = 1000

    if hasattr(model, "results") and len(model.results.data_vars) > 0:
        energy_cap = _get_data(model, "energy_cap")
        carrier_prod = _get_data(
            model, "carrier_prod", sum_dims=["timesteps", "carriers"]
        )
        carrier_con = _get_data(
            model, "carrier_con", sum_dims=["timesteps", "carriers"]
        )
        energy_flow = carrier_con.fillna(0) + carrier_prod.fillna(0)
    else:
        energy_cap = _get_data(model, "energy_cap_max")
        energy_flow = energy_cap.copy()
        energy_flow.loc[dict()] = 0

    if sorted(coordinates.coordinates.values) == ["lat", "lon"]:
        h_coord, v_coord = ("lat", "lon")
        if mapbox_access_token:
            scatter_type = "scattermapbox"
            layout_dict = dict(
                mapbox=dict(
                    accesstoken=mapbox_access_token,
                    center=_get_centre(coordinates),
                    zoom=_get_zoom(coordinates, plot_width),
                    style="light",
                )
            )
        else:
            scatter_type = "scattergeo"
            layout_dict = dict(
                geo=dict(
                    scope="world",
                    projection=dict(type="mercator", scale=1),
                    showland=True,
                    showcountries=True,
                    showsubunits=True,
                    showocean=True,
                    showrivers=True,
                    showlakes=True,
                    lonaxis=dict(range=get_range(coordinates, "lon", 0.1)),
                    lataxis=dict(range=get_range(coordinates, "lat", 0.1)),
                    resolution=50,
                    landcolor="rgba(240, 240, 240, 0.8)",
                    oceancolor="#aec6cf",
                    subunitcolor="blue",
                    countrycolor="green",
                    countrywidth=0.5,
                    subunitwidth=0.5,
                )
            )
    else:
        h_coord, v_coord = ("x", "y")
        scatter_type = "scatter"
        layout_dict = dict()

    mid_edge_scatter_dict = {
        "type": scatter_type,
        "showlegend": False,
        "mode": "markers",
        "hoverinfo": "text",
        "opacity": 0,
    }

    edge_scatter_dict = {
        "type": scatter_type,
        "mode": "lines",
        "hoverinfo": "none",
        "opacity": 0.8,
    }

    data = []

    for tech in sorted(energy_cap.techs.values):
        per_tech_mid_edge_dict = mid_edge_scatter_dict.copy()
        per_tech_mid_edge_dict = {
            **mid_edge_scatter_dict,
            **{
                h_coord: _fill_scatter(
                    coordinates, energy_cap, "mid_edge", h_coord, tech
                ),
                v_coord: _fill_scatter(
                    coordinates, energy_cap, "mid_edge", v_coord, tech
                ),
                "text": _fill_scatter(
                    coordinates, energy_cap, "mid_edge", "text", tech
                ),
                "legendgroup": tech,
                "marker": {"color": colors.loc[dict(techs=tech)].item()},
            },
        }

        h_edge = _fill_scatter(coordinates, energy_cap, "edge", h_coord, tech)
        v_edge = _fill_scatter(coordinates, energy_cap, "edge", v_coord, tech)
        showlegend = True
        for i in range(len(h_edge)):
            data.append(
                {
                    **edge_scatter_dict,
                    **{
                        h_coord: h_edge[i],
                        v_coord: v_edge[i],
                        "showlegend": showlegend,
                        "legendgroup": tech,
                        "name": break_name(names.loc[dict(techs=tech)].item(), 30),
                        "line": {"color": colors.loc[dict(techs=tech)].item()},
                    },
                }
            )
            showlegend = False
        data.append(per_tech_mid_edge_dict)

    node_scatter_dict = {
        h_coord: [
            coordinates.loc[dict(locs=loc, coordinates=h_coord)].item()
            for loc in coordinates.locs
        ],
        v_coord: [
            coordinates.loc[dict(locs=loc, coordinates=v_coord)].item()
            for loc in coordinates.locs
        ],
        "text": [loc.item() for loc in coordinates.locs],
        "name": "Locations",
        "type": scatter_type,
        "legendgroup": "locations",
        "mode": "markers",
        "hoverinfo": "text",
        "marker": {"symbol": "square", "size": 8, "color": "grey"},
    }

    data.append(node_scatter_dict)

    layout_dict.update(
        dict(
            width=plot_width,
            title=model.model_config["name"],
            autosize=True,
            hovermode="closest",
            showlegend=True,
        )
    )

    if kwargs.get("html_only", False):
        del layout_dict["title"]
        del layout_dict["width"]

    return data, layout_dict
