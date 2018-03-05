"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

plotting.py
~~~~~~~~~~~

Functionality to plot model data.

"""

import pandas as pd
import xarray as xr
import plotly.offline as pltly
import plotly.graph_objs as go

from calliope import exceptions
from calliope.analysis.util import get_zoom


def plot_timeseries(model, timeseries_type='carrier', loc=dict([]),
                    sum_dims='locs', squeeze=True, tech_order=[]):
    reindexer = dict(techs=model._model_data.techs.values,
                     locs=model._model_data.locs.values)
    if timeseries_type == 'carrier':
        title = 'Carrier flow'
        y_axis_title = 'Energy produced(+)/consumed(-) (kWh)'
        array_prod = model.get_formatted_array('carrier_prod')
        array_con = model.get_formatted_array('carrier_con')
        array_flow = (array_prod.reindex(**reindexer).fillna(0)
                      + array_con.reindex(**reindexer).fillna(0)).loc[loc]
    else:
        array_flow = model.get_formatted_array(timeseries_type).reindex(
            **reindexer).fillna(0).loc[loc]
        if timeseries_type == 'storage':
            title = 'Stored energy'
            y_axis_title = 'Stored energy (kWh)'
        elif timeseries_type == 'resource':
            title = 'Available resource'
            y_axis_title = 'Energy (kWh)'
            resource_area = model.get_formatted_array('resource_area').reindex(
                **reindexer).fillna(1).loc[loc]
            array_flow *= resource_area
        else:
            title = y_axis_title = '{}'.format(timeseries_type)
    if sum_dims:
        array_flow = array_flow.sum(sum_dims)

    if squeeze:
        dims_to_squeeze = [
            i for i in array_flow.dims
            if len(array_flow[i]) == 1 and i is not 'techs'
        ]
        array_flow = array_flow.squeeze(dims_to_squeeze)

    if 'timesteps' not in array_flow.dims:
        e = exceptions.ModelError
        raise e('Cannot plot timeseries for variable `{}` with indexing `loc: {}`'
                'and `sim_dims: {}`'.format(timeseries_type, loc, sum_dims))

    data = []
    colors = model._model_data.colors
    names = model._model_data.names
    timesteps = pd.to_datetime(model._model_data.timesteps.values)

    layout = dict(barmode='relative', title=title, yaxis=dict(title=y_axis_title))

    techs = tech_order if tech_order else model._model_data.techs.values
    for tech in techs:
        if tech not in model._model_data.techs.values:
            continue
        tech_dict = dict(techs=tech)
        base_tech = model._model_data.inheritance.loc[tech_dict].item().split('.')[0]
        if base_tech in ['transmission']:
            continue
        if base_tech == 'demand' and array_flow.loc[tech_dict].sum():
            data.append(go.Scatter(
                x=timesteps, y=-array_flow.loc[tech_dict].values,
                line=dict(color='red'), name=names.loc[tech_dict].item())
            )
        elif array_flow.loc[tech_dict].sum():
            data.append(go.Bar(
                x=timesteps, y=array_flow.loc[tech_dict].values,
                name=names.loc[tech_dict].item(), legendgroup=tech,
                marker=dict(color=colors.loc[tech_dict].item())))
    pltly.iplot(dict(data=data, layout=layout))


def plot_capacity(model, cap_type='energy_cap', loc=dict(),
                  sum_dims=None, squeeze=True, tech_order=[]):
    array_cap = model.get_formatted_array(cap_type).loc[loc]

    if 'area' in cap_type:
        y_axis_title = 'Installed area (m^2)'
    elif 'timesteps' in array_cap.dims:
        y_axis_title = 'Summed {} (kWh)'.format(cap_type)
    elif 'units' in cap_type:
        y_axis_title = 'Installed units'
    elif 'storage' in cap_type:
        y_axis_title = 'Installed capacity (kWh)'
    else:
        y_axis_title = 'Installed capacity (kW)'

    if sum_dims:
        array_cap = array_cap.sum(sum_dims)

    if squeeze:
        dims_to_squeeze = [
            i for i in array_cap.dims
            if len(array_cap[i]) == 1 and i is not 'techs'
        ]
        array_cap = array_cap.squeeze(dims_to_squeeze)

    if len(array_cap.dims) > 2:
        raise exceptions.ModelError(
            'Cannot use capacity plotting with {} as '
            'dimensions'.format(array_cap.dims)
        )
    if 'techs' not in array_cap.dims:
        raise exceptions.ModelError(
            'Cannot plot capacity without `techs` in dimensions'
        )

    data = []
    colors = model._model_data.colors
    names = model._model_data.names

    layout = dict(barmode='relative',
                  title='Installed {}'.format(cap_type),
                  yaxis=dict(title=y_axis_title), xaxis=dict(title='Location'),
                  showlegend=True)

    techs = tech_order if tech_order else model._model_data.techs.values
    for tech in techs:
        if tech not in model._model_data.techs.values:
            continue
        tech_dict = dict(techs=tech)
        base_tech = model._model_data.inheritance.loc[tech_dict].item().split('.')[0]
        if base_tech in ['transmission', 'demand', 'unmet_demand']:
            continue
        if tech in array_cap.techs.values and array_cap.loc[dict(techs=tech)].sum() > 0:
            data.append(go.Bar(
                x=array_cap.locs.values, y=array_cap.loc[dict(techs=tech)].values,
                name=names.loc[dict(techs=tech)].item(), legendgroup=base_tech,
                text=tech, hoverinfo='y+text',
                marker=dict(color=colors.loc[dict(techs=tech)].item())))

    pltly.iplot(dict(data=data, layout=layout))


def plot_transmission(model, mapbox_access_token=None):
    coordinates = model._model_data.loc_coordinates

    colors = model._model_data.colors
    names = model._model_data.names

    plot_width = 1000

    def _get_data(var, sum_dims=None):
        var_da = model.get_formatted_array(var).rename({'locs': 'locs_to'})

        if sum_dims:
            var_da = var_da.sum(dim=sum_dims)
        techs = list(
            set(model._model_data.techs_transmission.values).intersection(var_da.techs.values)
        )
        var_df = var_da.loc[dict(techs=techs)].to_pandas()

        clean_var = var_df[
            (var_df != 0) &
            (var_df.columns.isin(model._model_data.techs_transmission.values))
        ]

        clean_var.columns = pd.MultiIndex.from_tuples(
            clean_var.columns.str.split(':').tolist(), names=['techs', 'locs_from']
        )

        return xr.DataArray.from_series(clean_var.stack().stack())

    def _fill_scatter(scatter_dict, dict_entry, tech):
        mid_edge = lambda _from, _to: (
            coordinates.loc[{'locs': _from, 'coordinates': dict_entry}]
            + coordinates.loc[{'locs': _to, 'coordinates': dict_entry}]
        ).item() / 2

        edge = lambda _from, _to: [
            coordinates.loc[{'locs': _from, 'coordinates': dict_entry}].item(),
            coordinates.loc[{'locs': _to, 'coordinates': dict_entry}].item(), None
        ]

        links = []
        filled_list = []

        for loc_from in energy_cap.loc[dict(techs=tech)].locs_from:
            for loc_to in energy_cap.loc[dict(techs=tech)].locs_to:
                if [loc_to, loc_from] in links:
                    continue
                e_cap = energy_cap.loc[dict(techs=tech, locs_to=loc_to,
                                            locs_from=loc_from)].fillna(0)
                if e_cap:
                    links.append([loc_from, loc_to])
                    if dict_entry == 'text':
                        filled_list.append('{} capacity: {}'.format(tech, int(e_cap.item())))
                    else:
                        filled_list.append(edge(loc_from, loc_to)
                                           if scatter_dict == 'edge'
                                           else mid_edge(loc_from, loc_to))

        return filled_list

    def _get_centre(coordinates):
        """
        Get centre of a map based on given lat and lon coordinates
        """
        centre = (coordinates.max(dim='locs') + coordinates.min(dim='locs')) / 2

        return dict(lat=centre.loc[dict(coordinates='lat')].item(),
                    lon=centre.loc[dict(coordinates='lon')].item())

    if hasattr(model, 'results'):
        energy_cap = _get_data('energy_cap')
        carrier_prod = _get_data('carrier_prod', sum_dims=['timesteps', 'carriers'])
        carrier_con = _get_data('carrier_con', sum_dims=['timesteps', 'carriers'])
        energy_flow = carrier_con.fillna(0) + carrier_prod.fillna(0)
    else:
        energy_cap = _get_data('energy_cap_max')
        energy_flow = energy_cap.copy()
        energy_flow.loc[dict([])] = 0

    if sorted(coordinates.coordinates.values) == ['lat', 'lon']:
        h_coord, v_coord = ('lat', 'lon')
        if mapbox_access_token:
            scatter_type = 'scattermapbox'
            layout_dict = dict(
                mapbox=dict(
                    accesstoken=mapbox_access_token,
                    center=_get_centre(coordinates),
                    zoom=get_zoom(coordinates, plot_width),
                    style='light'
                )
            )
        else:
            scatter_type = 'scattergeo'
            layout_dict = dict(
                geo=dict(
                    scope='world',
                    projection=dict(type='mercator', scale=1),
                    showland=True,
                    showcountries=True,
                    showsubunits=True,
                    showocean=True,
                    showrivers=True,
                    showlakes=True,
                    lonaxis=dict(range=[
                        coordinates.loc[dict(coordinates='lon')].min().item() - 1,
                        coordinates.loc[dict(coordinates='lon')].max().item() + 1]),
                    lataxis=dict(range=[
                        coordinates.loc[dict(coordinates='lat')].min().item() - 1,
                        coordinates.loc[dict(coordinates='lat')].max().item() + 1]),
                    resolution=50,
                    landcolor="rgba(240, 240, 240, 0.8)",
                    oceancolor='#aec6cf',
                    subunitcolor="blue",
                    countrycolor="green",
                    countrywidth=0.5,
                    subunitwidth=0.5,
                )
            )
    else:
        h_coord, v_coord = ('x', 'y')
        scatter_type = 'scatter'
        layout_dict = dict()

    mid_edge_scatter_dict = {
        'type': scatter_type,
        'showlegend': False,
        'mode': 'markers',
        'hoverinfo': 'text',
        'opacity': 0
    }

    edge_scatter_dict = {
        'type': scatter_type,
        'mode': 'lines',
        'hoverinfo': 'none',
        'opacity': 0.8
    }

    data = []

    for tech in energy_cap.techs:
        per_tech_mid_edge_dict = mid_edge_scatter_dict.copy()
        per_tech_mid_edge_dict = {**mid_edge_scatter_dict, **{
            h_coord: _fill_scatter('mid_edge', h_coord, tech.item()),
            v_coord: _fill_scatter('mid_edge', v_coord, tech.item()),
            'text': _fill_scatter('mid_edge', 'text', tech.item()),
            'legendgroup': tech.item(),
            'marker': {'color': colors.loc[dict(techs=tech)].item()}}}

        h_edge = _fill_scatter('edge', h_coord, tech.item())
        v_edge = _fill_scatter('edge', v_coord, tech.item())
        showlegend = True
        for i in range(len(h_edge)):
            data.append({**edge_scatter_dict, **{
                h_coord: h_edge[i],
                v_coord: v_edge[i],
                'showlegend': showlegend,
                'legendgroup': tech.item(),
                'name': names.loc[dict(techs=tech)].item(),
                'line': {'color': colors.loc[dict(techs=tech)].item()}
            }})
            showlegend = False
        data.append(per_tech_mid_edge_dict)

    node_scatter_dict = {
        h_coord: [coordinates.loc[dict(locs=loc, coordinates=h_coord)].item()
                  for loc in coordinates.locs],
        v_coord: [coordinates.loc[dict(locs=loc, coordinates=v_coord)].item()
                  for loc in coordinates.locs],
        'text': [loc.item() for loc in coordinates.locs],
        'name': 'Locations',
        'type': scatter_type,
        'legendgroup': 'locations',
        'mode': 'markers',
        'hoverinfo': 'text',
        'marker': {'symbol': 'square', 'size': 8, 'color': 'grey'}
    }

    data.append(node_scatter_dict)

    layout_dict.update(dict(
        width=plot_width,
        title=model._model_data.attrs['model.name'],
        autosize=True,
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation='h', y=1.01)
    ))

    fig = go.Figure(data=data, layout=layout_dict)

    pltly.iplot(fig, filename='network')
