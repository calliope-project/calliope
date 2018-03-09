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


PLOTLY_KWARGS = dict(
    show_link=False,
    config={
        'displaylogo': False,
        'modeBarButtonsToRemove': ['sendDataToCloud']
    }
)


def plot_timeseries(
        model, timeseries_type='carrier', timesteps_zoom=None,
        loc=dict(), sum_dims='locs', squeeze=True, html_only=False):
    """
    Params
    ------
    timeseries_type : str, optional
        "carrier", "carrier_prod", "carrier_con", "storage", or "resource"
    timesteps_zoom : int, optional
        Number of timesteps to show initially on the x-axis (if not
        given, the full time range is shown by default).
    loc : dict, optional
        Dictionary by which data is selected (keys any of ['timeseries',
        'locs', 'techs', 'carriers']).
    sum_dims : str, optional
        List of dimension names to sum plot variable over.
    squeeze : bool, optional
        Whether to squeeze out dimensions containing only single values.
    html_only : bool, optional, default = False
        Returns a html string for embedding the plot in a webpage

    """
    if timeseries_type in ['carrier', 'carrier_prod', 'carrier_con']:
        reindexer = dict(
            techs=model._model_data.techs.values,
            locs=model._model_data.locs.values,
            carriers=model._model_data.carriers.values
        )
        title = 'Carrier flow'
        y_axis_title = 'Energy produced(+)/consumed(-) (kWh)'
        array_prod = model.get_formatted_array('carrier_prod')
        array_con = model.get_formatted_array('carrier_con')
        array_flow = (
            array_prod.reindex(**reindexer).fillna(0) +
            array_con.reindex(**reindexer).fillna(0)).loc[loc]
    else:
        reindexer = dict(
            techs=model._model_data.techs.values,
            locs=model._model_data.locs.values
        )
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
        elif timeseries_type == 'resource_con':
            title = 'Consumed resource'
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
    timesteps = pd.to_datetime(model._model_data.timesteps.values)

    layout = dict(
        barmode='relative', title=title,
        xaxis=dict(),
        yaxis=dict(title=y_axis_title),
        legend=(dict(traceorder='reversed'))
    )

    if timesteps_zoom:
        layout['xaxis']['range'] = [timesteps[0], timesteps[timesteps_zoom]]

    for tech in array_flow.techs.values:
        tech_dict = dict(techs=tech)
        base_tech = model._model_data.inheritance.loc[tech_dict].item().split('.')[0]

        if base_tech in ['transmission']:
            continue  # Transmission is not plotted here

        if base_tech == 'demand' and array_flow.loc[tech_dict].sum():
            # Always insert demand at position 0 in the list, to make
            # sure it appears on top in the legend
            data.insert(0, go.Scatter(
                x=timesteps, y=-array_flow.loc[tech_dict].values,
                line=dict(color='red'),
                name=model._model_data.names.loc[tech_dict].item())
            )

        elif array_flow.loc[tech_dict].sum():
            data.append(go.Bar(
                x=timesteps, y=array_flow.loc[tech_dict].values,
                name=model._model_data.names.loc[tech_dict].item(),
                legendgroup=tech,
                marker=dict(color=model._model_data.colors.loc[tech_dict].item())))

    if html_only:
        return pltly.plot(
            dict(data=data, layout=layout),
            include_plotlyjs=False, output_type='div',
            **PLOTLY_KWARGS
        )

    elif data:
        pltly.iplot(dict(data=data, layout=layout), **PLOTLY_KWARGS)

    else:
        print('No data to plot')


def plot_capacity(
        model, cap_type='energy_cap', orient='horizontal',
        loc=dict(), sum_dims=None, squeeze=True, html_only=False):
    """
    Params
    ------
    cap_type : str, optional
    orient : str, optional
        'horizontal' or 'vertical' barchart
    loc : dict, optional
        Dictionary by which data is selected (keys any of ['timeseries',
        'locs', 'techs', 'carriers']).
    sum_dims : str, optional
        List of dimension names to sum plot variable over.
    squeeze : bool, optional
        Whether to squeeze out dimensions containing only single values.
    html_only : bool, optional, default = False
        Returns a html string for embedding the plot in a webpage

    """
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

    if orient in ['horizontal', 'h']:
        orientation = 'h'
        xaxis = dict(title=y_axis_title)
        yaxis = dict(title='Location')
    elif orient in ['vertical', 'v']:
        orientation = 'v'
        yaxis = dict(title=y_axis_title)
        xaxis = dict(title='Location')
    else:
        raise ValueError('Orient must be `v`/`vertical` or `h`/`horizontal`')

    layout = dict(
        barmode='relative', title='Installed {}'.format(cap_type),
        showlegend=True, xaxis=xaxis, yaxis=yaxis
    )

    for tech in array_cap.techs.values:
        if tech not in model._model_data.techs.values:
            continue
        tech_dict = dict(techs=tech)
        base_tech = model._model_data.inheritance.loc[tech_dict].item().split('.')[0]
        if base_tech in ['transmission', 'demand', 'unmet_demand']:
            continue
        if tech in array_cap.techs.values and array_cap.loc[dict(techs=tech)].sum() > 0:
            x = array_cap.loc[dict(techs=tech)].values
            y = array_cap.locs.values[::-1]
            if orientation == 'v':
                x, y = y[::-1], x  # Fip axes
            data.append(go.Bar(
                x=x, y=y,
                name=model._model_data.names.loc[dict(techs=tech)].item(),
                legendgroup=base_tech,
                text=tech,
                hoverinfo='x+y+name',
                marker=dict(color=model._model_data.colors.loc[dict(techs=tech)].item()),
                orientation=orientation
            ))

    if html_only:
        return pltly.plot(
            dict(data=data, layout=layout),
            include_plotlyjs=False, output_type='div',
            **PLOTLY_KWARGS
        )

    elif data:
        pltly.iplot(dict(data=data, layout=layout), **PLOTLY_KWARGS)

    else:
        print('No data to plot')


def plot_transmission(model, mapbox_access_token=None, html_only=False):
    """
    Params
    ------
    mapbox_access_token : str, optional
        If given and a valid Mapbox API key, a Mapbox map is drawn
        for lat-lon coordinates, else (by default), a more simple
        built-in map.
    html_only : bool, optional, default = False
        Returns a html string for embedding the plot in a webpage

    """
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
        energy_flow.loc[dict()] = 0

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
        showlegend=True
    ))

    if html_only:
        return pltly.plot(
            dict(data=data, layout=layout_dict),
            include_plotlyjs=False, output_type='div',
            **PLOTLY_KWARGS
        )

    elif data:
        pltly.iplot(dict(data=data, layout=layout_dict), **PLOTLY_KWARGS)

    else:
        print('No data to plot')


class ModelPlotMethods:
    def __init__(self, model):
        self._model = model

    def timeseries(self, **kwargs):
        plot_timeseries(self._model, **kwargs)

    timeseries.__doc__ = plot_timeseries.__doc__

    def capacity(self, **kwargs):
        plot_capacity(self._model, **kwargs)

    capacity.__doc__ = plot_capacity.__doc__

    def transmission(self, **kwargs):
        plot_transmission(self._model, **kwargs)

    transmission.__doc__ = plot_transmission.__doc__

    def summary(self, carrier, mapbox_access_token=None):

        timeseries = plot_timeseries(self._model, html_only=True, loc=dict(carriers=carrier))

        capacity = plot_capacity(self._model, html_only=True, cap_type='energy_cap')

        transmission = plot_transmission(self._model, html_only=True, mapbox_access_token=mapbox_access_token)

        return dict(timeseries=timeseries, capacity=capacity, transmission=transmission)
