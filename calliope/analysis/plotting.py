"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

plotting.py
~~~~~~~~~~~

Functionality to plot model data.

"""

import os
import re

import numpy as np
import pandas as pd
import xarray as xr
import plotly.offline as pltly
import plotly.graph_objs as go
import jinja2

from calliope import exceptions
from calliope.analysis.util import get_zoom
from itertools import product

PLOTLY_KWARGS = dict(
    show_link=False,
    config={
        'displaylogo': False,
        'modeBarButtonsToRemove': ['sendDataToCloud'],
    }
)


def plot_summary(model, out_file=None, mapbox_access_token=None):
    timeseries = plot_timeseries(model, html_only=True)
    capacity = plot_capacity(model, html_only=True)
    transmission = plot_transmission(model, html_only=True, mapbox_access_token=mapbox_access_token)

    with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'plots_template.html'), 'r') as f:
        html_template = jinja2.Template(f.read())

    html = html_template.render(
        model_name=model._model_data.attrs['model.name'],
        calliope_version=model._model_data.attrs['calliope_version'],
        solution_time=(model._model_data.attrs['solution_time'] / 60),
        time_finished=model._model_data.attrs['time_finished'],
        top=timeseries,
        bottom_left=capacity,
        bottom_right=transmission,
    )

    # Strip plotly-inserted style="..." attributes
    html = re.sub(r'style=".+?"', '', html)

    if out_file:
        with open(out_file, 'w') as f:
            f.write(html)
    else:
        return html


def plot_timeseries(
        model, array='all', timesteps_zoom=None, subset=dict(), sum_dims='locs',
        squeeze=True, html_only=False, save_svg=False):
    """
    Params
    ------
    array : str or list; default = 'all'
        options: 'all', 'results', 'inputs', the name/list of any energy carrier(s)
        (e.g. 'power'), the name/list of any input/output DataArray(s).

        User can specify 'all' for all input/results timeseries plots, 'inputs'
        for just input timeseries, 'results' for just results timeseries, or the
        name of any data array to plot (in either inputs or results).
        In all but the last case, arrays can be picked from dropdown in visualisaiton.
        In the last case, output can be saved to SVG and a rangeslider can be used.

    timesteps_zoom : int, optional
        Number of timesteps to show initially on the x-axis (if not
        given, the full time range is shown by default).
    subset : dict, optional
        Dictionary by which data is subset (uses xarray `loc` indexing). Keys
        any of ['timeseries', 'locs', 'techs', 'carriers', 'costs'].
    sum_dims : str, optional
        List of dimension names to sum plot variable over.
    squeeze : bool, optional
        Whether to squeeze out dimensions of length = 1.
    html_only : bool, optional, default = False
        Returns a html string for embedding the plot in a webpage
    save_svg : bool, optional; default = false
        Will save plot to svg on rendering
    """

    dataset = model._model_data.copy()
    carriers = list(dataset.carriers.values)
    timesteps = pd.to_datetime(model._model_data.timesteps.values)

    if array not in (
        ['results', 'inputs', 'all'] + carriers + ['storage', 'resource_con', 'cost_var'] +
        [k for k, v in model.inputs.data_vars.items() if 'timesteps' in v.dims and len(v.dims) > 1]
    ):
        raise exceptions.ModelError(
            'Cannot plot array={}. If you want carrier flow (_prod, _con, _export) '
            'then specify the name of the energy carrier as array'.format(array)
        )

    # relevant_vars are all variables relevant to this plotting instance
    relevant_vars = []

    if array in ['results', 'all']:
        relevant_vars += carriers + ['resource_con', 'cost_var']
        if 'storage' in dataset:
            relevant_vars.append('storage')

    if array in ['inputs', 'all']:
        relevant_vars += [k for k, v in model.inputs.data_vars.items()
                          if 'timesteps' in v.dims and len(v.dims) > 1]

    # if not 'all', 'inputs', or 'results', take user input
    if not relevant_vars and not isinstance(array, list):
        relevant_vars = [array]

    relevant_vars = sorted(list(
        set(relevant_vars).intersection([k for k in dataset.data_vars.keys()] + carriers)
    ))
    # data_len is used to populate visibility of traces, for dropdown
    data_len = [0]
    data = []
    buttons = []

    def get_var_data(var, visible):
        """
        Get variable data from model_data and use it to populate a list with Plotly plots
        """
        # list to populate
        data = []

        def _get_reindexed_array(array, carrier_index=False, fillna=None):
            # reindexing data means that DataArrays have the same values in locs and techs
            reindexer = dict(techs=dataset.techs.values, locs=dataset.locs.values)
            formatted_array = model.get_formatted_array(array)
            if carrier_index:
                reindexer.update(dict(carriers=carriers))
            if fillna is not None:
                return formatted_array.reindex(**reindexer).fillna(fillna)
            else:
                return formatted_array.reindex(**reindexer)

        if hasattr(model, 'results'):
            array_prod = _get_reindexed_array('carrier_prod', carrier_index=True, fillna=0)
            array_con = _get_reindexed_array('carrier_con', carrier_index=True, fillna=0)

            resource_con = _get_reindexed_array('resource_con', fillna=0)

        # carrier flow is a combination of carrier_prod, carrier_con and
        # carrier_export for a given energy carrier
        if var in carriers:
            array_flow = (array_prod.loc[dict(carriers=var)] + array_con.loc[dict(carriers=var)])
            if 'carrier_export' in dataset:
                export_flow = _get_reindexed_array('carrier_export', carrier_index=True, fillna=0)
                array_flow = array_flow - export_flow.loc[dict(carriers=var)]
        # array flow for storage tracks stored energy. carrier_flow is
        # charge/discharge (including resource consumed for supply_plus techs)
        elif var == 'storage':
            array_flow = model.get_formatted_array('storage')
            # currently hacked to assume summing over some dimension
            carrier_flow = (array_prod.sum('carriers') + array_con.sum('carriers') - resource_con).sum(sum_dims)

        elif var == 'resource_con':
            array_flow = resource_con

        else:
            array_flow = model.get_formatted_array(var)

        if subset:  # first, subset the data
            allowed_subsets = [{k: v for k, v in subset.items() if k in array_flow.dims}]
            array_flow = array_flow.loc[allowed_subsets]

        if sum_dims:  # second, sum along all necessary dimensions
            array_flow = array_flow.sum(sum_dims)

        if squeeze and len(array_flow.techs.values) > 1:  # finally, squeeze out single length dimensions
            array_flow = array_flow.squeeze()

        if 'timesteps' not in array_flow.dims or len(array_flow.dims) > 2:
            e = exceptions.ModelError
            raise e('Cannot plot timeseries for variable `{}` with subset `{}`'
                    'and `sum_dims: {}`'.format(var, subset, sum_dims))

        for tech in array_flow.techs.values:
            tech_dict = dict(techs=tech)

            # We allow transmisison tech information to show up in some cases
            if tech in dataset.techs_transmission.values:
                base_tech = 'transmission'
                color = dataset.colors.loc[{'techs': tech.split(':')[0]}].item()
                name = dataset.names.loc[{'techs': tech.split(':')[0]}].item()
            else:
                base_tech = dataset.inheritance.loc[tech_dict].item().split('.')[0]
                color = dataset.colors.loc[tech_dict].item()
                name = dataset.names.loc[tech_dict].item()

            # no transmission in carrier flow
            if base_tech in ['transmission'] and var in carriers:
                continue  # Transmission is not plotted here

            elif base_tech == 'demand' and array_flow.loc[tech_dict].sum():
                # Always insert demand at position 0 in the list, to make
                # sure it appears on top in the legend
                data.insert(0, go.Scatter(
                    x=timesteps, y=-array_flow.loc[tech_dict].values,
                    visible=visible, line=dict(color=color), name=name)
                )

            elif var == 'storage' and array_flow.loc[tech_dict].sum():
                # stored energy as scatter, carrier/resource prod/con as stacked bar
                data.insert(0, go.Scatter(
                    x=timesteps, y=array_flow.loc[tech_dict].values, visible=visible,
                    line=dict(color=color), legendgroup='storage', mode='lines',
                    name=name + ' stored energy')
                )
                data.append(go.Bar(
                    x=timesteps, y=carrier_flow.loc[tech_dict].values, visible=visible,
                    name=name + ' charge/discharge', legendgroup=tech, marker=dict(color=color)
                ))

            elif array_flow.loc[tech_dict].sum():
                data.append(go.Bar(
                    x=timesteps, y=array_flow.loc[tech_dict].values, visible=visible,
                    name=name, legendgroup=tech, marker=dict(color=color)
                ))

        return data

    def get_var_layout(var, visible_data=None):
        """
        Variable-specific layout. Increases axis verbosity for some known variables.
        `visible` used in dropdown, not if only one array is shown.
        """

        args = {}
        if var in carriers:
            title = 'Carrier flow'
            y_axis_title = 'Energy produced(+)/consumed(-)'
        elif var == 'resource':
            title = 'Available resource'
            y_axis_title = 'Energy (per uniit of area)'
        elif var == 'resource_con':
            title = 'Consumed resource'
            y_axis_title = 'Energy'
        elif var == 'cost_var':
            title = 'Variable costs'
            y_axis_title = 'Cost'
        else:
            title = y_axis_title = '{}'.format(var)
        args.update({'yaxis': dict(title=y_axis_title), 'title': title})
        if visible_data is not None:
            return [{'visible': list(visible_data)}, args]
        else:
            return args

    # fill trace data and add number of traces per var to 'data_len' for use with
    # visibility. first var in loop has visibility == True by default
    visible = True
    for var in relevant_vars:
        data += get_var_data(var, visible)
        data_len.append(len(data))
        visible = False

    # Initialise all visibility to False for dropdown updates
    total_data_arrays = np.array([False for i in range(data_len[-1])])
    var_num = 0
    for var in relevant_vars:
        # update visibility to True for all traces linked to this variable `var`
        visible_data = total_data_arrays.copy()
        visible_data[data_len[var_num]:data_len[var_num + 1]] = True

        # Get variable-specific layout
        var_layout = get_var_layout(var, visible_data)

        var_num += 1
        if len(relevant_vars) > 1:
            buttons.append(dict(label=var, method='update', args=var_layout))

    layout = dict(barmode='relative', xaxis=dict(), legend=(dict(traceorder='reversed')), autosize=True)

    # If there are multiple vars to plot, use dropdowns via 'updatemenus'
    if len(relevant_vars) > 1:
        updatemenus = list([dict(
            active=0, buttons=buttons, type='dropdown',
            xanchor='auto', y=1.2, pad=dict(t=0.05, b=0.05, l=0.05, r=0.05)
        )])
        layout['updatemenus'] = updatemenus
    else:
        # If there is one var, rangeslider can be added without the ensuing plot
        # running too slowly
        layout['xaxis']['rangeslider'] = dict()

    if timesteps_zoom:
        layout['xaxis']['range'] = [timesteps[0], timesteps[timesteps_zoom]]

    # html_only used to create a custom dashboard by returning only the html
    # of the plot
    if html_only:

        return pltly.plot(
            dict(data=data, layout=layout),
            include_plotlyjs=False, output_type='div',
            **PLOTLY_KWARGS
        )

    if save_svg:
        if len(relevant_vars) > 1:
            print('Unable to save multiple arrays to SVG, pick one array only')
        else:
            PLOTLY_KWARGS.update(image='svg')
    elif data:
        pltly.iplot(dict(data=data, layout=layout), **PLOTLY_KWARGS)
    else:
        print('No data to plot')


def plot_capacity(
        model, orient='h', array='all',
        subset=dict(), sum_dims=None, squeeze=True, html_only=False, save_svg=False):
    """
    Params
    ------
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
    html_only : bool, optional, default = False
        Returns a html string for embedding the plot in a webpage
    save_svg : bool, optional; default = false
        Will save plot to svg on rendering

    """
    dataset = model._model_data.copy()
    locations = dataset.locs

    if array not in (
        ['results', 'inputs', 'all'] +
        ['resource_area', 'energy_cap', 'resource_cap', 'storage_cap', 'units'] +
        [i + j for i, j in
         product(['resource_area', 'energy_cap', 'resource_cap', 'storage_cap', 'units'],
                 ['_max', '_min', '_equals'])]
    ):
        raise exceptions.ModelError(
            'Cannot plot array={}. If you want carrier flow (_prod, _con, _export) '
            'then specify the name of the energy carrier as array'.format(array)
        )

    # relevant_vars are all variables relevant to this plotting instance
    relevant_vars = []

    if array in ['results', 'all']:
        relevant_vars += ['resource_area', 'energy_cap', 'resource_cap', 'storage_cap', 'units']

    if array in ['inputs', 'all']:
        relevant_vars += [
            i + j for i, j in
            product(['resource_area', 'energy_cap', 'resource_cap', 'storage_cap', 'units'],
                    ['_max', '_min', '_equals'])
        ]

    # if not 'all', 'inputs', or 'results', take user input
    if not relevant_vars and not isinstance(array, list):
        relevant_vars = [array]
    relevant_vars = sorted(list(set(relevant_vars).intersection(dataset.data_vars.keys())))
    # data_len is used to populate visibility of traces, for dropdown
    data_len = [0]
    data = []
    buttons = []

    if orient in ['horizontal', 'h']:
        orientation = 'h'
        location_axis = 'yaxis'
        value_axis = 'xaxis'
    elif orient in ['vertical', 'v']:
        orientation = 'v'
        location_axis = 'xaxis'
        value_axis = 'yaxis'
    else:
        raise ValueError('Orient must be `v`/`vertical` or `h`/`horizontal`')

    def get_cap_data(cap, visible):
        array_cap = model.get_formatted_array(cap).reindex(locs=locations)

        if subset:  # first, subset the data
            allowed_subsets = [{k: v for k, v in subset.items() if k in array_cap.dims}]
            array_cap = array_cap.loc[allowed_subsets]

        if sum_dims:  # second, sum along all necessary dimensions
            array_cap = array_cap.sum(sum_dims)

        if squeeze and len(array_cap.techs.values) > 1:  # finally, squeeze out single length dimensions
            array_cap = array_cap.squeeze()

        if len(array_cap.dims) > 2:
            raise exceptions.ModelError(
                'Maximum two dimensions allowed for plotting capacity, but {} '
                'given as dimensions for {}'.format(array_cap.dims, cap)
            )
        if 'techs' not in array_cap.dims:
            e = exceptions.ModelError
            raise e('Cannot plot capacity without `techs` in dimensions')

        data = []

        for tech in array_cap.techs.values:
            if tech not in dataset.techs.values:
                continue
            if tech in dataset.loc_techs_transmission.values:
                base_tech = 'transmission'
            else:
                base_tech = dataset.inheritance.loc[{'techs': tech}].item().split('.')[0]

            if base_tech in ['transmission', 'demand', 'unmet_demand']:
                continue
            if array_cap.loc[{'techs': tech}].sum() > 0:
                x = array_cap.loc[{'techs': tech}].values
                y = array_cap.locs.values
                if orientation == 'v':
                    x, y = y, x  # Flip axes
                data.append(go.Bar(
                    x=x, y=y, visible=visible,
                    name=model._model_data.names.loc[{'techs': tech}].item(),
                    legendgroup=base_tech,
                    text=tech,
                    hoverinfo='x+y+name',
                    marker=dict(color=model._model_data.colors.loc[{'techs': tech}].item()),
                    orientation=orientation
                ))
        return data

    def get_cap_layout(cap, visible_data):
        args = {}
        if 'area' in cap:
            value_axis_title = 'Installed area'
        elif 'units' in cap:
            value_axis_title = 'Installed units'
        elif 'storage' in cap:
            value_axis_title = 'Installed capacity'
        else:
            value_axis_title = 'Installed capacity'

        args.update({value_axis: dict(title=value_axis_title)})
        if visible_data is not None:
            return [{'visible': list(visible_data)}, args]
        else:
            return args

    # fill trace data and add number of traces per var to 'data_len' for use with
    # visibility. first var in loop has visibility == True by default
    visible = True
    for var in relevant_vars:
        data += get_cap_data(var, visible)
        data_len.append(len(data))
        visible = False

    # Initialise all visibility to False for dropdown updates
    total_data_arrays = np.array([False for i in range(data_len[-1])])
    var_num = 0
    for var in relevant_vars:
        # update visibility to True for all traces linked to this variable `var`
        visible_data = total_data_arrays.copy()
        visible_data[data_len[var_num]:data_len[var_num + 1]] = True

        # Get variable-specific layout
        cap_layout = get_cap_layout(var, visible_data)

        if len(relevant_vars) > 1:
            buttons.append(dict(label=var, method='update', args=cap_layout))

        var_num += 1

    layout = {'barmode': 'relative', location_axis: dict(title='Location'),
              'legend': (dict(traceorder='reversed')),
              'autosize': True}

    # If there are multiple vars to plot, use dropdowns via 'updatemenus'
    if len(relevant_vars) > 1:
        updatemenus = list([dict(
            active=0, buttons=buttons, type='dropdown',
            xanchor='auto', y=1.2, pad=dict(t=0.05, b=0.05, l=0.05, r=0.05)
        )])
        layout['updatemenus'] = updatemenus

    if html_only:
        return pltly.plot(
            dict(data=data, layout=layout),
            include_plotlyjs=False, output_type='div',
            **PLOTLY_KWARGS
        )

    if save_svg:
        if len(relevant_vars) > 1:
            print('Unable to save multiple arrays to SVG, pick one array only')
        else:
            PLOTLY_KWARGS.update(image='svg')

    elif data:
        pltly.iplot(dict(data=data, layout=layout), **PLOTLY_KWARGS)

    else:
        print('No data to plot')


def plot_transmission(model, mapbox_access_token=None, html_only=False, save_svg=False):
    """
    Params
    ------
    mapbox_access_token : str, optional
        If given and a valid Mapbox API key, a Mapbox map is drawn
        for lat-lon coordinates, else (by default), a more simple
        built-in map.
    html_only : bool, optional, default = False
        Returns a html string for embedding the plot in a webpage
    save_svg: bool, optional, default = False
        Saves the plot to svg, if True. Mapbox backgrounds are saved as a static
        image in this case.

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
        del layout_dict['title']
        del layout_dict['width']
        return pltly.plot(
            dict(data=data, layout=layout_dict),
            include_plotlyjs=False, output_type='div',
            **PLOTLY_KWARGS
        )
    if save_svg:
        PLOTLY_KWARGS.update({'image': 'svg'})
    elif data:
        pltly.iplot(dict(data=data, layout=layout_dict), **PLOTLY_KWARGS)
    else:
        print('No data to plot')


class ModelPlotMethods:
    def __init__(self, model):
        self._model = model

    def timeseries(self, **kwargs):
        return plot_timeseries(self._model, **kwargs)

    timeseries.__doc__ = plot_timeseries.__doc__

    def capacity(self, **kwargs):
        return plot_capacity(self._model, **kwargs)

    capacity.__doc__ = plot_capacity.__doc__

    def transmission(self, **kwargs):
        return plot_transmission(self._model, **kwargs)

    transmission.__doc__ = plot_transmission.__doc__

    def summary(self, **kwargs):
        return plot_summary(self._model, **kwargs)

    summary.__doc__ = plot_summary.__doc__
