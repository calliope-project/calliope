import xarray as xr


_TIMESERIES_PARAMS = ['r', 'e_eff']


def get_dataset(data):
    """Temporary solution to get an xr.Dataset of input data"""
    from ._version import __version__
    arrays = {}
    for p in _TIMESERIES_PARAMS:
        y_dim = '_y_def_{}'.format(p)
        arr = (xr.Dataset(data[p]).to_array(dim=y_dim)
                 .rename({'dim_0': 't', 'dim_1': 'x'})
               )
        arr = arr.loc[{y_dim: list(data[y_dim])}]
        arrays[p] = arr
    ds = xr.Dataset(arrays)

    ds['_weights'] = xr.DataArray(data['_weights'].as_matrix(), dims=['t'])
    ds['_time_res'] = xr.DataArray(data['time_res_series'].as_matrix(), dims=['t'])

    # Replace integer timestep index with actual date-time objects
    ds.coords['t'] = data._dt.as_matrix()

    for a in ['time_res_native', 'time_res_static', 'time_res_data']:
        ds.attrs[a] = data[a]

    ds.attrs['calliope_version'] = __version__

    return ds


##
# Functions based on xarray data
##


def get_y_coord(array):
    # assumes a single _y coord in array
    return [k for k in array.coords if '_y' in k][0]


def get_datavars(data):
    return [var for var in data.data_vars if not var.startswith('_')]


def get_timesteps_per_day(data):
    timesteps_per_day = data.attrs['time_res_static'] * 24
    if isinstance(timesteps_per_day, float):
        assert timesteps_per_day.is_integer(), 'Timesteps/day must be integer.'
        timesteps_per_day = int(timesteps_per_day)
    return timesteps_per_day


def get_freq(data):
    ts_per_day = get_timesteps_per_day(data)
    return ('{}H'.format(24 / ts_per_day))


def reattach(model, new_data):
    # FIXME attach updated data to the model object
    # FIXME TODO update metadata/attributes in model data
    ds = new_data

    for p in _TIMESERIES_PARAMS:
        y_dim = '_y_def_{}'.format(p)
        for y in ds[y_dim].values:
            model.data[p][y] = ds[p].loc[{y_dim: y}].to_pandas().reset_index(drop=True)

    model.data['_weights'] = ds['_weights'].to_pandas().reset_index(drop=True)
    model.data['time_res_series'] = ds['_time_res'].to_pandas().reset_index(drop=True)
    model.data['_dt'] = ds.coords['t'].to_pandas().reset_index(drop=True)
