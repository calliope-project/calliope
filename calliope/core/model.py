"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""
import os

import numpy as np

import xarray as xr
import pandas as pd
import ruamel.yaml as ruamel_yaml

from calliope.analysis import plotting, postprocess
from calliope.core import io
from calliope.core.preprocess import \
    model_run_from_dict, \
    build_model_data, \
    apply_time_clustering, \
    final_timedimension_processing
from calliope.core.util.dataset import split_loc_techs, reorganise_dataset_dimensions
from calliope.core.util.logging import log_time
from calliope.core.util.tools import apply_to_dict
from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope.backend.run import run as run_backend


def read_netcdf(path):
    """
    Return a Model object reconstructed from model data in a NetCDF file.

    """
    model_data = io.read_netcdf(path)
    return Model(config=None, model_data=model_data)


class Model(object):
    """
    A Calliope Model.

    """
    def __init__(self, config, model_data=None, *args, **kwargs):
        """
        Returns a new Model from either the path to a YAML model
        configuration file or a dict fully specifying the model.

        Parameters
        ----------
        config : str or dict or AttrDict
            If str, must be the path to a model configuration file.
            If dict or AttrDict, must fully specify the model.
        model_data : Dataset, optional
            Create a Model instance from a fully built model_data Dataset.
            This is only used if `config` is explicitly set to None
            and is primarily used to re-create a Model instance from
            a model previously saved to a NetCDF file.

        """
        self._timings = {}

        log_time(self._timings, 'model_creation', comment='Model: initialising')
        if config is not None:
            if isinstance(config, str):
                model_file = config
                config = AttrDict.from_yaml(model_file)
                config.config_path = model_file
            if isinstance(config, dict):
                model_run, debug_data = model_run_from_dict(config, *args, **kwargs)
            else:
                # expected input is a string pointing to a YAML file of the run
                # configuration or a dict/AttrDict in which the run and model
                # configurations are defined
                raise ValueError(
                    'Input configuration must either be a string or a dictionary.'
                )

            if 'scenario_file' in kwargs.keys():
                self._init_multiple_runs(model_run, debug_data)
            else:
                self._init_from_model_run(model_run, debug_data)
        elif model_data is not None:
            self._init_from_model_data(model_data)
        else:
            raise ValueError(
                'config (str or dict) or model_data (xarray Dataset) must be '
                'given to initialise model'
            )

        self.plot = plotting.ModelPlotMethods(self)

    def _init_from_model_run(self, model_run, debug_data):
        self._model_run = model_run
        self._debug_data = debug_data
        log_time(self._timings, 'model_run_creation', comment='Model: preprocessing stage 1 (model_run)')

        self._model_data_original = build_model_data(model_run)
        log_time(self._timings, 'model_data_original_creation', comment='Model: preprocessing stage 2 (model_data)')

        random_seed = self._model_run.get_key('model.random_seed', None)
        if random_seed:
            np.random.seed(seed=random_seed)

        # After setting the random seed, time clustering can take place
        time_config = model_run.model.get('time', None)
        if not time_config:
            _model_data = self._model_data_original
        else:
            _model_data = apply_time_clustering(
                self._model_data_original, model_run
            )
        self._model_data = final_timedimension_processing(_model_data)
        log_time(
            self._timings, 'model_data_creation',
            comment='Model: preprocessing complete',
            time_since_start=True
        )

        for var in self._model_data.data_vars:
            self._model_data[var].attrs['is_result'] = 0
        self.inputs = self._model_data.filter_by_attrs(is_result=0)

    def _init_multiple_runs(self, model_run, debug_data):
        self._model_run = model_run
        self._debug_data = debug_data
        log_time(self._timings, 'model_run_creation')

        model_data_original = []
        model_data = []

        for scenario_num, scenario in model_run.items():
            _model_data_original = build_model_data(scenario)
            if 'model.probability' in _model_data_original.attrs.keys():
                _model_data_original['probability'] = xr.DataArray(
                    data=[_model_data_original.attrs.pop('model.probability')],
                    dims='scenarios'
                )
            random_seed = scenario.get_key('model.random_seed', None)
            if random_seed:
                np.random.seed(seed=random_seed)
            time_config = scenario.model.get('time', None)
            if not time_config:
                _model_data = _model_data_original
            else:
                _model_data = apply_time_clustering(
                    _model_data_original, scenario
                )
            model_data_original.append(_model_data_original)
            model_data.append(final_timedimension_processing(_model_data))

        self._model_data_original = reorganise_dataset_dimensions(xr.concat(
            model_data_original, data_vars='different',
            dim=pd.Index(data=model_run.keys(), name='scenarios')
        ))

        self._model_data = reorganise_dataset_dimensions(xr.concat(
            model_data, data_vars='different',
            dim=pd.Index(data=model_run.keys(), name='scenarios')
        ))

        log_time(self._timings, 'model_data_creation', time_since_start=True)

        for var in self._model_data.data_vars:
            self._model_data[var].attrs['is_result'] = 0
        self.inputs = self._model_data.filter_by_attrs(is_result=0)

    def _init_from_model_data(self, model_data):
        self._model_run = None
        self._debug_data = None
        self._model_data = model_data
        self.inputs = self._model_data.filter_by_attrs(is_result=0)

        results = self._model_data.filter_by_attrs(is_result=1)
        if len(results.data_vars) > 0:
            self.results = results
        log_time(
            self._timings, 'model_data_loaded',
            comment='Model: loaded model_data',
            time_since_start=True
        )

    def save_commented_model_yaml(self, path):
        """
        Save a fully built and commented version of the model to a YAML file
        at the given ``path``. Comments in the file indicate where values
        were overridden. This is Calliope's internal representation of
        a model directly before the model_data xarray.Dataset is built,
        and can be useful for debugging possible issues in the model
        formulation.

        """
        # README: currently based on ruamel.yaml 0.15 which is a mix of old
        # and new API - possibly needs a bit of rewriting once ruamel.yaml
        # has progressed a bit further
        yaml = ruamel_yaml.YAML()

        model_run_debug = self._model_run.copy()
        del model_run_debug['timeseries_data']  # Can't be serialised!

        # Turn sets in model_run into lists for YAML serialization
        for k, v in model_run_debug.sets.items():
            model_run_debug.sets[k] = list(v)

        debug_comments = self._debug_data['comments']
        debug_yaml = yaml.load(yaml.dump(model_run_debug.as_dict()))
        for k in debug_comments.model_run.keys_nested():
            v = debug_comments.model_run.get_key(k)
            if v:
                keys = k.split('.')
                apply_to_dict(debug_yaml, keys[:-1], 'yaml_add_eol_comment', (v, keys[-1]))

        dumper = ruamel_yaml.dumper.RoundTripDumper
        dumper.ignore_aliases = lambda self, data: True

        with open(path, 'w') as f:
            ruamel_yaml.dump(
                debug_yaml, f,
                Dumper=dumper, default_flow_style=False
            )

    def run(self, force_rerun=False, scenario=None, **kwargs):
        """
        Run the model.

        Parameters
        ----------
        force_rerun : bool, default = False
            If True, any existing results will be overwritten
        scenario : str, int, or None, default = None
            If not None and model._model_data has a scenario dimension,
            the model will be sliced on that scenario before being sent to the
            backend. ``scenario`` must match an entry in the scenario dimension.

        Additional kwargs are passed to the backend.

        """

        # Check that results exist and are non-empty
        if hasattr(self, 'results') and self.results.data_vars and not force_rerun:
            raise exceptions.ModelError(
                'This model object already has results. '
                'Use model.run(force_rerun=True) to force'
                'the results to be overwritten with a new run.'
            )

        if (self._model_data.attrs['run.mode'] == 'operate' and
                not self._model_data.attrs['allow_operate_mode']):
            raise exceptions.ModelError(
                'Unable to run this model in operational mode, probably because '
                'there exist non-uniform timesteps (e.g. from time masking)'
            )

        if scenario is not None:
            if 'scenarios' not in self._model_data.dims:
                raise exceptions.ModelError(
                    'Unable to slice on scenario {} as the model data does not '
                    'have a scenario dimension'
                )
            elif scenario not in self._model_data.scenarios.values:
                raise KeyError('Scenario {} not in model_data scenarios dimension')
            else:
                model_data = self._model_data.loc[{'scenarios': [scenario]}]
        else:
            if ('scenarios' in self._model_data.dims
                    and self._model_data.attrs['run.mode'] != 'scenario_plan'):
                raise exceptions.ModelError(
                    'Unable to run this model in {} mode, as there are multiple '
                    'scenarios defined. Run in `scenario_plan` mode or select a '
                    'scenario on running the model `run(scenrio=...)`'
                    .format(self._model_data.attrs['run.mode'])
                )
            else:
                model_data = self._model_data

        results, self._backend_model, interface = run_backend(
            model_data, self._timings, **kwargs
        )

        # Add additional post-processed result variables to results
        if results.attrs.get('termination_condition', None) == 'optimal':
            results = postprocess.postprocess_model_results(
                results, self._model_data, self._timings
            )

        for var in results.data_vars:
            results[var].attrs['is_result'] = 1
        if scenario is not None:
            results = reorganise_dataset_dimensions(
                results.loc[{'scenarios': scenario}].expand_dims('scenarios')
            )
            self._model_data.merge(results, inplace=True)
        else:
            self._model_data.update(results)
        self._model_data.attrs.update(results.attrs)

        if 'run_solution_returned' in self._timings.keys():
            self._model_data.attrs['solution_time'] = (
                self._timings['run_solution_returned'] -
                self._timings['run_start']).total_seconds()
            self._model_data.attrs['time_finished'] = (
                self._timings['run_solution_returned'].strftime('%Y-%m-%d %H:%M:%S')
            )

        self.results = self._model_data.filter_by_attrs(is_result=1)

        self.backend = interface(self)

    def get_formatted_array(self, var):
        """
        Return an xr.DataArray with locs, techs, and carriers as
        separate dimensions.

        Parameters
        ----------
        var : str
            Decision variable for which to return a DataArray.

        """
        if var not in self._model_data.data_vars:
            raise KeyError("Variable {} not in Model data".format(var))

        return split_loc_techs(self._model_data[var])

    def to_netcdf(self, path):
        """
        Save complete model data (inputs and, if available, results)
        to a NetCDF file at the given ``path``.

        """
        io.save_netcdf(self._model_data, path)

    def to_csv(self, path, dropna=True):
        """
        Save complete model data (inputs and, if available, results)
        as a set of CSV files to the given ``path``.

        Parameters
        ----------
        dropna : bool, optional
            If True (default), NaN values are dropped when saving,
            resulting in significantly smaller CSV files.

        """
        io.save_csv(self._model_data, path, dropna)

    def info(self):
        info_strings = []
        model_name = self._model_data.attrs.get('model.name', 'None')
        info_strings.append('Model name:   {}'.format(model_name))
        msize = '{locs} locations, {techs} technologies, {times} timesteps'.format(
            locs=len(self._model_data.coords.get('locs', [])),
            techs=(
                len(self._model_data.coords.get('techs_non_transmission', [])) +
                len(self._model_data.coords.get('techs_transmission_names', []))
            ),
            times=len(self._model_data.coords.get('timesteps', [])))
        info_strings.append('Model size:   {}'.format(msize))
        return '\n'.join(info_strings)
