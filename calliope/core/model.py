"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""
import logging
import warnings

import numpy as np

from calliope.postprocess import results as postprocess_results
from calliope.postprocess import plotting
from calliope.core import io
from calliope.preprocess import (
    model_run_from_yaml,
    model_run_from_dict,
)
from calliope.preprocess.model_data import ModelDataFactory
from calliope.core.attrdict import AttrDict
from calliope.core.util.logging import log_time
from calliope.core.util.tools import apply_to_dict
from calliope.core.util.observed_dict import UpdateObserverDict
from calliope import exceptions
from calliope.backend.run import run as run_backend

logger = logging.getLogger(__name__)


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

    def __init__(self, config, model_data=None, debug=False, *args, **kwargs):
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
        # try to set logging output format assuming python interactive. Will
        # use CLI logging format if model called from CLI
        log_time(logger, self._timings, "model_creation", comment="Model: initialising")
        if isinstance(config, str):
            model_run, debug_data = model_run_from_yaml(config, *args, **kwargs)
            self._init_from_model_run(model_run, debug_data, debug)
        elif isinstance(config, dict):
            model_run, debug_data = model_run_from_dict(config, *args, **kwargs)
            self._init_from_model_run(model_run, debug_data, debug)
        elif model_data is not None and config is None:
            self._init_from_model_data(model_data)
        else:
            # expected input is a string pointing to a YAML file of the run
            # configuration or a dict/AttrDict in which the run and model
            # configurations are defined
            raise ValueError(
                "Input configuration must either be a string or a dictionary."
            )
        self._check_future_deprecation_warnings()

        self.plot = plotting.ModelPlotMethods(self)

    def _init_from_model_run(self, model_run, debug_data, debug):
        self._model_run = model_run
        log_time(
            logger,
            self._timings,
            "model_run_creation",
            comment="Model: preprocessing stage 1 (model_run)",
        )

        model_data_factory = ModelDataFactory(model_run)
        (
            model_data_pre_clustering,
            model_data,
            data_pre_time,
            stripped_keys,
        ) = model_data_factory()

        self._model_data_pre_clustering = model_data_pre_clustering
        self._model_data = model_data
        if debug:
            self._debug_data = debug_data
            self._model_data_pre_time = data_pre_time
            self._model_data_stripped_keys = stripped_keys
        self.inputs = self._model_data.filter_by_attrs(is_result=0)
        log_time(
            logger,
            self._timings,
            "model_data_original_creation",
            comment="Model: preprocessing stage 2 (model_data)",
        )

        # Ensure model and run attributes of _model_data update themselves
        model_config = {
            k: v for k, v in model_run.get("model", {}).items() if k != "file_allowed"
        }
        self.model_config = UpdateObserverDict(
            initial_dict=model_config, name="model_config", observer=self._model_data
        )
        self.run_config = UpdateObserverDict(
            initial_dict=model_run.get("run", {}),
            name="run_config",
            observer=self._model_data,
        )
        self.subsets = UpdateObserverDict(
            initial_dict=model_run.get("subsets").as_dict_flat(),
            name="subsets",
            observer=self._model_data,
        )

        log_time(
            logger,
            self._timings,
            "model_data_creation",
            comment="Model: preprocessing complete",
        )

    def _init_from_model_data(self, model_data):
        if "_model_run" in model_data.attrs:
            self._model_run = AttrDict.from_yaml_string(model_data.attrs["_model_run"])
            del model_data.attrs["_model_run"]

        if "_debug_data" in model_data.attrs:
            self._debug_data = AttrDict.from_yaml_string(
                model_data.attrs["_debug_data"]
            )
            del model_data.attrs["_debug_data"]

        self._model_data = model_data
        self.inputs = self._model_data.filter_by_attrs(is_result=0)
        self.model_config = UpdateObserverDict(
            initial_yaml_string=model_data.attrs.get("model_config", "{}"),
            name="model_config",
            observer=self._model_data,
        )
        self.run_config = UpdateObserverDict(
            initial_yaml_string=model_data.attrs.get("run_config", "{}"),
            name="run_config",
            observer=self._model_data,
        )
        self.subsets = UpdateObserverDict(
            initial_yaml_string=model_data.attrs.get("subsets", "{}"),
            name="subsets",
            observer=self._model_data,
            flat=True,
        )

        results = self._model_data.filter_by_attrs(is_result=1)
        if len(results.data_vars) > 0:
            self.results = results
        log_time(
            logger,
            self._timings,
            "model_data_loaded",
            comment="Model: loaded model_data",
        )

    def run(self, force_rerun=False, **kwargs):
        """
        Run the model. If ``force_rerun`` is True, any existing results
        will be overwritten.

        Additional kwargs are passed to the backend.

        """
        # Check that results exist and are non-empty
        if hasattr(self, "results") and self.results.data_vars and not force_rerun:
            raise exceptions.ModelError(
                "This model object already has results. "
                "Use model.run(force_rerun=True) to force"
                "the results to be overwritten with a new run."
            )

        if (
            self.run_config["mode"] == "operate"
            and not self._model_data.attrs["allow_operate_mode"]
        ):
            raise exceptions.ModelError(
                "Unable to run this model in operational mode, probably because "
                "there exist non-uniform timesteps (e.g. from time masking)"
            )

        results, self._backend_model, interface = run_backend(
            self._model_data, self._timings, **kwargs
        )

        # Add additional post-processed result variables to results
        if results.attrs.get("termination_condition", None) in ["optimal", "feasible"]:
            results = postprocess_results.postprocess_model_results(
                results, self._model_data, self._timings
            )

        for var in results.data_vars:
            results[var].attrs["is_result"] = 1

        self._model_data.update(results)
        self._model_data.attrs.update(results.attrs)

        self.results = self._model_data.filter_by_attrs(is_result=1)

        self.backend = interface(self)

    def get_formatted_array(self, var):
        """
        Return an xr.DataArray with nodes, techs, and carriers as
        separate dimensions.

        Parameters
        ----------
        var : str
            Decision variable for which to return a DataArray.

        """
        warnings.warn(
            "get_formatted_array() is deprecated and will be removed in a "
            "future version. Use `model.results.variable` instead.",
            DeprecationWarning,
        )
        if var not in self._model_data.data_vars:
            raise KeyError("Variable {} not in Model data".format(var))

        return self._model_data[var]

    def to_netcdf(self, path):
        """
        Save complete model data (inputs and, if available, results)
        to a NetCDF file at the given ``path``.

        """
        io.save_netcdf(self._model_data, path, model=self)

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

    def to_lp(self, path):
        """
        Save built model to LP format at the given ``path``. If the backend
        model has not been built yet, it is built prior to saving.
        """
        io.save_lp(self, path)

    def info(self):
        info_strings = []
        model_name = self.model_config.get("name", "None")
        info_strings.append("Model name:   {}".format(model_name))
        msize = "{nodes} nodes, {techs} technologies, {times} timesteps".format(
            nodes=len(self._model_data.coords.get("nodes", [])),
            techs=(
                len(self._model_data.coords.get("techs_non_transmission", []))
                + len(self._model_data.coords.get("techs_transmission_names", []))
            ),
            times=len(self._model_data.coords.get("timesteps", [])),
        )
        info_strings.append("Model size:   {}".format(msize))
        return "\n".join(info_strings)

    def _check_future_deprecation_warnings(self):
        """
        Method for all FutureWarnings and DeprecationWarnings. Comment above each
        warning should specify Calliope version in which it was added, and the
        version in which it should be updated/removed.
        """
