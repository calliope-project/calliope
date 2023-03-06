"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""
import logging
import warnings
from typing import Optional

import xarray as xr

from calliope.postprocess import results as postprocess_results
from calliope.core import io
from calliope.preprocess import (
    model_run_from_yaml,
    model_run_from_dict,
)
from calliope.preprocess.model_data import ModelDataFactory
from calliope.core.attrdict import AttrDict
from calliope.core.util.logging import log_time
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

        self._add_observed_dict("model_config", model_config)
        self._add_observed_dict("run_config", model_run["run"])
        self._add_observed_dict("subsets", model_run["subsets"])

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
        self._add_model_data_methods()

        log_time(
            logger,
            self._timings,
            "model_data_loaded",
            comment="Model: loaded model_data",
        )

    def _add_model_data_methods(self):
        self.inputs = self._model_data.filter_by_attrs(is_result=0)
        self.results = self._model_data.filter_by_attrs(is_result=1)
        self._add_observed_dict("model_config")
        self._add_observed_dict("run_config")
        self._add_observed_dict("subsets")

        results = self._model_data.filter_by_attrs(is_result=1)
        if len(results.data_vars) > 0:
            self.results = results
        log_time(
            logger,
            self._timings,
            "model_data_loaded",
            comment="Model: loaded model_data",
        )

    def _add_observed_dict(self, name: str, dict_to_add: Optional[dict] = None) -> None:
        """
        Add the same dictionary as property of model object and an attribute of the model xarray dataset.

        Args:
            name (str):
                Name of dictionary which will be set as the model property name and (if necessary) the dataset attribute name.
            dict_to_add (Optional[dict], optional):
                If given, set as both the model property and the dataset attribute, otherwise set an existing dataset attribute as a model property of the same name. Defaults to None.

        Raises:
            exceptions.ModelError: If `dict_to_add` is not given, it must be an attribute of model data.
            TypeError: `dict_to_add` must be a dictionary.
        """
        if dict_to_add is None:
            try:
                dict_to_add = self._model_data.attrs[name]
            except KeyError:
                raise exceptions.ModelError(
                    f"Expected the model property `{name}` to be a dictionary attribute of the model dataset. If you are loading the model from a NetCDF file, ensure it is a valid Calliope model."
                )
        if not isinstance(dict_to_add, dict):
            raise TypeError(
                f"Attempted to add dictionary property `{name}` to model, but received argument of type `{type(dict_to_add).__name__}`"
            )
        else:
            dict_to_add = AttrDict(dict_to_add)
        self._model_data.attrs[name] = dict_to_add
        setattr(self, name, dict_to_add)

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

        results, self._backend_model, self._backend_model_opt, interface = run_backend(
            self._model_data, self._timings, **kwargs
        )

        # Add additional post-processed result variables to results
        if results.attrs.get("termination_condition", None) in ["optimal", "feasible"]:
            results = postprocess_results.postprocess_model_results(
                results, self._model_data, self._timings
            )
        self._model_data.attrs.update(results.attrs)
        self._model_data = xr.merge(
            [results, self._model_data], compat="override", combine_attrs="no_conflicts"
        )
        self._add_model_data_methods()

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
