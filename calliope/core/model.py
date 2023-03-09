"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""
from __future__ import annotations

import logging
import warnings
from typing import Literal, Union, Optional
from contextlib import contextmanager
import os

import xarray as xr
import pandas as pd

import calliope
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
from calliope.backend import backends

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

    _BACKENDS: dict[str, type[backends.BackendModel]] = {
        "pyomo": backends.PyomoBackendModel
    }

    def __init__(
        self,
        config: Optional[Union[str, dict]],
        model_data: Optional[xr.Dataset] = None,
        debug: bool = False,
        *args,
        **kwargs,
    ):
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
        self._timings: dict = {}
        self.defaults: AttrDict
        self.model_config: AttrDict
        self.run_config: AttrDict
        self.component_config: AttrDict

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
        self._add_observed_dict("defaults", self._generate_default_dict())

        component_config = AttrDict.from_yaml(
            os.path.join(
                os.path.dirname(calliope.__file__), "config", "constraints.yaml"
            )
        )
        self._add_observed_dict("component_config", component_config)

        self.inputs = self._model_data.filter_by_attrs(is_result=0)
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
        self._add_observed_dict("component_config")

        self.inputs = self._model_data.filter_by_attrs(is_result=0)
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

    def _generate_default_dict(self) -> AttrDict:
        """Process input parameter default YAML configuration file into a dictionary of
        defaults that match parameter names in the processed model dataset
        (e.g., costs are prepended with `cost_`).

        Returns:
            AttrDict: Flat dictionary of `parameter_name`:`parameter_default` pairs.
        """
        raw_defaults = AttrDict.from_yaml(
            os.path.join(os.path.dirname(calliope.__file__), "config", "defaults.yaml")
        )
        default_tech_dict = raw_defaults.techs.default_tech
        default_cost_dict = {
            "cost_{}".format(k): v
            for k, v in default_tech_dict.costs.default_cost.items()
        }
        default_node_dict = {
            "available_area": raw_defaults.nodes.default_node.available_area
        }

        return AttrDict(
            {
                **default_tech_dict.constraints.as_dict(),
                **default_tech_dict.switches.as_dict(),
                **default_cost_dict,
                **default_node_dict,
            }
        )

    def build(self, backend_interface: Literal["pyomo"] = "pyomo") -> None:
        """Build description of the optimisation problem in the chosen backend interface.

        Args:
            backend_interface (Literal["pyomo"], optional):
                Backend interface in which to build the problem. Defaults to "pyomo".
        """

        with self.model_data_string_datetime():
            # FIXME: remove defaults as input arg (currently required by parsed "where" evalaution)
            # Maybe move it to being attached to the parsed component objects directly.
            backend = self._BACKENDS[backend_interface](
                defaults=self.defaults
            )  # type:ignore
            backend.add_all_parameters(self.inputs, self.defaults, self.run_config)
            log_time(
                logger,
                self._timings,
                "backend_parameters_generated",
                comment="Model: Generated optimisation problem parameters",
            )
            # The order of adding components matters!
            # 1. Variables, 2. Expressions, 3. Constraints, 4. Objectives
            for components in ["variables", "expressions", "constraints", "objectives"]:
                component = components.removesuffix("s")
                for name_, dict_ in self.component_config[components].items():
                    getattr(backend, f"add_{component}")(self.inputs, dict_, name_)
                log_time(
                    logger,
                    self._timings,
                    f"backend_{components}_generated",
                    comment=f"Model: Generated optimisation problem {components}",
                )

            self.backend = backend

    @contextmanager
    def model_data_string_datetime(self):
        """
        Temporarily turn model data input timeseries objects into strings with maximum
        resolution of minutes.
        """
        self._datetime_to_string()
        try:
            yield
        finally:
            self._string_to_datetime(self.inputs)

    def _datetime_to_string(self) -> None:
        """
        Convert model data inputs from datetime to string xarray dataarrays, to reduce the memory
        footprint of converting datetimes from numpy.datetime64 -> pandas.Timestamp
        when creating the pyomo model object.
        """
        datetime_data = set()
        for attr in ["coords", "data_vars"]:
            for set_name, set_data in getattr(self.inputs, attr).items():
                if set_data.dtype.kind == "M":
                    attrs = self.inputs[set_name].attrs
                    self.inputs[set_name] = self.inputs[set_name].dt.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    self.inputs[set_name].attrs = attrs
                    datetime_data.add((attr, set_name))

        self._datetime_data = datetime_data

        return None

    def _string_to_datetime(self, da: xr.Dataset) -> None:
        """
        Convert from string to datetime xarray dataarrays, reverting the process
        undertaken in `_datetime_to_string`. Operation is undertaken in-place.

        Without running `_datetime_to_string` earlier, this function will not function
        as expected since it will not be able to identify which coordinates should be
        converted to datetime format.

        Args:
            da (xr.Dataset): Dataset in which to convert timeseries data arrays.

        """
        for attr, set_name in self._datetime_data:
            if attr == "coords" and set_name in da:
                da.coords[set_name] = da[set_name].astype("datetime64[ns]")
            elif set_name in da:
                da[set_name] = xr.apply_ufunc(
                    pd.to_datetime, da[set_name], keep_attrs=True
                )
        return None

    def solve(self, force_rerun: bool = False, warmstart: bool = False) -> None:
        """
        Run the built optimisation problem.

        Args:
            force_rerun (bool, optional):
                If ``force_rerun`` is True, any existing results will be overwritten.
                Defaults to False.
            warmstart (bool, optional):
                If True and the optimisation problem has already been run in this session
                (i.e., `force_rerun` is not True), the next optimisation will be run with
                decision variables initially set to their previously optimal values.
                If the optimisation problem is similar to the previous run, this can
                decrease the solution time.
                Warmstart will not work with some solvers (e.g., CBC, GLPK).
                Defaults to False.

        Raises:
            exceptions.ModelError: Optimisation problem must already be built.
            exceptions.ModelError: Cannot run the model if there are already results loaded, unless `force_rerun` is True.
            exceptions.ModelError: Some preprocessing steps will stop a run mode of "operate" from being possible.
        """
        # Check that results exist and are non-empty
        if not hasattr(self, "backend"):
            raise exceptions.ModelError(
                "You must build the optimisation problem (`.build()`) "
                "before you can run it."
            )

        if hasattr(self, "results"):
            if self.results.data_vars and not force_rerun:
                raise exceptions.ModelError(
                    "This model object already has results. "
                    "Use model.run(force_rerun=True) to force"
                    "the results to be overwritten with a new run."
                )
            else:
                to_drop = self.results.data_vars
        else:
            to_drop = []

        if (
            self.run_config["mode"] == "operate"
            and not self._model_data.attrs["allow_operate_mode"]
        ):
            raise exceptions.ModelError(
                "Unable to run this model in operational mode, probably because "
                "there exist non-uniform timesteps (e.g. from time masking)"
            )

        termination_condition = self.backend.solve(
            solver=self.run_config["solver"],
            solver_io=self.run_config.get("solver_io", None),
            solver_options=self.run_config.get("solver_options", None),
            save_logs=self.run_config.get("save_logs", None),
            warmstart=warmstart,
        )

        # Add additional post-processed result variables to results
        if termination_condition in ["optimal", "feasible"]:
            results = self.backend.load_results()
            self._string_to_datetime(results)
            results = postprocess_results.postprocess_model_results(
                results, self._model_data, self._timings
            )
        else:
            results = xr.Dataset()

        self._model_data = self._model_data.drop_vars(to_drop)

        self._model_data.attrs.update(results.attrs)
        self._model_data.attrs["termination_condition"] = termination_condition
        self._model_data = xr.merge(
            [results, self._model_data], compat="override", combine_attrs="no_conflicts"
        )
        self._add_model_data_methods()

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
            "future version. Use `model.results[var]` instead.",
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
