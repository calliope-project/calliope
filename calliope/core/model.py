"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""
import logging
import warnings
import textwrap
from typing import TypedDict, Callable, TypeVar, Union, Optional
from contextlib import contextmanager
import os
import datetime

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
from calliope.core.util.observed_dict import UpdateObserverDict
from calliope import exceptions
from calliope.backend.run import run as run_backend
from calliope.backend.parsing import (
    ParsedConstraint,
    ParsedVariable,
    ParsedObjective,
    ParsedExpression,
)
from calliope.backend import backends

logger = logging.getLogger(__name__)


T = TypeVar(
    "T",
    bound=Union[ParsedVariable, ParsedConstraint, ParsedObjective, ParsedExpression],
)


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

    BACKENDS: dict[str, type[backends.BackendModel]] = {
        "pyomo": backends.PyomoBackendModel
    }

    DEFAULTS = AttrDict.from_yaml(
        os.path.join(os.path.dirname(calliope.__file__), "config", "defaults.yaml")
    )
    BACKEND_COMPONENTS = AttrDict.from_yaml(
        os.path.join(os.path.dirname(calliope.__file__), "config", "constraints.yaml")
    )

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
        parsed_variables = self._generate_parsing_components(
            unparsed=self.component_config["variables"],
            parse_class=ParsedVariable,
        )
        parsed_expression = self._generate_parsing_components(
            unparsed=self.component_config["expressions"],
            parse_class=ParsedExpression,
        )
        parsed_constraints = self._generate_parsing_components(
            unparsed=self.component_config["constraints"],
            parse_class=ParsedConstraint,
        )
        parsed_objectives = self._generate_parsing_components(
            unparsed=self.component_config["objectives"],
            parse_class=ParsedObjective,
        )
        self.parsed_components: backends.ParsedComponents = {
            "variables": parsed_variables,
            "expressions": parsed_expression,
            "constraints": parsed_constraints,
            "objectives": parsed_objectives,
        }

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
        self.component_config = UpdateObserverDict(
            initial_dict=self.BACKEND_COMPONENTS,
            name="component_config",
            observer=self._model_data,
        )
        default_tech_dict = self.DEFAULTS.techs.default_tech
        default_cost_dict = {
            "cost_{}".format(k): v
            for k, v in default_tech_dict.costs.default_cost.items()
        }
        default_node_dict = {
            "available_area": self.DEFAULTS.nodes.default_node.available_area
        }

        defaults = AttrDict(
            {
                **default_tech_dict.constraints.as_dict(),
                **default_tech_dict.switches.as_dict(),
                **default_cost_dict,
                **default_node_dict,
            }
        )
        self.defaults = UpdateObserverDict(
            initial_dict=defaults,
            name="defaults",
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
        self.model_config = UpdateObserverDict(
            initial_yaml_string=self._model_data.attrs.get("model_config", "{}"),
            name="model_config",
            observer=self._model_data,
        )
        self.run_config = UpdateObserverDict(
            initial_yaml_string=self._model_data.attrs.get("run_config", "{}"),
            name="run_config",
            observer=self._model_data,
        )
        self.subsets = UpdateObserverDict(
            initial_yaml_string=self._model_data.attrs.get("subsets", "{}"),
            name="subsets",
            observer=self._model_data,
            flat=True,
        )
        self.defaults = UpdateObserverDict(
            initial_yaml_string=self._model_data.attrs.get("defaults", "{}"),
            name="defaults",
            observer=self._model_data,
        )
        self.component_config = UpdateObserverDict(
            initial_yaml_string=self._model_data.attrs.get("component_config", "{}"),
            name="component_config",
            observer=self._model_data,
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

    def _generate_parsing_components(
        self, unparsed: dict, parse_class: type[T]
    ) -> dict[str, T]:

        parsed_components: dict[str, T] = dict()
        for component_name, component_config in unparsed.items():
            parsed_ = parse_class(component_config, component_name)
            parsed_components.update({component_name: parsed_})
        return parsed_components

    def build(self, backend_interface: str = "pyomo") -> None:


        with self.model_data_string_datetime():
            backend = self.BACKENDS[backend_interface](  # type: ignore
                parsed_components=self.parsed_components, defaults=self.defaults
            )
            backend.generate_backend_dataset(self._model_data, self.defaults, self.run_config)

            for parsed_variable in self.parsed_components["variables"].values():
                backend.dataset[parsed_variable.name] = backend.add_variable(
                    self._model_data, parsed_variable
                )
            for parsed_expression in self.parsed_components["expressions"].values():
                backend.dataset[parsed_expression.name] = backend.add_expression(
                    self._model_data, parsed_expression
                )

            for parsed_constraint in self.parsed_components["constraints"].values():
                backend.add_constraint(self._model_data, parsed_constraint)

            for parsed_objective in self.parsed_components["objectives"].values():
                backend.dataset[parsed_objective.name] = backend.add_objective(
                    self._model_data, parsed_objective
                )

            self.backend = backend

    @contextmanager
    def model_data_string_datetime(self):
        self._datetime_to_string()
        try:
            yield
        finally:
            self._string_to_datetime(self._model_data)

    def _datetime_to_string(self) -> None:
        """
        Convert from datetime to string xarray dataarrays, to reduce the memory
        footprint of converting datetimes from numpy.datetime64 -> pandas.Timestamp
        when creating the pyomo model object.

        """
        datetime_data = set()
        for attr in ["coords", "data_vars"]:
            for set_name, set_data in getattr(self._model_data, attr).items():
                if set_data.dtype.kind == "M":
                    attrs = self._model_data[set_name].attrs
                    self._model_data[set_name] = self._model_data[set_name].dt.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    self._model_data[set_name].attrs = attrs
                    datetime_data.add((attr, set_name))

        self._datetime_data = datetime_data

        return None

    def _string_to_datetime(self, da: xr.DataArray) -> None:
        """
        Convert from string to datetime xarray dataarrays, reverting the process
        undertaken in datetime_to_string

        """
        for attr, set_name in self._datetime_data:
            if attr == "coords" and set_name in da:
                da.coords[set_name] = da[set_name].astype("datetime64[ns]")
            elif set_name in da:
                da[set_name] = xr.apply_ufunc(
                    pd.to_datetime, da[set_name], keep_attrs=True
                )
        return None

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
