# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Implements the core Model class."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

import calliope
from calliope import backend, exceptions, io, preprocess
from calliope.attrdict import AttrDict
from calliope.postprocess import postprocess as postprocess_results
from calliope.preprocess.data_tables import DataTable
from calliope.preprocess.model_data import ModelDataFactory
from calliope.util.logging import log_time
from calliope.util.schema import (
    CONFIG_SCHEMA,
    MODEL_SCHEMA,
    extract_from_schema,
    update_then_validate_config,
    validate_dict,
)
from calliope.util.tools import climb_template_tree, relative_path

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel

LOGGER = logging.getLogger(__name__)


def read_netcdf(path):
    """Return a Model object reconstructed from model data in a NetCDF file."""
    model_data = io.read_netcdf(path)
    return Model(model_definition=model_data)


class Model:
    """A Calliope Model."""

    _TS_OFFSET = pd.Timedelta(1, unit="nanoseconds")
    ATTRS_SAVED = ("_def_path", "applied_math")

    def __init__(
        self,
        model_definition: str | Path | dict | xr.Dataset,
        scenario: str | None = None,
        override_dict: dict | None = None,
        data_table_dfs: dict[str, pd.DataFrame] | None = None,
        **kwargs,
    ):
        """Returns a new Model from YAML model configuration files or a fully specified dictionary.

        Args:
            model_definition (str | Path | dict | xr.Dataset):
                If str or Path, must be the path to a model configuration file.
                If dict or AttrDict, must fully specify the model.
                If an xarray dataset, must be a valid calliope model.
            scenario (str | None, optional):
                Comma delimited string of pre-defined `scenarios` to apply to the model.
                Defaults to None.
            override_dict (dict | None, optional):
                Additional overrides to apply to `config`.
                These will be applied *after* applying any defined `scenario` overrides.
                Defaults to None.
            data_table_dfs (dict[str, pd.DataFrame] | None, optional):
                Model definition `data_table` entries can reference in-memory pandas DataFrames.
                The referenced data must be supplied here as a dictionary of those DataFrames.
                Defaults to None.
            **kwargs: initialisation overrides.
        """
        self._timings: dict = {}
        self.config: AttrDict
        self.defaults: AttrDict
        self.applied_math: preprocess.CalliopeMath
        self._def_path: str | None = None
        self.backend: BackendModel
        self._is_built: bool = False
        self._is_solved: bool = False

        # try to set logging output format assuming python interactive. Will
        # use CLI logging format if model called from CLI
        timestamp_model_creation = log_time(
            LOGGER, self._timings, "model_creation", comment="Model: initialising"
        )
        if isinstance(model_definition, xr.Dataset):
            self._init_from_model_data(model_definition)
        else:
            if isinstance(model_definition, dict):
                model_def_dict = AttrDict(model_definition)
            else:
                self._def_path = str(model_definition)
                model_def_dict = AttrDict.from_yaml(model_definition)

            (model_def, applied_overrides) = preprocess.load_scenario_overrides(
                model_def_dict, scenario, override_dict, **kwargs
            )

            self._init_from_model_def_dict(
                model_def, applied_overrides, scenario, data_table_dfs
            )

        self._model_data.attrs["timestamp_model_creation"] = timestamp_model_creation
        version_def = self._model_data.attrs["calliope_version_defined"]
        version_init = self._model_data.attrs["calliope_version_initialised"]
        if version_def is not None and not version_init.startswith(version_def):
            exceptions.warn(
                f"Model configuration specifies calliope version {version_def}, "
                f"but you are running {version_init}. Proceed with caution!"
            )

    @property
    def name(self):
        """Get the model name."""
        return self._model_data.attrs["name"]

    @property
    def inputs(self):
        """Get model input data."""
        return self._model_data.filter_by_attrs(is_result=0)

    @property
    def results(self):
        """Get model result data."""
        return self._model_data.filter_by_attrs(is_result=1)

    @property
    def is_built(self):
        """Get built status."""
        return self._is_built

    @property
    def is_solved(self):
        """Get solved status."""
        return self._is_solved

    def _init_from_model_def_dict(
        self,
        model_definition: calliope.AttrDict,
        applied_overrides: str,
        scenario: str | None,
        data_table_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Initialise the model using pre-processed YAML files and optional dataframes/dicts.

        Args:
            model_definition (calliope.AttrDict): preprocessed model configuration.
            applied_overrides (str): overrides specified by users
            scenario (str | None): scenario specified by users
            data_table_dfs (dict[str, pd.DataFrame] | None, optional): files with additional model information. Defaults to None.
        """
        # First pass to check top-level keys are all good
        validate_dict(model_definition, CONFIG_SCHEMA, "Model definition")

        log_time(
            LOGGER,
            self._timings,
            "model_run_creation",
            comment="Model: preprocessing stage 1 (model_run)",
        )
        model_config = AttrDict(extract_from_schema(CONFIG_SCHEMA, "default"))
        model_config.union(model_definition.pop("config"), allow_override=True)

        init_config = update_then_validate_config("init", model_config)

        if init_config["time_cluster"] is not None:
            init_config["time_cluster"] = relative_path(
                self._def_path, init_config["time_cluster"]
            )

        param_metadata = {"default": extract_from_schema(MODEL_SCHEMA, "default")}
        attributes = {
            "calliope_version_defined": init_config["calliope_version"],
            "calliope_version_initialised": calliope.__version__,
            "applied_overrides": applied_overrides,
            "scenario": scenario,
            "defaults": param_metadata["default"],
        }
        templates = model_definition.get("templates", AttrDict())
        data_tables: list[DataTable] = []
        for table_name, table_dict in model_definition.pop("data_tables", {}).items():
            table_dict, _ = climb_template_tree(table_dict, templates, table_name)
            data_tables.append(
                DataTable(
                    init_config, table_name, table_dict, data_table_dfs, self._def_path
                )
            )

        model_data_factory = ModelDataFactory(
            init_config, model_definition, data_tables, attributes, param_metadata
        )
        model_data_factory.build()

        self._model_data = model_data_factory.dataset

        log_time(
            LOGGER,
            self._timings,
            "model_data_creation",
            comment="Model: preprocessing stage 2 (model_data)",
        )

        self._add_observed_dict("config", model_config)

        self._model_data.attrs["name"] = init_config["name"]
        log_time(
            LOGGER,
            self._timings,
            "model_preprocessing_complete",
            comment="Model: preprocessing complete",
        )

    def _init_from_model_data(self, model_data: xr.Dataset) -> None:
        """Initialise the model using a pre-built xarray dataset.

        This must be a Calliope-compatible dataset, usually a dataset from another Calliope model.

        Args:
            model_data (xr.Dataset):
                Model dataset with input parameters as arrays and configuration stored in the dataset attributes dictionary.
        """
        if "_def_path" in model_data.attrs:
            self._def_path = model_data.attrs.pop("_def_path")
        if "applied_math" in model_data.attrs:
            self.applied_math = preprocess.CalliopeMath.from_dict(
                model_data.attrs.pop("applied_math")
            )

        self._model_data = model_data
        self._add_model_data_methods()

        if self.results:
            self._is_solved = True

        log_time(
            LOGGER,
            self._timings,
            "model_data_loaded",
            comment="Model: loaded model_data",
        )

    def _add_model_data_methods(self):
        """Add observed data to `model`.

        1. Filter model dataset to produce views on the input/results data
        2. Add top-level configuration dictionaries simultaneously to the model data attributes and as attributes of this class.

        """
        self._add_observed_dict("config")

    def _add_observed_dict(self, name: str, dict_to_add: dict | None = None) -> None:
        """Add the same dictionary as property of model object and an attribute of the model xarray dataset.

        Args:
            name (str):
                Name of dictionary which will be set as the model property name and
                (if necessary) the dataset attribute name.
            dict_to_add (dict | None, optional):
                If given, set as both the model property and the dataset attribute,
                otherwise set an existing dataset attribute as a model property of the
                same name. Defaults to None.

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

    def build(
        self, force: bool = False, add_math_dict: dict | None = None, **kwargs
    ) -> None:
        """Build description of the optimisation problem in the chosen backend interface.

        Args:
            force (bool, optional):
                If ``force`` is True, any existing results will be overwritten.
                Defaults to False.
            add_math_dict (dict | None, optional):
                Additional math to apply on top of the YAML base / additional math files.
                Content of this dictionary will override any matching key:value pairs in the loaded math files.
            **kwargs: build configuration overrides.
        """
        if self._is_built and not force:
            raise exceptions.ModelError(
                "This model object already has a built optimisation problem. Use model.build(force=True) "
                "to force the existing optimisation problem to be overwritten with a new one."
            )
        self._model_data.attrs["timestamp_build_start"] = log_time(
            LOGGER,
            self._timings,
            "build_start",
            comment="Model: backend build starting",
        )

        backend_config = {**self.config["build"], **kwargs}
        mode = backend_config["mode"]
        if mode == "operate":
            if not self._model_data.attrs["allow_operate_mode"]:
                raise exceptions.ModelError(
                    "Unable to run this model in operate (i.e. dispatch) mode, probably because "
                    "there exist non-uniform timesteps (e.g. from time clustering)"
                )
            start_window_idx = backend_config.pop("start_window_idx", 0)
            backend_input = self._prepare_operate_mode_inputs(
                start_window_idx, **backend_config
            )
        else:
            backend_input = self._model_data

        init_math_list = [] if backend_config.get("ignore_mode_math") else [mode]
        end_math_list = [] if add_math_dict is None else [add_math_dict]
        full_math_list = init_math_list + backend_config["add_math"] + end_math_list
        LOGGER.debug(f"Math preprocessing | Loading math: {full_math_list}")
        model_math = preprocess.CalliopeMath(full_math_list, self._def_path)

        backend_name = backend_config.pop("backend")
        self.backend = backend.get_model_backend(
            backend_name, backend_input, model_math, **backend_config
        )
        self.backend.add_optimisation_components()

        self.applied_math = model_math

        self._model_data.attrs["timestamp_build_complete"] = log_time(
            LOGGER,
            self._timings,
            "build_complete",
            comment="Model: backend build complete",
        )
        self._is_built = True

    def solve(self, force: bool = False, warmstart: bool = False, **kwargs) -> None:
        """Solve the built optimisation problem.

        Args:
            force (bool, optional):
                If ``force`` is True, any existing results will be overwritten.
                Defaults to False.
            warmstart (bool, optional):
                If True and the optimisation problem has already been run in this session
                (i.e., `force` is not True), the next optimisation will be run with
                decision variables initially set to their previously optimal values.
                If the optimisation problem is similar to the previous run, this can
                decrease the solution time.
                Warmstart will not work with some solvers (e.g., CBC, GLPK).
                Defaults to False.
            **kwargs: solve configuration overrides.

        Raises:
            exceptions.ModelError: Optimisation problem must already be built.
            exceptions.ModelError: Cannot run the model if there are already results loaded, unless `force` is True.
            exceptions.ModelError: Some preprocessing steps will stop a run mode of "operate" from being possible.
        """
        # Check that results exist and are non-empty
        if not self._is_built:
            raise exceptions.ModelError(
                "You must build the optimisation problem (`.build()`) "
                "before you can run it."
            )

        if hasattr(self, "results"):
            if self.results.data_vars and not force:
                raise exceptions.ModelError(
                    "This model object already has results. "
                    "Use model.solve(force=True) to force"
                    "the results to be overwritten with a new run."
                )
            else:
                to_drop = self.results.data_vars
        else:
            to_drop = []

        run_mode = self.backend.inputs.attrs["config"]["build"]["mode"]
        self._model_data.attrs["timestamp_solve_start"] = log_time(
            LOGGER,
            self._timings,
            "solve_start",
            comment=f"Optimisation model | starting model in {run_mode} mode.",
        )

        solver_config = update_then_validate_config("solve", self.config, **kwargs)

        shadow_prices = solver_config.get("shadow_prices", [])
        self.backend.shadow_prices.track_constraints(shadow_prices)

        if run_mode == "operate":
            results = self._solve_operate(**solver_config)
        else:
            results = self.backend._solve(warmstart=warmstart, **solver_config)

        log_time(
            LOGGER,
            self._timings,
            "solver_exit",
            time_since_solve_start=True,
            comment="Backend: solver finished running",
        )

        # Add additional post-processed result variables to results
        if results.attrs["termination_condition"] in ["optimal", "feasible"]:
            results = postprocess_results.postprocess_model_results(
                results, self._model_data
            )

        log_time(
            LOGGER,
            self._timings,
            "postprocess_complete",
            time_since_solve_start=True,
            comment="Postprocessing: ended",
        )

        self._model_data = self._model_data.drop_vars(to_drop)

        self._model_data.attrs.update(results.attrs)
        self._model_data = xr.merge(
            [results, self._model_data], compat="override", combine_attrs="no_conflicts"
        )
        self._add_model_data_methods()

        self._model_data.attrs["timestamp_solve_complete"] = log_time(
            LOGGER,
            self._timings,
            "solve_complete",
            time_since_solve_start=True,
            comment="Backend: model solve completed",
        )

        self._is_solved = True

    def run(self, force_rerun=False, **kwargs):
        """Run the model.

        If ``force_rerun`` is True, any existing results will be overwritten.

        Additional kwargs are passed to the backend.
        """
        exceptions.warn(
            "`run()` is deprecated and will be removed in a "
            "future version. Use `model.build()` followed by `model.solve()`.",
            FutureWarning,
        )
        self.build(force=force_rerun)
        self.solve(force=force_rerun)

    def to_netcdf(self, path):
        """Save complete model data (inputs and, if available, results) to a NetCDF file at the given `path`."""
        saved_attrs = {}
        for attr in set(self.ATTRS_SAVED) & set(self.__dict__.keys()):
            if not isinstance(getattr(self, attr), str | list | None):
                saved_attrs[attr] = dict(getattr(self, attr))
            else:
                saved_attrs[attr] = getattr(self, attr)

        io.save_netcdf(self._model_data, path, **saved_attrs)

    def to_csv(
        self, path: str | Path, dropna: bool = True, allow_overwrite: bool = False
    ):
        """Save complete model data (inputs and, if available, results) as a set of CSV files to the given ``path``.

        Args:
            path (str | Path): file path to save at.
            dropna (bool, optional):
                If True, NaN values are dropped when saving, resulting in significantly smaller CSV files.
                Defaults to True
            allow_overwrite (bool, optional):
                If True, allow the option to overwrite the directory contents if it already exists.
                This will overwrite CSV files one at a time, so if the dataset has different arrays to the previous saved models, you will get a mix of old and new files.
                Defaults to False.

        """
        io.save_csv(self._model_data, path, dropna, allow_overwrite)

    def info(self) -> str:
        """Generate basic description of the model, combining its name and a rough indication of the model size.

        Returns:
            str: Basic description of the model.
        """
        info_strings = []
        model_name = self.name
        info_strings.append(f"Model name:   {model_name}")
        msize = dict(self._model_data.dims)
        msize_exists = self._model_data.definition_matrix.sum()
        info_strings.append(
            f"Model size:   {msize} ({msize_exists.item()} valid node:tech:carrier combinations)"
        )
        return "\n".join(info_strings)

    def _prepare_operate_mode_inputs(
        self, start_window_idx: int = 0, **config_kwargs
    ) -> xr.Dataset:
        """Slice the input data to just the length of operate mode time horizon.

        Args:
            start_window_idx (int, optional):
                Set the operate `window` to start at, based on integer index.
                This is used when re-initialising the backend model for shorter time horizons close to the end of the model period.
                Defaults to 0.
            **config_kwargs: kwargs related to operate mode configuration.

        Returns:
            xr.Dataset: Slice of input data.
        """
        window = config_kwargs["operate_window"]
        horizon = config_kwargs["operate_horizon"]
        self._model_data.coords["windowsteps"] = pd.date_range(
            self.inputs.timesteps[0].item(),
            self.inputs.timesteps[-1].item(),
            freq=window,
        )
        horizonsteps = self._model_data.coords["windowsteps"] + pd.Timedelta(horizon)
        # We require an offset because pandas / xarray slicing is _inclusive_ of both endpoints
        # where we only want it to be inclusive of the left endpoint.
        # Except in the last time horizon, where we want it to include the right endpoint.
        clipped_horizonsteps = horizonsteps.clip(
            max=self._model_data.timesteps[-1] + self._TS_OFFSET
        ).drop_vars("timesteps")
        self._model_data.coords["horizonsteps"] = clipped_horizonsteps - self._TS_OFFSET
        sliced_inputs = self._model_data.sel(
            timesteps=slice(
                self._model_data.windowsteps[start_window_idx],
                self._model_data.horizonsteps[start_window_idx],
            )
        )
        if config_kwargs.get("operate_use_cap_results", False):
            to_parameterise = extract_from_schema(MODEL_SCHEMA, "x-operate-param")
            if not self._is_solved:
                raise exceptions.ModelError(
                    "Cannot use plan mode capacity results in operate mode if a solution does not yet exist for the model."
                )
            for parameter in to_parameterise.keys():
                if parameter in self._model_data:
                    self._model_data[parameter].attrs["is_result"] = 0

        return sliced_inputs

    def _solve_operate(self, **solver_config) -> xr.Dataset:
        """Solve in operate (i.e. dispatch) mode.

        Optimisation is undertaken iteratively for slices of the timeseries, with
        some data being passed between slices.

        Returns:
            xr.Dataset: Results dataset.
        """
        if self.backend.inputs.timesteps[0] != self._model_data.timesteps[0]:
            LOGGER.info("Optimisation model | Resetting model to first time window.")
            self.build(
                force=True,
                **{"mode": "operate", **self.backend.inputs.attrs["config"]["build"]},
            )

        LOGGER.info("Optimisation model | Running first time window.")

        step_results = self.backend._solve(warmstart=False, **solver_config)

        results_list = []

        for idx, windowstep in enumerate(self._model_data.windowsteps[1:]):
            windowstep_as_string = windowstep.dt.strftime("%Y-%m-%d %H:%M:%S").item()
            LOGGER.info(
                f"Optimisation model | Running time window starting at {windowstep_as_string}."
            )
            results_list.append(
                step_results.sel(timesteps=slice(None, windowstep - self._TS_OFFSET))
            )
            previous_step_results = results_list[-1]
            horizonstep = self._model_data.horizonsteps.sel(windowsteps=windowstep)
            new_inputs = self.inputs.sel(
                timesteps=slice(windowstep, horizonstep)
            ).drop_vars(["horizonsteps", "windowsteps"], errors="ignore")

            if len(new_inputs.timesteps) != len(step_results.timesteps):
                LOGGER.info(
                    "Optimisation model | Reaching the end of the timeseries. "
                    "Re-building model with shorter time horizon."
                )
                self.build(
                    force=True,
                    start_window_idx=idx + 1,
                    **self.backend.inputs.attrs["config"]["build"],
                )
            else:
                self.backend._dataset.coords["timesteps"] = new_inputs.timesteps
                self.backend.inputs.coords["timesteps"] = new_inputs.timesteps
                for param_name, param_data in new_inputs.data_vars.items():
                    if "timesteps" in param_data.dims:
                        self.backend.update_parameter(param_name, param_data)
                        self.backend.inputs[param_name] = param_data

            if "storage" in step_results:
                self.backend.update_parameter(
                    "storage_initial",
                    self._recalculate_storage_initial(previous_step_results),
                )

            step_results = self.backend._solve(warmstart=False, **solver_config)

        results_list.append(step_results.sel(timesteps=slice(windowstep, None)))
        results = xr.concat(results_list, dim="timesteps", combine_attrs="no_conflicts")
        results.attrs["termination_condition"] = ",".join(
            set(result.attrs["termination_condition"] for result in results_list)
        )

        return results

    def _recalculate_storage_initial(self, results: xr.Dataset) -> xr.DataArray:
        """Calculate the initial level of storage devices for a new operate mode time slice.

        Based on storage levels at the end of the previous time slice.

        Args:
            results (xr.Dataset): Results from the previous time slice.

        Returns:
            xr.DataArray: `storage_initial` values for the new time slice.
        """
        end_storage = results.storage.isel(timesteps=-1).drop_vars("timesteps")

        new_initial_storage = end_storage / self.inputs.storage_cap
        return new_initial_storage
