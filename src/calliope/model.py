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
from calliope import backend, config, exceptions, io, preprocess
from calliope.attrdict import AttrDict
from calliope.postprocess import postprocess as postprocess_results
from calliope.preprocess.data_tables import DataTable
from calliope.preprocess.model_data import ModelDataFactory
from calliope.util.logging import log_time
from calliope.util.schema import (
    CONFIG_SCHEMA,
    MODEL_SCHEMA,
    extract_from_schema,
    validate_dict,
)

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
    ATTRS_SAVED = ("applied_math", "config")

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
        self.config: config.CalliopeConfig
        self.defaults: AttrDict
        self.applied_math: preprocess.CalliopeMath
        self.backend: BackendModel
        self._is_built: bool = False
        self._is_solved: bool = False

        # try to set logging output format assuming python interactive. Will
        # use CLI logging format if model called from CLI
        timestamp_model_creation = log_time(
            LOGGER, self._timings, "model_creation", comment="Model: initialising"
        )
        if isinstance(model_definition, xr.Dataset):
            if kwargs:
                raise exceptions.ModelError(
                    "Cannot apply initialisation configuration overrides when loading data from an xarray Dataset."
                )
            self._init_from_model_data(model_definition)
        else:
            if not isinstance(model_definition, dict):
                # Only file definitions allow relative files.
                kwargs["def_path"] = str(model_definition)
            self._init_from_model_definition(
                model_definition, scenario, override_dict, data_table_dfs, **kwargs
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

    def _init_from_model_definition(
        self,
        model_definition: dict | str | Path,
        scenario: str | None,
        override_dict: dict | None,
        data_table_dfs: dict[str, pd.DataFrame] | None,
        **kwargs,
    ) -> None:
        """Initialise the model using pre-processed YAML files and optional dataframes/dicts.

        Args:
            model_definition (calliope.AttrDict): preprocessed model configuration.
            scenario (str | None): scenario specified by users
            override_dict (dict | None): overrides to apply after scenarios.
            data_table_dfs (dict[str, pd.DataFrame] | None): files with additional model information.
            **kwargs: initialisation overrides.
        """
        (model_def_full, applied_overrides) = preprocess.prepare_model_definition(
            model_definition, scenario, override_dict
        )
        model_def_full.union({"config.init": kwargs}, allow_override=True)
        # First pass to check top-level keys are all good. FIXME-config: remove after pydantic is ready
        validate_dict(model_def_full, CONFIG_SCHEMA, "Model definition")

        log_time(
            LOGGER,
            self._timings,
            "model_run_creation",
            comment="Model: preprocessing stage 1 (model_run)",
        )
        model_config = config.CalliopeConfig(**model_def_full.pop("config"))
        init_config = model_config.init

        param_metadata = {"default": extract_from_schema(MODEL_SCHEMA, "default")}
        attributes = {
            "calliope_version_defined": init_config.calliope_version,
            "calliope_version_initialised": calliope.__version__,
            "applied_overrides": applied_overrides,
            "scenario": scenario,
            "defaults": param_metadata["default"],
        }
        data_tables: list[DataTable] = []
        for table_name, table_dict in model_def_full.pop("data_tables", {}).items():
            data_tables.append(
                DataTable(table_name, table_dict, data_table_dfs, init_config.def_path)
            )
        model_data_factory = ModelDataFactory(
            init_config, model_def_full, data_tables, attributes, param_metadata
        )
        model_data_factory.build()

        self._model_data = model_data_factory.dataset

        log_time(
            LOGGER,
            self._timings,
            "model_data_creation",
            comment="Model: preprocessing stage 2 (model_data)",
        )

        self._model_data.attrs["name"] = init_config.name

        # Unlike at the build and solve phases, we store the init config overrides in the main model config.
        model_config.init = init_config  # FIXME-config: unnecessary?
        self.config = model_config

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
        if "applied_math" in model_data.attrs:
            self.applied_math = preprocess.CalliopeMath.from_dict(
                model_data.attrs.pop("applied_math")
            )
        if "config" in model_data.attrs:
            self.config = config.CalliopeConfig(**model_data.attrs.pop("config"))
            self.config.update(model_data.attrs.pop("config_kwarg_overrides"))

        self._model_data = model_data

        if self.results:
            self._is_solved = True

        log_time(
            LOGGER,
            self._timings,
            "model_data_loaded",
            comment="Model: loaded model_data",
        )

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

        build_config = self.config.update({"build": kwargs}).build
        mode = build_config.mode
        if mode == "operate":
            if not self._model_data.attrs["allow_operate_mode"]:
                raise exceptions.ModelError(
                    "Unable to run this model in operate (i.e. dispatch) mode, probably because "
                    "there exist non-uniform timesteps (e.g. from time clustering)"
                )
            backend_input = self._prepare_operate_mode_inputs(build_config.operate)
        else:
            backend_input = self._model_data

        init_math_list = [] if build_config.ignore_mode_math else [mode]
        end_math_list = [] if add_math_dict is None else [add_math_dict]
        full_math_list = init_math_list + build_config.add_math + end_math_list
        LOGGER.debug(f"Math preprocessing | Loading math: {full_math_list}")
        model_math = preprocess.CalliopeMath(full_math_list, self.config.init.def_path)

        self.backend = backend.get_model_backend(
            build_config, backend_input, model_math
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
        if not self.is_built:
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

        solve_config = self.config.update({"solve": kwargs}).solve
        # FIXME: find a way to avoid overcomplicated passing of settings between modes
        mode = self.config.update(self.config.applied_keyword_overrides).build.mode
        self._model_data.attrs["timestamp_solve_start"] = log_time(
            LOGGER,
            self._timings,
            "solve_start",
            comment=f"Optimisation model | starting model in {mode} mode.",
        )

        shadow_prices = solve_config.shadow_prices
        self.backend.shadow_prices.track_constraints(shadow_prices)

        if mode == "operate":
            results = self._solve_operate(**solve_config.model_dump())
        else:
            results = self.backend._solve(
                warmstart=warmstart, **solve_config.model_dump()
            )

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
                results, self._model_data, self.config.solve.zero_threshold
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
        saved_attrs["config_kwarg_overrides"] = self.config.applied_keyword_overrides

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
        self, operate_config: config.BuildOperate
    ) -> xr.Dataset:
        """Slice the input data to just the length of operate mode time horizon.

        Args:
            operate_config (config.BuildOperate): operate mode configuration options.

        Returns:
            xr.Dataset: Slice of input data.
        """
        self._model_data.coords["windowsteps"] = pd.date_range(
            self.inputs.timesteps[0].item(),
            self.inputs.timesteps[-1].item(),
            freq=operate_config.window,
        )
        horizonsteps = self._model_data.coords["windowsteps"] + pd.Timedelta(
            operate_config.horizon
        )
        # We require an offset because pandas / xarray slicing is _inclusive_ of both endpoints
        # where we only want it to be inclusive of the left endpoint.
        # Except in the last time horizon, where we want it to include the right endpoint.
        clipped_horizonsteps = horizonsteps.clip(
            max=self._model_data.timesteps[-1] + self._TS_OFFSET
        ).drop_vars("timesteps")
        self._model_data.coords["horizonsteps"] = clipped_horizonsteps - self._TS_OFFSET
        sliced_inputs = self._model_data.sel(
            timesteps=slice(
                self._model_data.windowsteps[operate_config.start_window_idx],
                self._model_data.horizonsteps[operate_config.start_window_idx],
            )
        )
        if operate_config.use_cap_results:
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
            self.build(force=True, **self.config.build.applied_keyword_overrides)

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
                build_kwargs = AttrDict(self.config.build.applied_keyword_overrides)
                build_kwargs.set_key("operate.start_window_idx", idx + 1)
                self.build(force=True, **build_kwargs)
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
