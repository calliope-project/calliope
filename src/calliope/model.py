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
from calliope.preprocess.model_data import ModelDataFactory
from calliope.schemas import config_schema, model_def_schema
from calliope.util.logging import log_time
from calliope.util.schema import MODEL_SCHEMA, extract_from_schema

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel

LOGGER = logging.getLogger(__name__)


def read_netcdf(path):
    """Return a Model object reconstructed from model data in a NetCDF file."""
    model_data = io.read_netcdf(path)
    model = Model(model_definition=model_data["inputs"])
    model.results = model_data["results"]
    return model


class Model:
    """A Calliope Model."""

    _TS_OFFSET = pd.Timedelta(1, unit="nanoseconds")
    ATTRS_SAVED = ("applied_math", "config", "def_path")

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
        self.config: config_schema.CalliopeConfig
        self.defaults: AttrDict
        self.user_math: AttrDict
        self.applied_math: preprocess.CalliopeMath
        self.backend: BackendModel
        self.def_path: str | None = None
        self._start_window_idx: int = 0
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
                self.def_path = str(model_definition)
            self._init_from_model_definition(
                model_definition, scenario, override_dict, data_table_dfs, **kwargs
            )

        self.inputs.attrs["timestamp_model_creation"] = timestamp_model_creation
        version_def = self.inputs.attrs["calliope_version_defined"]
        version_init = self.inputs.attrs["calliope_version_initialised"]
        if version_def is not None and not version_init.startswith(version_def):
            exceptions.warn(
                f"Model configuration specifies calliope version {version_def}, "
                f"but you are running {version_init}. Proceed with caution!"
            )

    @property
    def name(self):
        """Get the model name."""
        return self.inputs.attrs["name"]

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
        # Ensure model definition is correct
        model_def_schema.CalliopeModelDef(**model_def_full)

        log_time(
            LOGGER,
            self._timings,
            "model_run_creation",
            comment="Model: preprocessing stage 1 (model_run)",
        )
        model_config = config_schema.CalliopeConfig(**model_def_full.pop("config"))

        param_metadata = {"default": extract_from_schema(MODEL_SCHEMA, "default")}
        attributes = {
            "calliope_version_defined": model_config.init.calliope_version,
            "calliope_version_initialised": calliope.__version__,
            "applied_overrides": applied_overrides,
            "scenario": scenario,
            "defaults": param_metadata["default"],
        }
        # FIXME-config: remove config input once model_def_full uses pydantic
        model_data_factory = ModelDataFactory(
            model_config.init,
            model_def_full,
            self.def_path,
            data_table_dfs,
            attributes,
            param_metadata,
        )
        model_data_factory.build()

        self.inputs = model_data_factory.dataset

        log_time(
            LOGGER,
            self._timings,
            "model_data_creation",
            comment="Model: preprocessing stage 2 (model_data)",
        )

        self._model_data.attrs["name"] = model_config.init.name
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
            self.config = config_schema.CalliopeConfig(**model_data.attrs.pop("config"))

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
        self.inputs.attrs["timestamp_build_start"] = log_time(
            LOGGER,
            self._timings,
            "build_start",
            comment="Model: backend build starting",
        )
        math_dict = self.user_math.copy()
        if add_math_dict is not None:
            math_dict.union(add_math_dict)

        self.backend = backend.manager.get_backend_model(
            self.inputs, math_dict, **kwargs
        )
        self.backend.add_optimisation_components()

        self.applied_math = self.backend.math

        self.inputs.attrs["timestamp_build_complete"] = log_time(
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
        if not self.is_built:
            raise exceptions.ModelError(
                "You must build the optimisation problem (`.build()`) "
                "before you can run it."
            )

        if hasattr(self, "results"):  # Check that results exist and are non-empty
            if self.results.data_vars and not force:
                raise exceptions.ModelError(
                    "This model object already has results. "
                    "Use model.solve(force=True) to force"
                    "the results to be overwritten with a new run."
                )

        self.config = self.config.update({"solve": kwargs})

        shadow_prices = self.config.solve.shadow_prices
        self.backend.shadow_prices.track_constraints(shadow_prices)

        mode = self.config.build.mode
        self._model_data.attrs["timestamp_solve_start"] = log_time(
            LOGGER,
            self._timings,
            "solve_start",
            comment=f"Optimisation model | starting model in {mode} mode.",
        )
        if mode == "operate":
            results = self._solve_operate(**self.config.solve.model_dump())
        else:
            results = self.backend._solve(
                warmstart=warmstart, **self.config.solve.model_dump()
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

        self.results = results

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

    def run(self, force_rerun=False):
        """Run the model.

        If ``force_rerun`` is True, any existing results will be overwritten.
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
            if attr == "config":
                saved_attrs[attr] = self.config.model_dump()
            elif not isinstance(getattr(self, attr), str | list | None):
                saved_attrs[attr] = dict(getattr(self, attr))
            else:
                saved_attrs[attr] = getattr(self, attr)

        io.save_netcdf(self.inputs, "inputs", path, **saved_attrs)
        io.save_netcdf(self.results, "results", path)

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
        io.save_csv(self.inputs, "inputs", path, dropna, allow_overwrite)

        if self.results:
            io.save_csv(self.results, "results", path, dropna, allow_overwrite)
        else:
            exceptions.warn("No results available, saving inputs only.")

    def info(self) -> str:
        """Generate basic description of the model, combining its name and a rough indication of the model size.

        Returns:
            str: Basic description of the model.
        """
        info_strings = []
        model_name = self.name
        info_strings.append(f"Model name:   {model_name}")
        msize = dict(self.inputs.dims)
        msize_exists = self.inputs.definition_matrix.sum()
        info_strings.append(
            f"Model size:   {msize} ({msize_exists.item()} valid node:tech:carrier combinations)"
        )
        return "\n".join(info_strings)
