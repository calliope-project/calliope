# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Implements the core Model class."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from calliope import _version, backend, exceptions, io, postprocess, preprocess
from calliope.attrdict import AttrDict
from calliope.preprocess.model_data import ModelDataFactory
from calliope.schemas import CalliopeAttrs, ModelStructure, config_schema
from calliope.util.logging import log_time

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel

LOGGER = logging.getLogger(__name__)


def read_netcdf(path: str | Path) -> Model:
    """Return a Model object reconstructed from model data in a NetCDF file.

    Args:
        path (str | Path): Path to Calliope model NetCDF file.

    Returns:
        Model: Calliope Model instance.
    """
    model_data = io.read_netcdf(path)
    return Model(
        model_data["inputs"],
        CalliopeAttrs(**model_data["attrs"].attrs),
        model_data["results"],
    )


def read_yaml(
    file: str | Path,
    scenario: str | None = None,
    override_dict: dict | None = None,
    math_dict: dict | None = None,
    data_table_dfs: dict[str, pd.DataFrame] | None = None,
    **kwargs,
) -> Model:
    """Return a Model object reconstructed from a model defined in YAML + CSV files.

    Args:
        file (str | Path):
            Path to core model YAML file.
            If defining your model in multiple YAML files, they should be listed in the core model YAML file `import` list.
        scenario (str | None, optional):
            Comma delimited string of pre-defined `scenarios` to apply to the model.
            Defaults to None.
        override_dict (dict | None, optional):
            Additional overrides to apply to `config`.
            These will be applied *after* applying any defined `scenario` overrides.
            Defaults to None.
        math_dict (dict | None, optional):
            Additional math definitions to apply after loading the math paths.
            Defaults to None.
        data_table_dfs (dict[str, pd.DataFrame] | None, optional):
            Model definition `data_table` entries can reference in-memory pandas DataFrames.
            The referenced data must be supplied here as a dictionary of those DataFrames.
            Defaults to None.
        **kwargs: initialisation overrides.

    Returns:
        Model: Calliope Model instance.
    """
    raw_data = io.read_rich_yaml(file)
    return read_dict(
        raw_data, scenario, override_dict, math_dict, data_table_dfs, file, **kwargs
    )


def read_dict(
    model_definition: dict,
    scenario: str | None = None,
    override_dict: dict | None = None,
    math_dict: dict | None = None,
    data_table_dfs: dict[str, pd.DataFrame] | None = None,
    definition_path: str | Path | None = None,
    **kwargs,
):
    """Return a Model object reconstructed from a model definition dictionary loaded into memory.

    Args:
        model_definition (dict): Model definition YAML loaded into memory.
        scenario (str | None, optional):
            Comma delimited string of pre-defined `scenarios` to apply to the model.
            Defaults to None.
        override_dict (dict | None, optional):
            Additional overrides to apply to `config`.
            These will be applied *after* applying any defined `scenario` overrides.
            Defaults to None.
        math_dict (dict | None, optional):
            Additional math definitions to apply after loading the math paths.
            Defaults to None.
        data_table_dfs (dict[str, pd.DataFrame] | None, optional):
            Model definition `data_table` entries can reference in-memory pandas DataFrames.
            The referenced data must be supplied here as a dictionary of those DataFrames.
            Defaults to None.
        definition_path (Path | None): If given, the path relative to which all path references in `model_definition` will be taken.
        **kwargs: initialisation overrides.
    """
    model_def = preprocess.prepare_model_definition(
        model_definition, scenario, override_dict, math_dict, definition_path, **kwargs
    )
    log_time(
        LOGGER,
        model_def.runtime.timings,
        "preprocess_start",
        comment="Model: preprocessing data",
    )
    model_data_factory = ModelDataFactory(
        model_def.config.init,
        AttrDict(model_def.definition.model_dump(exclude_defaults=True)),
        model_def.math.init,
        definition_path,
        data_table_dfs,
    )
    model_data_factory.build()
    model_data_factory.clean()
    model_def = model_def.update({"math.build": model_data_factory.math.model_dump()})
    return Model(inputs=model_data_factory.dataset, attrs=model_def, _reentry=False)


class Model(ModelStructure):
    """A Calliope Model."""

    _TS_OFFSET = pd.Timedelta(1, unit="nanoseconds")

    def __init__(
        self,
        inputs: xr.Dataset,
        attrs: CalliopeAttrs,
        results: xr.Dataset | None = None,
        _reentry: bool = True,
        **kwargs,
    ) -> None:
        """Returns a instantiated Calliope Model.

        Args:
            inputs (xr.Dataset): Input dataset.
            attrs (CalliopeAttrs): Model attributes & properties.
            results (xr.Dataset | None, optional):
                Results dataset from another Calliope Model with compatible math formulation.
                Defaults to None.
            _reentry (bool, optional):
                Specifies model math and configuration must be reinitialised.
                Should only be set to `False` if this is the first time the model has been instantiated.
                Defaults to True.
            **kwargs:
                initialisation keyword arguments
        """
        self.inputs: xr.Dataset
        self.results: xr.Dataset = xr.Dataset() if results is None else results
        self.backend: BackendModel

        self.definition = attrs.definition
        self.config = attrs.config
        self.math = attrs.math
        self.runtime = attrs.runtime

        self._start_window_idx: int = 0
        self._is_built: bool = False
        self._is_solved: bool = False if results is None else True

        if _reentry:
            # Data may come from a previous run. Update math and clean inputs.
            log_time(
                LOGGER,
                self.runtime.timings,
                "preprocess_start",
                comment="Model: preprocessing data (reentry)",
            )
            self.config = self.config.update({"init": kwargs})
            model_data_factory = ModelDataFactory(
                self.config.init, inputs, self.math.init
            )
            model_data_factory.clean()

            self.inputs = model_data_factory.dataset
            self.math = self.math.update(
                {"build": model_data_factory.math.model_dump()}
            )
        else:
            # First time the model has been created. No need for cleanups.
            self.inputs = inputs

        self._check_versions()
        log_time(
            LOGGER,
            self.runtime.timings,
            "init_complete",
            comment="Model: initialisation complete",
        )

    def _check_versions(self) -> None:
        """Check the initialised and defined calliope version."""
        version_def = self.config.init.calliope_version
        version_init = self.runtime.calliope_version_initialised

        if not _version.__version__.startswith(version_init):
            exceptions.warn(
                f"Model was initialised with calliope version {version_init}, "
                f"but you are running {_version.__version__}. Proceed with caution!"
            )

        if version_def is not None and not version_init.startswith(version_def):
            exceptions.warn(
                f"Model configuration specifies calliope version {version_def}, "
                f"but you are running {version_init}. Proceed with caution!"
            )

    @property
    def name(self) -> str | None:
        """Get the model name."""
        return self.config.init.name

    @property
    def is_built(self) -> bool:
        """Get built status."""
        return self._is_built

    @property
    def is_solved(self) -> bool:
        """Get solved status."""
        return self._is_solved

    @property
    def all_attrs(self) -> CalliopeAttrs:
        """Get all model attributes as a CalliopeAttrs object."""
        return CalliopeAttrs(
            **{
                k: getattr(self, k).model_dump()
                for k in CalliopeAttrs.model_fields.keys()
            }
        )

    def dump_all_attrs(self) -> dict:
        """Dump of all class pydantic model attributes as a single dictionary."""
        return self.all_attrs.model_dump()

    def build(self, force: bool = False, **kwargs) -> None:
        """Build description of the optimisation problem in the chosen backend interface.

        Args:
            force (bool, optional):
                If ``force`` is True, any existing results will be overwritten.
                Defaults to False.
            **kwargs: build configuration overrides.
        """
        if self._is_built and not force:
            raise exceptions.ModelError(
                "This model object already has a built optimisation problem. Use model.build(force=True) "
                "to force the existing optimisation problem to be overwritten with a new one."
            )
        log_time(
            LOGGER,
            self.runtime.timings,
            "build_start",
            comment="Model: backend build starting",
        )

        self.config = self.config.update({"build": kwargs})

        mode = self.config.init.mode
        if mode == "operate":
            backend_input = self._prepare_operate_mode_inputs(self.config.build.operate)
        else:
            backend_input = self.inputs

        self.backend = backend.get_model_backend(
            self.config.build, backend_input, self.math.build
        )
        self.backend.add_optimisation_components()

        log_time(
            LOGGER,
            self.runtime.timings,
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

        # Check that results exist and are non-empty
        if self.results.data_vars and not force:
            raise exceptions.ModelError(
                "This model object already has results. "
                "Use model.solve(force=True) to force"
                "the results to be overwritten with a new run."
            )

        self.config = self.config.update({"solve": kwargs})

        self.backend.shadow_prices.track_constraints(self.config.solve.shadow_prices)

        mode = self.config.init.mode
        log_time(
            LOGGER,
            self.runtime.timings,
            "solve_start",
            comment=f"Optimisation model | starting model in {mode} mode.",
        )
        if mode == "operate":
            results = self._solve_operate(self.config.solve)
        elif mode == "spores":
            results = self._solve_spores(self.config.solve)
        else:
            results = self.backend._solve(self.config.solve, warmstart=warmstart)

        log_time(
            LOGGER,
            self.runtime.timings,
            "solver_exit",
            time_since_solve_start=True,
            comment="Backend: solver finished running",
        )

        # Add additional post-processed result variables to results
        if results.attrs["termination_condition"] in ["optimal", "feasible"]:
            results = postprocess.postprocess_model_results(results, self)

        self.math = self.math.update({"build": self.backend.math.model_dump()})
        self.runtime = self.runtime.update(
            {"termination_condition": results.attrs.pop("termination_condition")}
        )

        log_time(
            LOGGER,
            self.runtime.timings,
            "postprocess_complete",
            time_since_solve_start=True,
            comment="Postprocessing: ended",
        )

        log_time(
            LOGGER,
            self.runtime.timings,
            "solve_complete",
            time_since_solve_start=True,
            comment="Backend: model solve completed",
        )
        self.results = results

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
        io.save_netcdf(self.inputs, "inputs", "w", path)
        io.save_netcdf(self.results, "results", "a", path)
        io.save_netcdf(xr.Dataset(attrs=self.dump_all_attrs()), "attrs", "a", path)

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
            io.save_csv(self.results, "results", path, dropna, allow_overwrite=True)
        else:
            exceptions.warn("No results available, saving inputs only.")

        io.to_yaml(self.dump_all_attrs(), path=Path(path) / "attrs.yaml")

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

    def _prepare_operate_mode_inputs(
        self, operate_config: config_schema.BuildOperate
    ) -> xr.Dataset:
        """Slice the input data to just the length of operate mode time horizon.

        Args:
            operate_config (config.BuildOperate): operate mode configuration options.

        Returns:
            xr.Dataset: Slice of input data.
        """
        if self.config.init.time_cluster is not None:
            # TODO: Consider moving this to validator in config schema
            raise exceptions.ModelError(
                "Unable to run this model in operate (i.e. dispatch) mode because time clustering is in use"
            )
        self.inputs.coords["windowsteps"] = pd.date_range(
            self.inputs.timesteps[0].item(),
            self.inputs.timesteps[-1].item(),
            freq=operate_config.window,
        )
        horizonsteps = self.inputs.coords["windowsteps"] + pd.Timedelta(
            operate_config.horizon
        )
        # We require an offset because pandas / xarray slicing is _inclusive_ of both endpoints
        # where we only want it to be inclusive of the left endpoint.
        # Except in the last time horizon, where we want it to include the right endpoint.
        clipped_horizonsteps = horizonsteps.clip(
            max=self.inputs.timesteps[-1] + self._TS_OFFSET
        ).drop_vars("timesteps")
        self.inputs.coords["horizonsteps"] = clipped_horizonsteps - self._TS_OFFSET
        sliced_inputs = self.inputs.sel(
            timesteps=slice(
                self.inputs.windowsteps[self._start_window_idx],
                self.inputs.horizonsteps[self._start_window_idx],
            )
        )
        return sliced_inputs

    def _solve_operate(self, solver_config: config_schema.Solve) -> xr.Dataset:
        """Solve in operate (i.e. dispatch) mode.

        Optimisation is undertaken iteratively for slices of the timeseries, with
        some data being passed between slices.

        Args:
            solver_config (config_schema.Solve): Calliope Solver configuration object.

        Returns:
            xr.Dataset: Results dataset.
        """
        if self._start_window_idx != 0:
            LOGGER.info("Optimisation model | Resetting model to first time window.")
            self._start_window_idx = 0
            self.build(force=True)

        LOGGER.info("Optimisation model | Running first time window.")

        iteration_results = self.backend._solve(solver_config, warmstart=False)

        results_list = []

        for idx, windowstep in enumerate(self.inputs.windowsteps[1:]):
            self._start_window_idx = idx + 1
            windowstep_as_string = windowstep.dt.strftime("%Y-%m-%d %H:%M:%S").item()
            LOGGER.info(
                f"Optimisation model | Running time window starting at {windowstep_as_string}."
            )
            results_list.append(
                iteration_results.sel(
                    timesteps=slice(None, windowstep - self._TS_OFFSET)
                )
            )
            previous_iteration_results = results_list[-1]
            horizonstep = self.inputs.horizonsteps.sel(windowsteps=windowstep)
            new_inputs = self.inputs.sel(
                timesteps=slice(windowstep, horizonstep)
            ).drop_vars(["horizonsteps", "windowsteps"], errors="ignore")
            new_ts = new_inputs.timesteps.copy()

            if len(new_inputs.timesteps) != len(iteration_results.timesteps):
                LOGGER.info(
                    "Optimisation model | Reaching the end of the timeseries. "
                    "Re-building model with shorter time horizon."
                )
                self.build(force=True)
            else:
                new_inputs.coords["timesteps"] = self.backend.inputs.coords["timesteps"]
                for param_name, param_data in new_inputs.data_vars.items():
                    if (
                        "timesteps" in param_data.dims
                        and param_name in self.backend.parameters
                        and not param_data.equals(self.backend.inputs[param_name])
                    ):
                        self.backend.update_input(param_name, param_data)
                        self.backend.inputs[param_name] = param_data

            if "storage" in iteration_results:
                self.backend.update_input(
                    "storage_initial",
                    self._recalculate_storage_initial(previous_iteration_results),
                )

            iteration_results = self.backend._solve(solver_config, warmstart=False)
            iteration_results.coords["timesteps"] = new_ts

        results_list.append(iteration_results.sel(timesteps=slice(windowstep, None)))
        results = xr.concat(results_list, dim="timesteps", combine_attrs="drop")
        results.attrs["termination_condition"] = ",".join(
            set(
                result.attrs["termination_condition"]
                for result in results_list
                if "termination_condition" in result.attrs
            )
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

        new_initial_storage = end_storage / self.inputs.storage_cap.where(
            lambda x: x > 0
        )
        return new_initial_storage

    def _solve_spores(self, solver_config: config_schema.Solve) -> xr.Dataset:
        """Solve in spores (i.e. modelling to generate alternatives - MGA) mode.

        Optimisation is undertaken iteratively after setting the total monetary cost of the system.
        Technology "spores" costs are updated between iterations.

        Returns:
            xr.Dataset: Results dataset.
        """
        LOGGER.info("Optimisation model | Resetting SPORES parameters.")
        self.backend.update_input(
            "spores_score", self.math.build.parameters["spores_score"].default
        )

        spores_config: config_schema.SolveSpores = solver_config.spores

        latest_spore: int = 0
        if not spores_config.use_latest_results:
            LOGGER.info("Optimisation model | Running baseline model.")
            self.backend.set_objective(self.config.build.objective)
            baseline_results = self.backend._solve(solver_config, warmstart=False)
            self._spores_save_model(baseline_results, spores_config, 0)
        elif "spores" in self.results.dims:
            latest_spore = self.results.spores.max().item()
            LOGGER.info(
                f"Optimisation model | Restarting SPORES from SPORE {latest_spore} results."
            )
            baseline_results = self.results.sel(spores=latest_spore).drop_vars("spores")
            self.backend.update_input(
                "spores_score", baseline_results.spores_score_cumulative
            )
        else:
            LOGGER.info("Optimisation model | Using existing baseline model results.")
            baseline_results = self.results.copy()

        if latest_spore >= spores_config.number:
            raise exceptions.ModelError(
                f"Cannot restart SPORES from SPORE {latest_spore} as it is greater or equal "
                f"to the configured number of SPORES to run ({spores_config.number})."
            )

        spore_range: list[int] = [
            i for i in range(latest_spore + 1, spores_config.number + 1)
        ]

        if not baseline_results:
            raise exceptions.ModelError(
                "Cannot run SPORES without baseline results. "
                "This issue may be caused by an infeasible baseline model."
                "Ensure your baseline model can solve successfully by running it in `base` mode."
            )

        base_cost_default = self.math.build.parameters["spores_baseline_cost"].default
        constraining_cost = baseline_results.get(
            "spores_baseline_cost_tracked", self.inputs.get("spores_baseline_cost")
        )
        if not constraining_cost or constraining_cost == base_cost_default:
            # Update the slack-cost backend value based on the calculated minimum feasible system design cost
            constraining_cost = baseline_results[self.config.build.objective]
        self.backend.update_input("spores_baseline_cost", constraining_cost)

        self.backend.set_objective("min_spores")
        # We store the results from each iteration in the `results_list` to later concatenate into a single dataset.
        results_list: list[xr.Dataset] = [baseline_results]
        LOGGER.info(
            f"Optimisation model | Running SPORES with `{spores_config.scoring_algorithm}` scoring algorithm."
        )

        for spore in spore_range:
            LOGGER.info(f"Optimisation model | Running SPORE {spore}.")
            self._spores_update_model(results_list, spores_config)

            iteration_results = self.backend._solve(solver_config, warmstart=False)
            self._spores_save_model(iteration_results, spores_config, spore)

            if not iteration_results:
                exceptions.warn(
                    f"Stopping SPORES run after SPORE {spore} due to model infeasibility."
                )
                break

            results_list.append(iteration_results)

        spores_dim = pd.Index(
            [latest_spore, *spore_range[: len(results_list) - 1]], name="spores"
        )
        results = xr.concat(results_list, dim=spores_dim, combine_attrs="drop")
        if latest_spore > 0 and spores_config.use_latest_results:
            results = xr.concat(
                [self.results, results.drop_sel(spores=latest_spore)],
                dim="spores",
                combine_attrs="no_conflicts",
            )
        results.attrs["termination_condition"] = ",".join(
            set(
                result.attrs["termination_condition"]
                for result in results_list
                if "termination_condition" in result.attrs
            )
        )

        return results

    def _spores_save_model(
        self, results: xr.Dataset, spores_config: config_schema.SolveSpores, spore: int
    ) -> None:
        """Save results per SPORE.

        Args:
            results (xr.Dataset): Results to save.
            spores_config (config_schema.SolveSpores): SPORES configuration.
            spore (int): Spore number.

        """
        if spores_config.save_per_spore_path is None:
            return None

        if results.attrs["termination_condition"] in ["optimal", "feasible"]:
            log_time(
                LOGGER,
                self.runtime.timings,
                "solve_complete",
                time_since_solve_start=True,
                comment=f"Optimisation model | SPORE {spore} complete",
            )
            results = postprocess.postprocess_model_results(results, self)

            spores_config.save_per_spore_path.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Optimisation model | Saving SPORE {spore} to file.")
            outpath = spores_config.save_per_spore_path / f"spore_{spore}.nc"

            io.save_netcdf(results.expand_dims(spores=[spore]), "results", "w", outpath)
            io.save_netcdf(
                xr.Dataset(attrs=self.dump_all_attrs()), "attrs", "a", outpath
            )
            if spore == 0:
                io.save_netcdf(
                    self.inputs.expand_dims(spores=[spore]), "inputs", "a", outpath
                )

        else:
            LOGGER.info(
                "Optimisation model | Infeasible or unbounded problem | "
                f"No SPORE {spore} results to save to file."
            )

    def _spores_update_model(
        self,
        all_previous_results: list[xr.Dataset],
        spores_config: config_schema.SolveSpores,
    ):
        """Assign SPORES scores for the next iteration of the model run.

        Algorithms applied are based on those introduced in <https://doi.org/10.1016/j.apenergy.2023.121002>.

        Args:
            all_previous_results (list[xr.Dataset]):
                A list of all previous iterations.
                 This includes the baseline results, which will be the first item in the list.
            spores_config (config_schema.SolveSpores):
                The SPORES configuration.
        """

        def _score_integer(
            spores_techs: xr.DataArray, old_score: xr.DataArray
        ) -> xr.DataArray:
            """Integer scoring algorithm."""
            previous_cap = latest_results["flow_cap"].where(spores_techs)

            # Make sure that penalties are applied only to non-negligible deployments of capacity
            min_relevant_size = spores_config.score_threshold_factor * previous_cap.max(
                ["nodes", "techs"]
            )

            new_score = (
                # Where capacity was deployed more than the minimal relevant size, assign an integer penalty (score)
                previous_cap.where(previous_cap > min_relevant_size)
                .clip(min=1000, max=1000)
                .fillna(0)
                .where(spores_techs)
            )
            return new_score + old_score

        def _score_relative_deployment(
            spores_techs: xr.DataArray, old_score: xr.DataArray
        ) -> xr.DataArray:
            """Relative deployment scoring algorithm."""
            previous_cap = latest_results["flow_cap"]
            if (
                "flow_cap_max" not in self.inputs
                or (self.inputs["flow_cap_max"].where(spores_techs) == np.inf).any()
            ):
                raise exceptions.BackendError(
                    "Cannot score SPORES with `relative_deployment` when `flow_cap_max` is undefined for some or all tracked technologies."
                )
            relative_cap = previous_cap / self.inputs["flow_cap_max"]

            new_score = (
                # Make sure that penalties are applied only to non-negligible relative capacities
                relative_cap.where(relative_cap > spores_config.score_threshold_factor)
                .fillna(0)
                .where(spores_techs)
            )
            return new_score + old_score

        def _score_random(
            spores_techs: xr.DataArray, old_score: xr.DataArray
        ) -> xr.DataArray:
            """Random scoring algorithm."""
            previous_cap = latest_results["flow_cap"].where(spores_techs)
            new_score = (
                previous_cap.fillna(0)
                .where(
                    previous_cap.isnull(),
                    other=np.random.choice([0, 1000], size=(previous_cap.shape)),
                )
                .where(spores_techs)
            )

            return new_score + old_score

        def _score_evolving_average(
            spores_techs: xr.DataArray, old_score: xr.DataArray
        ) -> xr.DataArray:
            """Evolving average scoring algorithm."""
            previous_cap = latest_results["flow_cap"]
            evolving_average = sum(
                results["flow_cap"] for results in all_previous_results
            ) / len(all_previous_results)

            relative_change = abs(evolving_average - previous_cap) / evolving_average

            if relative_change.sum() == 0:
                # first iteration
                new_score = _score_integer(spores_techs, old_score)
            else:
                # If capacity is exactly the same as the average, we give the relative difference an arbitrarily small value
                # which will give it a _large_ score since we take the reciprocal of the change.
                cleaned_relative_change = (
                    relative_change.clip(min=0.001).fillna(0).where(spores_techs)
                )
                # Any zero values that make their way through to the scoring are kept as zero after taking the reciprocal.
                new_score = (cleaned_relative_change**-1).where(
                    cleaned_relative_change > 0, other=0
                )

            # We don't add on the old score in this algorithm
            return new_score

        latest_results = all_previous_results[-1]
        allowed_methods: dict[
            config_schema.SPORES_SCORING_OPTIONS,
            Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
        ] = {
            "integer": _score_integer,
            "relative_deployment": _score_relative_deployment,
            "random": _score_random,
            "evolving_average": _score_evolving_average,
        }

        # Filter for technologies of interest
        spores_techs = (
            self.inputs.get(
                spores_config.tracking_parameter, xr.DataArray(True)
            ).notnull()
            & self.inputs.definition_matrix
        )
        old_score = self.backend.get_parameter(
            "spores_score", as_backend_objs=False
        ).fillna(0)
        new_score = allowed_methods[spores_config.scoring_algorithm](
            spores_techs, old_score
        )

        self.backend.update_input("spores_score", new_score)
