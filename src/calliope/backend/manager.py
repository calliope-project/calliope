"""Calliope's optimisation backend module manager."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from calliope import exceptions
from calliope.backend import helper_functions, parsing
from calliope.backend.gurobi_backend_model import GurobiBackendModel
from calliope.backend.pyomo_backend_model import PyomoBackendModel
from calliope.exceptions import BackendError
from calliope.io import load_config
from calliope.preprocess import CalliopeMath
from calliope.schemas import config_schema
from calliope.util.schema import update_then_validate_config

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel

TS_OFFSET = pd.Timedelta(1, unit="nanoseconds")

LOGGER = logging.getLogger(__name__)


def get_backend_model(data: xr.Dataset, math_dict: dict, **kwargs) -> BackendModel:
    """Build backend model.

    Args:
        data (xr.Dataset): Input data.
        math_dict (dict): Math definition dictionary.
        **kwargs: build configuration options.

    Returns:
        BackendModel: Instantiated backend object.
    """
    config = update_then_validate_config("build", data.attrs["config"], **kwargs)
    math = prepare_math(math_dict, config.mode, config.ignore_mode_math)
    if config["pre_validate_math_strings"]:
        math.validate()
    updated_data = prepare_inputs(data, config, math)

    updated_data.attrs["config"]["build"] = config

    backend = get_model_backend(config["backend"], updated_data, math)
    return backend


def get_model_backend(name: str, data: xr.Dataset, math: CalliopeMath) -> BackendModel:
    """Assign a backend using the given configuration.

    Args:
        name (str): name of the backend to use.
        data (Dataset): model data for the backend.
        math (str | Path | None): Calliope math.
        **kwargs: backend keyword arguments corresponding to model.config.build.

    Raises:
        exceptions.BackendError: If invalid backend was requested.

    Returns:
        BackendModel: Initialized backend object.
    """
    match name:
        case "pyomo":
            return PyomoBackendModel(data, math)
        case "gurobi":
            return GurobiBackendModel(data, math)
        case _:
            raise BackendError(f"Incorrect backend '{name}' requested.")


def prepare_inputs(data: xr.Dataset, config: dict, math: CalliopeMath) -> xr.Dataset:
    """Prepare inputs.

    Args:
        data (xr.Dataset): Input dataset.
        config (dict): Build configuration.
        math (CalliopeMath): Math dictionary.

    Raises:
        exceptions.ModelError: Cannot run some models in operate mode.

    Returns:
        xr.Dataset: Prepared input dataset (copy of `data`).
    """
    backend_data = xr.Dataset(attrs=data.attrs)
    for dim_name, dim_def_dict in math.data["dims"].items():
        backend_data.coords[dim_name] = _get_dim(data, dim_name, dim_def_dict)
    for obj_type in ["parameters", "lookups"]:
        for array_name, array_def_dict in math.data[obj_type].items():
            backend_data[array_name] = _get_array(data, array_name, array_def_dict)
            backend_data[array_name].attrs["obj_type"] = obj_type

    _input_data_checks(backend_data)
    backend_data.attrs = deepcopy(data.attrs)
    if config["time_resample"] is not None:
        data = resample(backend_data, config["time_resample"], math)
    if config["mode"] == "operate":
        if not backend_data.attrs["allow_operate_mode"]:
            raise exceptions.ModelError(
                "Unable to run this model in operate (i.e. dispatch) mode, probably because "
                "there exist non-uniform timesteps (e.g. from time clustering)"
            )
        start_window_idx = config.pop("start_window_idx", 0)
        backend_data = prepare_operate_mode_inputs(
            backend_data, start_window_idx, **config
        )
    return backend_data


def _get_dim(data: xr.Dataset, dim_name: str, dim_def_dict: dict) -> xr.DataArray:
    if dim_name in data:
        dim = data[dim_name].astype(dim_def_dict["type"])
        dim.attrs = dim_def_dict
    else:
        dim = xr.DataArray(np.nan, attrs=dim_def_dict)
    return dim


def _get_array(data: xr.Dataset, array_name: str, array_def_dict: dict) -> xr.DataArray:
    if array_name in data:
        array = data[array_name]
        if array.dims:
            array = array.groupby(data[array_name].notnull()).map(
                _update_dtypes, dtype=array_def_dict["type"]
            )
        else:
            array = array.astype(array_def_dict["type"])

        array.attrs = array_def_dict
    else:
        array = xr.DataArray(np.nan, attrs=array_def_dict)
    return array


def _update_dtypes(da: xr.DataArray, *, dtype: str, **kwargs) -> xr.DataArray:
    if da.notnull().any():
        LOGGER.info(f"Updating non-NaN values of parameter `{da.name}` to {dtype} type")
        da = da.astype(dtype)

    return da


def _input_data_checks(data: xr.Dataset):
    data_checks = load_config("model_data_checks.yaml")
    check_results: dict[str, list] = {"fail": [], "warn": []}
    parser_ = parsing.where_parser.generate_where_string_parser()
    eval_kwargs: parsing.where_parser.EvalAttrs = {
        "equation_name": "",
        "backend_interface": None,
        "input_data": data,
        "helper_functions": helper_functions._registry["where"],
        "apply_where": True,
        "references": set(),
    }
    for check_type, check_list in check_results.items():
        for check in data_checks[check_type]:
            parsed_ = parser_.parse_string(check["where"], parse_all=True)
            failed = parsed_[0].eval("array", **eval_kwargs) & data.definition_matrix
            if failed.any():
                check_list.append(check["message"])

    exceptions.print_warnings_and_raise_errors(
        check_results["warn"], check_results["fail"]
    )


def prepare_math(user_math: dict, mode: str, ignore_mode_math: bool) -> CalliopeMath:
    """Prepare configured math as a CalliopeMath object.

    Args:
        user_math (dict): User-defined math.
        mode (str): Build mode.
        ignore_mode_math (bool): If True, initialise math with pre-defined mode math, otherwise only use user-defined math.

    Returns:
        CalliopeMath: Prepared math dictionary.
    """
    init_math_list = [] if ignore_mode_math else [mode]

    full_math_list = init_math_list + [user_math]

    LOGGER.debug(f"Math preprocessing | Loading math: {full_math_list}")
    math = CalliopeMath(full_math_list)

    return math


def prepare_operate_mode_inputs(
    data: xr.Dataset, start_window_idx: int, operate_config: config_schema.BuildOperate
) -> xr.Dataset:
    """Slice the input data to just the length of operate mode time horizon.

    Args:
        data (xr.Dataset): input data.
        start_window_idx (int, optional):
            Set the operate `window` to start at, based on integer index.
            This is used when re-initialising the backend model for shorter time horizons close to the end of the model period.
            Defaults to 0.
        operate_config (config_schema.BuildOperate): Operate mode configuration.

    Returns:
        xr.Dataset: Slice of input data.
    """
    data.coords["windowsteps"] = pd.date_range(
        data.timesteps[0].item(), data.timesteps[-1].item(), freq=operate_config.window
    )
    horizonsteps = data.coords["windowsteps"] + pd.Timedelta(operate_config.horizon)
    # We require an offset because pandas / xarray slicing is _inclusive_ of both endpoints
    # where we only want it to be inclusive of the left endpoint.
    # Except in the last time horizon, where we want it to include the right endpoint.
    clipped_horizonsteps = horizonsteps.clip(
        max=data.timesteps[-1] + TS_OFFSET
    ).drop_vars("timesteps")
    data.coords["horizonsteps"] = clipped_horizonsteps - TS_OFFSET
    sliced_inputs = data.sel(
        timesteps=slice(
            data.windowsteps[start_window_idx], data.horizonsteps[start_window_idx]
        )
    )

    return sliced_inputs


def resample(
    data: xr.Dataset, resolution: str, math: CalliopeMath, dim: str = "timesteps"
) -> xr.Dataset:
    """Function to resample timeseries data.

    Transforms the input resolution (e.g. 1h), to the given resolution (e.g. 2h).

    Args:
        data (xarray.Dataset): Calliope model data, containing only timeseries data variables.
        resolution (str):
            time resolution of the output data, given in Pandas time frequency format.
            E.g. 1h = 1 hour, 1W = 1 week, 1M = 1 month, 1T = 1 minute.
            Multiples allowed.
        math (str): Math dictionary.
        dim (str, optional): Dimension on which to resample. Defaults to "timesteps".

    Returns:
        xarray.Dataset:
            `data` resampled according to `resolution`.

    """
    resample_kwargs = {"indexer": {dim: resolution}, "skipna": True}
    new_data = data.drop_dims(dim)
    ts_data = data.drop_vars(new_data.data_vars)
    for param, param_data in ts_data.data_vars.items():
        resampler = param_data.resample(**resample_kwargs)
        resample_method = math.data["parameters"][param]["resample_method"]
        if resample_method == "sum":
            method_kwargs = {"min_count": 1}
        else:
            method_kwargs = {}

        new_data[param] = getattr(resampler, resample_method)(
            keep_attrs=True, **method_kwargs
        )
        LOGGER.debug(
            f"Time Resampling | {param} | resampling function used: {resample_method}"
        )

    return new_data


def solve_operate(
    input_data: xr.Dataset,
    backend: BackendModel,
    definition_path: str | Path | None,
    **solver_config,
) -> xr.Dataset:
    """Solve in operate (i.e. dispatch) mode.

    Optimisation is undertaken iteratively for slices of the timeseries, with
    some data being passed between slices.

    Returns:
        xr.Dataset: Results dataset.
    """
    if backend.inputs.timesteps[0] != input_data.timesteps[0]:
        LOGGER.info("Optimisation model | Resetting model to first time window.")
        backend = get_backend_model(
            input_data, definition_path, **backend.inputs.config
        )

    LOGGER.info("Optimisation model | Running first time window.")

    step_results = backend._solve(warmstart=False, **solver_config)

    results_list = []

    for idx, windowstep in enumerate(input_data.windowsteps[1:]):
        windowstep_as_string = windowstep.dt.strftime("%Y-%m-%d %H:%M:%S").item()
        LOGGER.info(
            f"Optimisation model | Running time window starting at {windowstep_as_string}."
        )
        results_list.append(
            step_results.sel(timesteps=slice(None, windowstep - TS_OFFSET))
        )
        previous_step_results = results_list[-1]
        horizonstep = input_data.horizonsteps.sel(windowsteps=windowstep)
        new_inputs = input_data.sel(timesteps=slice(windowstep, horizonstep)).drop_vars(
            ["horizonsteps", "windowsteps"], errors="ignore"
        )

        if len(new_inputs.timesteps) != len(step_results.timesteps):
            LOGGER.info(
                "Optimisation model | Reaching the end of the timeseries. "
                "Re-building model with shorter time horizon."
            )
            backend = get_backend_model(
                input_data,
                definition_path,
                start_window_idx=idx + 1,
                **backend.inputs.config,
            )

        else:
            backend._dataset.coords["timesteps"] = new_inputs.timesteps
            backend.inputs.coords["timesteps"] = new_inputs.timesteps
            for array_name, param_data in new_inputs.data_vars.items():
                if "timesteps" in param_data.dims:
                    backend.update_parameter(array_name, param_data)
                    backend.inputs[array_name] = param_data

        if "storage" in step_results:
            backend.update_parameter(
                "storage_initial",
                _recalculate_storage_initial(
                    input_data.storage_cap, previous_step_results
                ),
            )

        step_results = backend._solve(warmstart=False, **solver_config)

    results_list.append(step_results.sel(timesteps=slice(windowstep, None)))
    results = xr.concat(results_list, dim="timesteps", combine_attrs="no_conflicts")
    results.attrs["termination_condition"] = ",".join(
        set(result.attrs["termination_condition"] for result in results_list)
    )

    return results


def _recalculate_storage_initial(
    storage_cap: xr.DataArray, results: xr.Dataset
) -> xr.DataArray:
    """Calculate the initial level of storage devices for a new operate mode time slice.

    Based on storage levels at the end of the previous time slice.

    Args:
        storage_cap (xr.DataArray): Input storage capacities.
        results (xr.Dataset): Results from the previous time slice.

    Returns:
        xr.DataArray: `storage_initial` values for the new time slice.
    """
    end_storage = results.storage.isel(timesteps=-1).drop_vars("timesteps")

    new_initial_storage = end_storage / storage_cap
    return new_initial_storage


def solve_spores(
    input_data: xr.Dataset,
    results: xr.Dataset,
    backend: BackendModel,
    build_config: config_schema.Build,
    solver_config: config_schema.Solve,
) -> xr.Dataset:
    """Solve in spores (i.e. modelling to generate alternatives - MGA) mode.

    Optimisation is undertaken iteratively after setting the total monetary cost of the system.
    Technology "spores" costs are updated between iterations.

    Returns:
        xr.Dataset: Results dataset.
    """
    LOGGER.info("Optimisation model | Resetting SPORES parameters.")
    for init_param in ["spores_score", "spores_baseline_cost"]:
        default = xr.DataArray(input_data.attrs["defaults"][init_param])
        backend.update_parameter(init_param, input_data.get(init_param, default))

    backend.set_objective(build_config.objective)

    spores_config: config_schema.SolveSpores = solver_config.spores
    if not spores_config.skip_baseline_run:
        LOGGER.info("Optimisation model | Running baseline model.")
        baseline_results = backend._solve(solver_config, warmstart=False)
    else:
        LOGGER.info("Optimisation model | Using existing baseline model results.")
        baseline_results = results.copy()

    if spores_config.save_per_spore_path is not None:
        spores_config.save_per_spore_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Optimisation model | Saving SPORE baseline to file.")
        baseline_results.assign_coords(spores="baseline").to_netcdf(
            spores_config.save_per_spore_path / "baseline.nc"
        )

    # We store the results from each iteration in the `results_list` to later concatenate into a single dataset.
    results_list: list[xr.Dataset] = [baseline_results]
    spore_range = range(1, spores_config.number + 1)
    LOGGER.info(
        f"Optimisation model | Running SPORES with `{spores_config.scoring_algorithm}` scoring algorithm."
    )
    for spore in spore_range:
        LOGGER.info(f"Optimisation model | Running SPORE {spore}.")
        _spores_update_model(baseline_results, results_list, spores_config)

        iteration_results = backend._solve(solver_config, warmstart=False)
        results_list.append(iteration_results)

        if spores_config.save_per_spore_path is not None:
            LOGGER.info(f"Optimisation model | Saving SPORE {spore} to file.")
            iteration_results.assign_coords(spores=spore).to_netcdf(
                spores_config.save_per_spore_path / f"spore_{spore}.nc"
            )

    spores_dim = pd.Index(["baseline", *spore_range], name="spores")
    results = xr.concat(results_list, dim=spores_dim, combine_attrs="no_conflicts")
    results.attrs["termination_condition"] = ",".join(
        set(result.attrs["termination_condition"] for result in results_list)
    )

    return results


def _spores_update_model(
    input_data: xr.Dataset,
    backend: BackendModel,
    baseline_results: xr.Dataset,
    all_previous_results: list[xr.Dataset],
    spores_config: config_schema.SolveSpores,
):
    """Assign SPORES scores for the next iteration of the model run.

    Algorithms applied are based on those introduced in <https://doi.org/10.1016/j.apenergy.2023.121002>.

    Args:
        input_data (xr.Dataset): Input data.
        backend (BackendModel): Model backend.
        baseline_results (xr.Dataset): The initial results (before applying SPORES scoring)
        all_previous_results (list[xr.Dataset]):
            A list of all previous iterations.
                This includes the baseline results, which will be the first item in the list.
        spores_config (config_schema.SolveSpores): The SPORES configuration.
    """

    def _score_integer() -> xr.DataArray:
        """Integer scoring algorithm."""
        previous_cap = latest_results["flow_cap"].where(spores_techs)

        # Make sure that penalties are applied only to non-negligible deployments of capacity
        min_relevant_size = spores_config.score_threshold_factor * previous_cap.max(
            ["nodes", "techs"]
        )

        new_score = (
            # Where capacity was deployed more than the minimal relevant size, assign an integer penalty (score)
            previous_cap.where(previous_cap > min_relevant_size)
            .clip(min=1, max=1)
            .fillna(0)
            .where(spores_techs)
        )
        return new_score

    def _score_relative_deployment() -> xr.DataArray:
        """Relative deployment scoring algorithm."""
        previous_cap = latest_results["flow_cap"]
        if (
            "flow_cap_max" not in input_data
            or (input_data["flow_cap_max"].where(spores_techs) == np.inf).any()
        ):
            raise exceptions.BackendError(
                "Cannot score SPORES with `relative_deployment` when `flow_cap_max` is undefined for some or all tracked technologies."
            )
        relative_cap = previous_cap / input_data["flow_cap_max"]

        new_score = (
            # Make sure that penalties are applied only to non-negligible relative capacities
            relative_cap.where(relative_cap > spores_config.score_threshold_factor)
            .fillna(0)
            .where(spores_techs)
        )
        return new_score

    def _score_random() -> xr.DataArray:
        """Random scoring algorithm."""
        previous_cap = latest_results["flow_cap"].where(spores_techs)
        new_score = (
            previous_cap.fillna(0)
            .where(previous_cap.isnull(), other=np.random.rand(*previous_cap.shape))
            .where(spores_techs)
        )

        return new_score

    def _score_evolving_average() -> xr.DataArray:
        """Evolving average scoring algorithm."""
        previous_cap = latest_results["flow_cap"]
        evolving_average = sum(
            results["flow_cap"] for results in all_previous_results
        ) / len(all_previous_results)

        relative_change = abs(evolving_average - previous_cap) / evolving_average
        # first iteration
        if relative_change.sum() == 0:
            # first iteration
            new_score = _score_integer()
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

        return new_score

    latest_results = all_previous_results[-1]
    allowed_methods: dict[
        config_schema.SPORES_SCORING_OPTIONS, Callable[[], xr.DataArray]
    ] = {
        "integer": _score_integer,
        "relative_deployment": _score_relative_deployment,
        "random": _score_random,
        "evolving_average": _score_evolving_average,
    }
    # Update the slack-cost backend parameter based on the calculated minimum feasible system design cost
    constraining_cost = baseline_results.cost.groupby("costs").sum(..., min_count=1)
    backend.update_parameter("spores_baseline_cost", constraining_cost)

    # Filter for technologies of interest
    spores_techs = (
        input_data.get(spores_config.tracking_parameter, xr.DataArray(True)).notnull()
        & input_data.definition_matrix
    )
    new_score = allowed_methods[spores_config.scoring_algorithm]()

    new_score += backend.get_parameter("spores_score", as_backend_objs=False).fillna(0)

    backend.update_parameter("spores_score", new_score)

    backend.set_objective("min_spores")
