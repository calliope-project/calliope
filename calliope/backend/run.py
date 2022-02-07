"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""
import logging

import numpy as np
import pandas as pd
import xarray as xr
from calliope import exceptions
from calliope.backend import checks
from calliope.backend.pyomo import interface as pyomo_interface
from calliope.backend.pyomo import model as run_pyomo
from calliope.core import io
from calliope.core.attrdict import AttrDict
from calliope.core.util.dataset import split_loc_techs
from calliope.core.util.logging import log_time
from calliope.core.util.observed_dict import UpdateObserverDict

logger = logging.getLogger(__name__)


def run(model_data, timings, build_only=False):
    """
    Parameters
    ----------
    model_data : xarray.Dataset
        Pre-processed dataset of Calliope model data.
    timings : dict
        Stores timings of various stages of model processing.
    build_only : bool, optional
        If True, the backend only constructs its in-memory representation
        of the problem rather than solving it. Used for debugging and
        testing.

    """

    BACKEND = {"pyomo": run_pyomo}

    INTERFACE = {"pyomo": pyomo_interface}

    run_config = AttrDict.from_yaml_string(model_data.attrs["run_config"])

    if run_config["mode"] == "plan":
        results, backend, opt = run_plan(
            model_data,
            timings,
            backend=BACKEND[run_config.backend],
            build_only=build_only,
        )

    elif run_config["mode"] == "operate":
        results, backend, opt = run_operate(
            model_data,
            timings,
            backend=BACKEND[run_config.backend],
            build_only=build_only,
        )

    elif run_config["mode"] == "spores":
        results, backend, opt = run_spores(
            model_data,
            timings,
            interface=INTERFACE[run_config.backend],
            backend=BACKEND[run_config.backend],
            build_only=build_only,
        )

    return results, backend, opt, INTERFACE[run_config.backend].BackendInterfaceMethods


def run_plan(
    model_data,
    timings,
    backend,
    build_only,
    backend_rerun=False,
    allow_warmstart=False,
    persistent=True,
    opt=None,
):

    log_time(logger, timings, "run_start", comment="Backend: starting model run")

    warmstart = False
    if not backend_rerun:
        backend_model = backend.generate_model(model_data)
        log_time(
            logger,
            timings,
            "run_backend_model_generated",
            time_since_run_start=True,
            comment="Backend: model generated",
        )

    else:
        backend_model = backend_rerun
        if allow_warmstart:
            warmstart = True

    run_config = backend_model.__calliope_run_config
    solver = run_config["solver"]
    solver_io = run_config.get("solver_io", None)
    solver_options = run_config.get("solver_options", None)
    save_logs = run_config.get("save_logs", None)

    if build_only:
        results = xr.Dataset()

    else:
        if "persistent" in solver and persistent is False:
            exceptions.warn(
                f"The chosen solver, `{solver}` will not be used in this run. "
                f"`{solver.replace('_persistent', '')}` will be used instead."
            )
            solver = solver.replace("_persistent", "")
        log_time(
            logger,
            timings,
            "run_solver_start",
            comment="Backend: sending model to solver",
        )

        backend_results, opt = backend.solve_model(
            backend_model,
            solver=solver,
            solver_io=solver_io,
            solver_options=solver_options,
            save_logs=save_logs,
            warmstart=warmstart,
            opt=opt,
        )

        log_time(
            logger,
            timings,
            "run_solver_exit",
            time_since_run_start=True,
            comment="Backend: solver finished running",
        )

        termination = backend.load_results(backend_model, backend_results, opt)

        log_time(
            logger, timings, "run_results_loaded", comment="Backend: loaded results"
        )

        if termination in ["optimal", "feasible"]:
            results = backend.get_result_array(backend_model, model_data)
            results.attrs["termination_condition"] = termination
            if "persistent" in opt.name and persistent is True:
                results.attrs["objective_function_value"] = opt.get_model_attr("ObjVal")
            else:
                results.attrs["objective_function_value"] = backend_model.obj()
        else:
            results = xr.Dataset(attrs={"termination_condition": termination})

        log_time(
            logger,
            timings,
            "run_solution_returned",
            time_since_run_start=True,
            comment="Backend: generated solution array",
        )

    return results, backend_model, opt


def run_spores(
    model_data, timings, interface, backend, build_only, backend_rerun=False, opt=None
):
    """
    For use when mode is 'spores', to allow the model to be built, edited, and
    iteratively run within Pyomo, modifying, at each iteration, the score of
    each loc::tech in such a way to generate Spatially explicit Practically Optimal
    RESults (SPORES).

    """
    log_time(
        logger,
        timings,
        "run_start",
        comment="Backend: starting model run in SPORES mode",
    )

    def _cap_loc_score_default(results):
        # Define default scoring function, based on integer scoring method
        # TODO: make it possible to point to a custom function instead of this one
        cap_loc_score = xr.where(results.energy_cap > 1e-3, 100, 0).loc[
            {"loc_techs": results.loc_techs_investment_cost}
        ]
        return cap_loc_score.to_series().rename_axis(index="loc_techs_investment_cost")

    # Define function to update "spores_score" after each iteration of the method
    def _update_spores_score(backend_model, cap_loc_score):
        print("Updating loc::tech spores scores")
        loc_tech_score_dict = cap_loc_score.to_dict()

        interface.update_pyomo_param(
            backend_model, opt, "cost_energy_cap", loc_tech_score_dict
        )

    def _warn_on_infeasibility():
        return exceptions.warn(
            "Infeasible SPORE detected. Please check your model configuration. "
            "No more SPORES will be generated."
        )

    def _get_updated_spores_inputs(backend_model):
        inputs = interface.access_pyomo_model_inputs(backend_model)
        inputs_to_keep = ["cost_energy_cap", "group_cost_max"]
        return inputs[inputs_to_keep]

    def _combine_spores_results_and_inputs(
        backend_model, results, spore_num, model_data=None
    ):
        inputs_to_keep = _get_updated_spores_inputs(backend_model)
        for var_data in results.data_vars.values():
            if "is_result" not in var_data.attrs.keys():
                var_data.attrs["is_result"] = 1
        if model_data is not None:
            datasets = [inputs_to_keep, model_data, results]
        else:
            datasets = [inputs_to_keep, results]
        new_ds = xr.combine_by_coords(
            datasets, compat="override", combine_attrs="no_conflicts"
        )
        return new_ds.assign_coords(spores=("spores", [spore_num]))

    def _add_results_to_list(backend_model, spores_results, results, spore_num):
        results_to_add = _combine_spores_results_and_inputs(
            backend_model, results, spore_num
        )
        spores_results[spore_num] = results_to_add

    def _save_spore(backend_model, results, spore_num, model_data=None):
        _path = spores_config["save_per_spore_path"].format(spore_num)
        new_ds = _combine_spores_results_and_inputs(
            backend_model, results, spore_num, model_data
        )
        print(f"Saving SPORE {spore_num} to {_path}")
        io.save_netcdf(new_ds, _path)

    def _update_slack_cost_constraint(backend_model):
        slack_costs = model_data.group_cost_max.loc[
            {"group_names_cost_max": spores_config["slack_cost_group"]}
        ].dropna("costs")
        interface.update_pyomo_param(
            backend_model,
            opt,
            "group_cost_max",
            {
                (_cost_class, spores_config["slack_cost_group"]): results.cost.loc[
                    {"costs": _cost_class}
                ]
                .sum()
                .item()
                * (1 + spores_config["slack"])
                for _cost_class in slack_costs.costs.values
            },
        )

    def _update_to_spores_objective(backend_model):
        interface.update_pyomo_param(
            backend_model,
            opt,
            "objective_cost_class",
            spores_config["objective_cost_class"],
        )

    def _initialise_backend_model():
        if backend_rerun:
            kwargs = {"backend_rerun": backend_rerun, "opt": opt}
        else:
            kwargs = {}
        if spores_config["skip_cost_op"]:
            print(
                "Skipping cost optimal run and using model_data to initialise SPORES directly"
            )
            return run_plan(model_data, timings, backend, build_only=True, **kwargs)
        else:
            # Run once for the 'cost-optimal' solution
            return run_plan(
                model_data, timings, backend, build_only, persistent=False, **kwargs
            )

    def _initialise_spores_number():
        if "spores" in model_data.coords and spores_config["skip_cost_op"]:
            return model_data.spores.max().item()
        else:
            return 0

    def _error_on_malformed_input():
        if backend_rerun:
            try:
                backend_rerun.obj()
            except ValueError:  # model has not yet been run
                pass
            else:
                raise exceptions.ModelError(
                    "Cannot run SPORES if the backend model already has a solution. "
                    "Consider using the `build_only` optional `run()` argument to avoid this."
                )
        if "spores" in model_data.filter_by_attrs(is_result=0).squeeze().dims:
            raise exceptions.ModelError(
                "Cannot run SPORES with a SPORES dimension in any input (e.g. `cost_energy_cap`)."
            )

    _error_on_malformed_input()

    if backend_rerun:
        model_data = _combine_spores_results_and_inputs(
            backend_rerun, xr.Dataset(), 0, model_data=model_data
        )
    run_config = UpdateObserverDict(
        initial_yaml_string=model_data.attrs["run_config"],
        name="run_config",
        observer=model_data,
    )

    spores_config = run_config["spores_options"]
    if "spores" in model_data.dims and model_data.spores.size == 1:
        model_data = model_data.squeeze("spores")

    init_spores_scores = (
        model_data.cost_energy_cap.loc[{"costs": [spores_config["score_cost_class"]]}]
        .to_series()
        .dropna()
    )
    spores_results = {}

    results, backend_model, opt = _initialise_backend_model()
    if build_only:
        return results, backend_model, opt

    init_spore = _initialise_spores_number()

    if spores_config["skip_cost_op"]:
        cumulative_spores_scores = init_spores_scores.copy()
        _update_to_spores_objective(backend_model)
    elif results.attrs["termination_condition"] in ["optimal", "feasible"]:
        results.attrs["objective_function_value"] = backend_model.obj()
        if spores_config["save_per_spore"] is True:
            _save_spore(backend_model, results, init_spore, model_data=model_data)
        # Storing results and scores in the specific dictionaries
        _add_results_to_list(backend_model, spores_results, results, 0)
        cumulative_spores_scores = init_spores_scores + _cap_loc_score_default(results)
        # Set group constraint "cost_max" equal to slacked cost
        _update_slack_cost_constraint(backend_model)
        _update_to_spores_objective(backend_model)
        # Update "spores_score" based on previous iteration
        _update_spores_score(backend_model, cumulative_spores_scores)

        log_time(
            logger,
            timings,
            "run_solution_returned",
            time_since_run_start=True,
            comment="Backend: generated solution array for the cost-optimal case",
        )
    else:
        _warn_on_infeasibility()
        return results, backend_model, opt

    # Iterate over the number of SPORES requested by the user
    for _spore in range(init_spore + 1, spores_config["spores_number"] + 1):
        print(f"Running SPORES {_spore}")
        if opt is not None and "persistent" in opt.type:
            opt = interface.regenerate_persistent_pyomo_solver(
                backend_model,
                opt,
                obj=True,
                constraints={
                    "cost_investment_constraint": cumulative_spores_scores.index,
                    "cost_constraint": cumulative_spores_scores.index,
                },
            )
        else:
            opt = None
        results, backend_model, opt = run_plan(
            model_data,
            timings,
            backend,
            build_only=False,
            backend_rerun=backend_model,
            allow_warmstart=False,
            persistent=True,
            opt=opt,
        )

        if results.attrs["termination_condition"] in ["optimal", "feasible"]:
            results.attrs["objective_function_value"] = backend_model.obj()
            if spores_config["save_per_spore"] is True:
                _save_spore(backend_model, results, _spore)
            # Storing results and scores in the specific dictionaries
            _add_results_to_list(backend_model, spores_results, results, _spore)
            print(f"Updating capacity scores from {cumulative_spores_scores.sum()}...")
            cumulative_spores_scores += _cap_loc_score_default(results)
            print(f"... to {cumulative_spores_scores.sum()}")
            # Update "spores_score" based on previous iteration
            _update_spores_score(backend_model, cumulative_spores_scores)
        else:
            _warn_on_infeasibility()
            break
        log_time(
            logger,
            timings,
            "run_solution_returned",
            time_since_run_start=True,
            comment=f"Backend: generated solution array for the SPORE {_spore}",
        )

    results = xr.concat(
        spores_results.values(), dim=pd.Index(spores_results.keys(), name="spores")
    )

    return results, backend_model, opt


def run_operate(model_data, timings, backend, build_only):
    """
    For use when mode is 'operate', to allow the model to be built, edited, and
    iteratively run within Pyomo.

    """
    log_time(
        logger,
        timings,
        "run_start",
        comment="Backend: starting model run in operational mode",
    )

    defaults = UpdateObserverDict(
        initial_yaml_string=model_data.attrs["defaults"],
        name="defaults",
        observer=model_data,
    )
    run_config = UpdateObserverDict(
        initial_yaml_string=model_data.attrs["run_config"],
        name="run_config",
        observer=model_data,
    )

    # New param defaults = old maximum param defaults (e.g. energy_cap gets default from energy_cap_max)
    operate_params = {
        k.replace("_max", ""): v for k, v in defaults.items() if k.endswith("_max")
    }
    operate_params[
        "purchased"
    ] = 0  # no _max to work from here, so we hardcode a default

    defaults.update(operate_params)

    # Capacity results (from plan mode) can be used as the input to operate mode
    if any(model_data.filter_by_attrs(is_result=1).data_vars) and run_config.get(
        "operation.use_cap_results", False
    ):
        # Anything with is_result = 1 will be ignored in the Pyomo model
        for varname, varvals in model_data.data_vars.items():
            if varname in operate_params.keys():
                varvals.attrs["is_result"] = 1
                varvals.attrs["operate_param"] = 1

    else:
        cap_max = xr.merge(
            [
                v.rename(k.replace("_max", ""))
                for k, v in model_data.data_vars.items()
                if "_max" in k
            ]
        )
        cap_equals = xr.merge(
            [
                v.rename(k.replace("_equals", ""))
                for k, v in model_data.data_vars.items()
                if "_equals" in k
            ]
        )
        caps = cap_max.update(cap_equals)
        for cap in caps.data_vars.values():
            cap.attrs["is_result"] = 1
            cap.attrs["operate_param"] = 1
        model_data.update(caps)

    comments, warnings, errors = checks.check_operate_params(model_data)
    exceptions.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # Initialize our variables
    solver = run_config["solver"]
    solver_io = run_config.get("solver_io", None)
    solver_options = run_config.get("solver_options", None)
    save_logs = run_config.get("save_logs", None)
    window = run_config["operation"]["window"]
    horizon = run_config["operation"]["horizon"]
    window_to_horizon = horizon - window

    # get the cumulative sum of timestep resolution, to find where we hit our window and horizon
    timestep_cumsum = model_data.timestep_resolution.cumsum("timesteps").to_pandas()
    # get the timesteps at which we start and end our windows
    window_ends = timestep_cumsum.where(
        (timestep_cumsum % window == 0) | (timestep_cumsum == timestep_cumsum[-1])
    )
    window_starts = timestep_cumsum.where(
        (~np.isnan(window_ends.shift(1))) | (timestep_cumsum == timestep_cumsum[0])
    ).dropna()

    window_ends = window_ends.dropna()
    horizon_ends = timestep_cumsum[
        timestep_cumsum.isin(window_ends.values + window_to_horizon)
    ]
    # solves bug where horizon_ends is empty if window < # of timesteps < horizon
    if len(horizon_ends) == 0:
        horizon_ends = timestep_cumsum[timestep_cumsum == window_ends.values[-1]]

    if not any(window_starts):
        raise exceptions.ModelError(
            "Not enough timesteps or incorrect timestep resolution to run in "
            "operational mode with an optimisation window of {}".format(window)
        )

    # We will only update timseries parameters
    timeseries_data_vars = [
        k
        for k, v in model_data.data_vars.items()
        if "timesteps" in v.dims and v.attrs["is_result"] == 0
    ]

    # Loop through each window, solve over the horizon length, and add result to
    # result_array we only go as far as the end of the last horizon, which may
    # clip the last bit of data
    result_array = []
    # track whether each iteration finds an optimal solution or not
    terminations = []

    if build_only:
        iterations = [0]
    else:
        iterations = range(len(window_starts))

    for i in iterations:
        start_timestep = window_starts.index[i]

        # Build full model in first instance
        if i == 0:
            warmstart = False
            end_timestep = horizon_ends.index[i]
            timesteps = slice(start_timestep, end_timestep)
            window_model_data = model_data.loc[dict(timesteps=timesteps)]

            log_time(
                logger,
                timings,
                "model_gen_1",
                comment="Backend: generating initial model",
            )

            backend_model = backend.generate_model(window_model_data)

        # Build the full model in the last instance(s),
        # where the number of timesteps may be less than the horizon length
        elif i > len(horizon_ends) - 1 or i == iterations[-1]:
            warmstart = False
            end_timestep = window_ends.index[i]
            timesteps = slice(start_timestep, end_timestep)
            window_model_data = model_data.loc[dict(timesteps=timesteps)]

            log_time(
                logger,
                timings,
                "model_gen_{}".format(i + 1),
                comment=(
                    "Backend: iteration {}: generating new model for "
                    "end of timeseries, with horizon = {} timesteps".format(
                        i + 1, window_ends[i] - window_starts[i]
                    )
                ),
            )

            backend_model = backend.generate_model(window_model_data)

        # Update relevent Pyomo Params in intermediate instances
        else:
            warmstart = True
            end_timestep = horizon_ends.index[i]
            timesteps = slice(start_timestep, end_timestep)
            window_model_data = model_data.loc[dict(timesteps=timesteps)]

            log_time(
                logger,
                timings,
                "model_gen_{}".format(i + 1),
                comment="Backend: iteration {}: updating model parameters".format(
                    i + 1
                ),
            )
            # Pyomo model sees the same timestamps each time, we just change the
            # values associated with those timestamps
            for var in timeseries_data_vars:
                # New values
                var_series = (
                    window_model_data[var].to_series().dropna().replace("inf", np.inf)
                )
                # Same timestamps
                var_series.index = backend_model.__calliope_model_data["data"][
                    var
                ].keys()
                var_dict = var_series.to_dict()
                # Update pyomo Param with new dictionary

                getattr(backend_model, var).store_values(var_dict)

        if not build_only:
            log_time(
                logger,
                timings,
                "model_run_{}".format(i + 1),
                time_since_run_start=True,
                comment="Backend: iteration {}: sending model to solver".format(i + 1),
            )
            # After iteration 1, warmstart = True, which should speed up the process
            # Note: Warmstart isn't possible with GLPK (dealt with later on)
            _results, _opt = backend.solve_model(
                backend_model,
                solver=solver,
                solver_io=solver_io,
                solver_options=solver_options,
                save_logs=save_logs,
                warmstart=warmstart,
            )

            log_time(
                logger,
                timings,
                "run_solver_exit_{}".format(i + 1),
                time_since_run_start=True,
                comment="Backend: iteration {}: solver finished running".format(i + 1),
            )
            # xarray dataset is built for each iteration
            _termination = backend.load_results(backend_model, _results, _opt)
            terminations.append(_termination)
            if _termination in ["optimal", "feasible"]:
                _results = backend.get_result_array(backend_model, model_data)
            else:
                _results = xr.Dataset()

            # We give back the actual timesteps for this iteration and take a slice
            # equal to the window length
            _results["timesteps"] = window_model_data.timesteps.copy()

            # We always save the window data. Until the last window(s) this will crop
            # the window_to_horizon timesteps. In the last window(s), optimistion will
            # only be occurring over a window length anyway
            _results = _results.loc[dict(timesteps=slice(None, window_ends.index[i]))]
            result_array.append(_results)

            # Set up initial storage for the next iteration
            if "loc_techs_store" in model_data.dims.keys() and _termination in [
                "optimal",
                "feasible",
            ]:
                storage_initial = (
                    _results.storage.loc[{"timesteps": window_ends.index[i]}].drop_vars(
                        "timesteps"
                    )
                    / model_data.storage_cap
                )
                model_data["storage_initial"].loc[
                    storage_initial.coords
                ] = storage_initial.values
                backend_model.storage_initial.store_values(
                    storage_initial.to_series().dropna().to_dict()
                )

            # Set up total operated units for the next iteration
            if "loc_techs_milp" in model_data.dims.keys() and _termination in [
                "optimal",
                "feasible",
            ]:
                operated_units = _results.operating_units.sum("timesteps").astype(int)
                model_data["operated_units"].loc[{}] += operated_units.values
                backend_model.operated_units.store_values(
                    operated_units.to_series().dropna().to_dict()
                )

            log_time(
                logger,
                timings,
                "run_solver_exit_{}".format(i + 1),
                time_since_run_start=True,
                comment="Backend: iteration {}: generated solution array".format(i + 1),
            )

    if build_only:
        results = xr.Dataset()
        _opt = None
    else:
        # Concatenate results over the timestep dimension to get a single
        # xarray Dataset of interest
        results = xr.concat(result_array, dim="timesteps")
        if all(i == "optimal" for i in terminations):
            results.attrs["termination_condition"] = "optimal"
        elif all(i in ["optimal", "feasible"] for i in terminations):
            results.attrs["termination_condition"] = "feasible"
        else:
            results.attrs["termination_condition"] = ",".join(terminations)

        log_time(
            logger,
            timings,
            "run_solution_returned",
            time_since_run_start=True,
            comment="Backend: generated full solution array",
        )

    return results, backend_model, _opt
