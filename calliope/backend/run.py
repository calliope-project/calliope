"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""
import logging

import numpy as np
import xarray as xr
import pandas as pd

from calliope.core.util.logging import log_time
from calliope import exceptions
from calliope.backend import checks
from calliope.backend.pyomo import model as run_pyomo
from calliope.backend.pyomo import interface as pyomo_interface

from calliope.core.util.observed_dict import UpdateObserverDict
from calliope.core.attrdict import AttrDict
from calliope.core.util.dataset import split_loc_techs
from calliope.core import io

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
        results, backend = run_operate(
            model_data,
            timings,
            backend=BACKEND[run_config.backend],
            build_only=build_only,
        )

    elif run_config["mode"] == "spores":
        results, backend = run_spores(
            model_data,
            timings,
            interface=INTERFACE[run_config.backend],
            backend=BACKEND[run_config.backend],
            build_only=build_only,
        )

    return results, backend, INTERFACE[run_config.backend].BackendInterfaceMethods


def run_plan(
    model_data,
    timings,
    backend,
    build_only,
    backend_rerun=False,
    allow_warmstart=False,
    persistent=False,
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
        opt = None

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

        results, opt = backend.solve_model(
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
            "run_solver_exit",
            time_since_run_start=True,
            comment="Backend: solver finished running",
        )

        termination = backend.load_results(backend_model, results, opt)

        log_time(
            logger, timings, "run_results_loaded", comment="Backend: loaded results"
        )

        if termination in ["optimal", "feasible"]:
            results = backend.get_result_array(backend_model, model_data)
            results.attrs["termination_condition"] = termination
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


def run_spores(model_data, timings, interface, backend, build_only):
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

    run_config = UpdateObserverDict(
        initial_yaml_string=model_data.attrs["run_config"],
        name="run_config",
        observer=model_data,
    )

    n_spores = run_config["spores_options"]["spores_number"]
    slack = run_config["spores_options"]["slack"]
    spores_score = run_config["spores_options"]["score_cost_class"]
    objective_cost_class = run_config["spores_options"]["objective_cost_class"]
    slack_group = run_config["spores_options"]["slack_cost_group"]
    save_per_spore = run_config["spores_options"]["save_per_spore"]
    save_per_spore_path = run_config["spores_options"]["save_per_spore_path"]
    skip_cost_op = run_config["spores_options"]["skip_cost_op"]

    solver = run_config["solver"]
    solver_io = run_config.get("solver_io", None)
    solver_options = run_config.get("solver_options", None)
    save_logs = run_config.get("save_logs", None)

    loc_tech_df = (
        model_data.cost_energy_cap.loc[{"costs": [spores_score]}].to_series().dropna()
    )
    spores_results = {}
    # Define default scoring function, based on integer scoring method
    # TODO: make the function to run optional
    def _cap_loc_score_default(results):
        cap_loc_score = xr.where(results.energy_cap > 1e-3, 100, 0)
        return cap_loc_score.to_series().rename_axis(index="loc_techs_investment_cost")

    # Define function to update "spores_score" after each iteration of the method
    def _update_spores_score(backend_model, cap_loc_score):
        print("Updating loc::tech spores scores")
        loc_tech_score_dict = loc_tech_df.align(cap_loc_score)[1].to_dict()

        interface.update_pyomo_param(
            backend_model, "cost_energy_cap", loc_tech_score_dict
        )

    def _warn_on_infeasibility():
        return exceptions.warn(
            "Infeasible SPORE detected. Please check your model configuration. "
            "No more SPORES will be generated."
        )

    def _save_spore(backend_model, results, spore_num, model_data=None):
        inputs = interface.access_pyomo_model_inputs(backend_model)
        inputs_to_keep = inputs[
            ["cost_energy_cap", "group_cost_max", "objective_cost_class"]
        ]
        for var in results.data_vars:
            results[var].attrs["is_result"] = 1
        if model_data is not None:
            datasets = [inputs_to_keep, model_data, results]
        else:
            datasets = [inputs_to_keep, results]
        new_ds = xr.combine_by_coords(
            datasets, compat="override", combine_attrs="no_conflicts"
        )
        new_ds = new_ds.assign_coords(spores=("spores", [spore_num]))
        print(f"Saving SPORE {spore_num} to {save_per_spore_path.format(spore_num)}")
        io.save_netcdf(new_ds, save_per_spore_path.format(spore_num))

    if not skip_cost_op:
        # Run once for the 'cost-optimal' solution
        results, backend_model, opt = run_plan(
            model_data, timings, backend, build_only, persistent=False
        )
        if build_only:
            return (
                results,
                backend_model,
            )  # We have what we need, so break out of the loop
        init_spore = 0
        if results.attrs["termination_condition"] in ["optimal", "feasible"]:

            results.attrs["objective_function_value"] = backend_model.obj()
            if save_per_spore is True:
                _save_spore(backend_model, results, init_spore, model_data=model_data)
            # Storing results and scores in the specific dictionaries
            spores_results[0] = results
            print("Getting capacity scores")
            cum_scores = _cap_loc_score_default(results)
            # Set group constraint "cost_max" equal to slacked cost
            slack_costs = model_data.group_cost_max.loc[
                {"group_names_cost_max": slack_group}
            ].dropna("costs")
            print("Updating cost group constraint")
            interface.update_pyomo_param(
                backend_model,
                "group_cost_max",
                {
                    (_cost_class, slack_group): results.cost.loc[{"costs": _cost_class}]
                    .sum()
                    .item()
                    * (1 + slack)
                    for _cost_class in slack_costs.costs.values
                },
            )
            print("Updating objective")
            interface.update_pyomo_param(
                backend_model, "objective_cost_class", objective_cost_class
            )
            print("Updating spores scores")
            # Update "spores_score" based on previous iteration
            _update_spores_score(backend_model, cum_scores)

            log_time(
                logger,
                timings,
                "run_solution_returned",
                time_since_run_start=True,
                comment="Backend: generated solution array for the cost-optimal case",
            )
        else:
            _warn_on_infeasibility()
            return results, backend_model
    else:
        print("Skipping cost optimal run and using model_data as a direct SPORES result")
        cum_scores = (
            model_data.cost_energy_cap.loc[{"costs": spores_score}]
            .to_series()
            .dropna()
            .rename_axis(index="loc_techs_investment_cost")
        )
        print(f"Input SPORES scores amount to {cum_scores.sum()}")
        cum_scores += _cap_loc_score_default(model_data)
        print(f"SPORES scores being used for next run amount to {cum_scores.sum()}")
        slack_costs = model_data.group_cost_max.loc[
            {"group_names_cost_max": slack_group}
        ].dropna("costs")
        results, backend_model, opt = run_plan(
            model_data, timings, backend, build_only=True
        )
        print("Updating objective")
        interface.update_pyomo_param(
            backend_model, "objective_cost_class", objective_cost_class
        )
        print("Updating capacity scores")
        # Update "spores_score" based on previous iteration
        _update_spores_score(backend_model, cum_scores)
        init_spore = model_data.spores.max().item()


    # Iterate over the number of SPORES requested by the user
    for _spore in range(init_spore + 1, n_spores + 1):
        print(f"Running SPORES {_spore}")
        if "persistent" in solver and _spore > 1 and skip_cost_op is False:
            opt.set_objective(backend_model.obj)
            for _cost_class in slack_costs.costs.values:
                opt.remove_constraint(
                    backend_model.group_cost_max_constraint[
                        slack_group, _cost_class, "max"
                    ]
                )
                opt.add_constraint(
                    backend_model.group_cost_max_constraint[
                        slack_group, _cost_class, "max"
                    ]
                )

            results, opt = backend.solve_model(
                backend_model,
                solver=solver,
                solver_io=solver_io,
                solver_options=solver_options,
                save_logs=save_logs,
                opt=opt,
            )
            termination = backend.load_results(backend_model, results, opt)

            log_time(
                logger, timings, "run_results_loaded", comment="Backend: loaded results"
            )

            results = backend.get_result_array(backend_model, model_data)
            results.attrs["termination_condition"] = termination

            if results.attrs["termination_condition"] in ["optimal", "feasible"]:
                results.attrs["objective_function_value"] = backend_model.obj()

            log_time(
                logger,
                timings,
                "run_solution_returned",
                time_since_run_start=True,
                comment="Backend: generated solution array",
            )
        else:
            results, backend_model, opt = run_plan(
                model_data,
                timings,
                backend,
                build_only=False,
                backend_rerun=backend_model,
                allow_warmstart=False,
                persistent=True,
            )

        if results.attrs["termination_condition"] in ["optimal", "feasible"]:
            results.attrs["objective_function_value"] = backend_model.obj()
            if save_per_spore is True:
                _save_spore(backend_model, results, _spore)
            # Storing results and scores in the specific dictionaries
            spores_results[_spore] = results
            print(f"Updating capacity scores from {cum_scores.sum()}...")
            cum_scores += _cap_loc_score_default(results)
            print(f"... to {cum_scores.sum()}")
            # Update "spores_score" based on previous iteration
            _update_spores_score(backend_model, cum_scores)
            skip_cost_op = False
            print(backend_model.obj())
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
        # TODO: make this function work with the spores dimension,
        # so that postprocessing can take place in core/model.py, as with run_plan and run_operate

    results = xr.concat(
        spores_results.values(), dim=pd.Index(spores_results.keys(), name="spores")
    )

    return results, backend_model


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
        # where number of timesteps is less than the horizon length
        elif i > len(horizon_ends) - 1:
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
            _results = backend.solve_model(
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
            _termination = backend.load_results(backend_model, _results)
            terminations.append(_termination)

            _results = backend.get_result_array(backend_model, model_data)

            # We give back the actual timesteps for this iteration and take a slice
            # equal to the window length
            _results["timesteps"] = window_model_data.timesteps.copy()

            # We always save the window data. Until the last window(s) this will crop
            # the window_to_horizon timesteps. In the last window(s), optimistion will
            # only be occurring over a window length anyway
            _results = _results.loc[dict(timesteps=slice(None, window_ends.index[i]))]
            result_array.append(_results)

            # Set up initial storage for the next iteration
            if "loc_techs_store" in model_data.dims.keys():
                storage_initial = _results.storage.loc[
                    {"timesteps": window_ends.index[i]}
                ].drop("timesteps")
                model_data["storage_initial"].loc[
                    storage_initial.coords
                ] = storage_initial.values
                backend_model.storage_initial.store_values(
                    storage_initial.to_series().dropna().to_dict()
                )

            # Set up total operated units for the next iteration
            if "loc_techs_milp" in model_data.dims.keys():
                operated_units = _results.operating_units.sum("timesteps").astype(
                    np.int
                )
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

    return results, backend_model
