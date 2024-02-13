
# Interfacing with the built optimisation problem

On loading a model, there is no solver backend, only the input dataset.
The backend is generated when a user calls `build()` on their model.
Currently this will call back to [Pyomo](https://www.pyomo.org/) to build the model and send it off to the solver, given by the user in the run configuration `#!yaml config.solve.solver`.
Once built, solved, and returned, the user has access to the results dataset [`model.results`][calliope.Model.results] and interface functions with the backend [`model.backend`][calliope.backend.backend_model.BackendModel].

You can use this interface to:

1. Get the optimisation problem component data.
The optimisation problem has input parameters, decision variables, global expressions, constraints, and an objective.
The data of all these components are stored as [xarray.DataArray][]s and you can query the backend to inspect them.
For instance, [`#!python model.backend.parameters`][calliope.backend.backend_model.BackendModel.parameters] will provide you with an [xarray.Dataset][] of input parameters transformed into mutable objects that are used in the optimisation.
In addition to the input data you provided, these arrays fill in missing data with default values if the parameter is one of those [predefined in Calliope][model-definition-schema] (the `Parameters` section of our [pre-defined base math documentation][base-math] shows where these parameters are used within math expressions).

1. Update a parameter value.
If you are interested in updating a few values in the model, you can run [`#!python model.backend.update_parameter`][calliope.backend.backend_model.BackendModel.update_parameter].
For example, to update the energy efficiency of your `ccgt` technology in location `region1` from 0.5 to 0.1, you can run:

    ```python
    new_data = xr.DataArray(0.1, coords={"techs": "ccgt", "nodes": "region1"})
    model.backend.update_param("flow_out_eff", new_data)
    ```

This will not affect results at this stage, you'll need to rerun the backend (point 4) to optimise with these new values.

1. Update decision variable bounds.
Most of the time, decision variable bounds are actually input parameters (e.g., `flow_cap_max` for the upper bound of the `flow_cap` decision variable).
Therefore, to update the bounds you will update the parameter with [`#!python model.backend.update_parameter`][calliope.backend.backend_model.BackendModel.update_parameter].
If a fixed numeric value is instead used, e.g. in [math that you have additionally defined](../user_defined_math/index.md), you can update bounds using [`#!python model.backend.update_variable_bounds`][calliope.backend.backend_model.BackendModel.update_variable_bounds].
For instance, to update `flow_out` lower bound to 70 for `battery` at `region2`:

    ```python
    new_data = xr.DataArray(70, coords={"techs": "battery", "nodes": "region2"})
    model.backend.update_variable_bounds("flow_out", max=new_data)
    ```

1. Fix a decision variable.
If you have already run your optimisation once and you want to re-run it with some decisions fixed to their previous optimal values, you can use [`#!python model.backend.fix_variable`][calliope.backend.backend_model.BackendModel.fix_variable].
This can be useful if you change an input parameter and wish to see its effect on a limited set of decisions.
As with [`#!python model.backend.update_parameter`][calliope.backend.backend_model.BackendModel.update_parameter], you pass an [xarray.DataArray][] to the method, this time with binary values.
Where the value is `True` in the array, that variable will be fixed. E.g., to fix all `pv` area use:

    ```python
    new_data = xr.DataArray(True, coords={"techs": "pv"})
    model.backend.fix_variable("area_use", new_data)
    ```

    !!! info
        You can also _un_fix a variable using `unfix_variable`.

1. Rerunning the optimisation problem.
If you have edited parameters or variables, you will need to solve the optimisation problem again to propagate the effects.
Any time you call [`model.solve(force=true)`][calliope.Model.solve], it will use the current state of the backend.
`force=true` is required to tell Calliope that you are happy to replace the existing results with the new solution.
Once solved, [`model.results`][calliope.Model.results] will contain new result data.

!!! info "See also"
    [Backend model API][calliope.backend.backend_model.BackendModel],
    [Tutorial on interacting with the Backend][building-and-checking-the-optimisation-problem],
    [Troubleshooting strategies](../troubleshooting.md)
