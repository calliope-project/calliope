
# Choosing an optimisation problem "backend"

On loading a model, there is no solver backend, only the input dataset.
The backend is generated when a user calls `build()` on their model.
By default this will call back to [Pyomo](https://www.pyomo.org/) to build the model and send it off to the solver given by the user in the run configuration `#!yaml config.solve.solver`.

Pyomo has the benefit of _mutable_ input parameters, which allows you to [update parameter values in your optimisation problem](backend_interface.md) without having to re-build any Pyomo objects.
However, it is otherwise a relatively memory and time-consuming library for building optimisation problems.

Since larger models tend to require a commercial solver to successfully complete in a reasonable amount of time (see our [solver comparison for justification](solver.md)), we have also introduced a direct interface to the Gurobi solver Python API.
Our tests show that this reduces peak memory consumption and time to solution compared to using the Pyomo backend with Gurobi as the solver in both cases.
If you have access to a Gurobi license, this does not require any extra effort on your part, besides having to:

1. Install the Gurobi Python library into your Calliope environment: `mamba install gurobi::gurobi`.
1. Select the Gurobi backend in your YAML configuration (`!#yaml config.build.backend: gurobi`) or at build time if running in a Python script or interactively (`!#python model.build(backend="gurobi")`).

You can still [interface with your optimisation problem](backend_interface.md), but some methods will raise an exception when the Gurobi Python API does not allow for something that the Pyomo API does.