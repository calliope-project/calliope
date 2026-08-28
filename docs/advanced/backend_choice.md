
# Choosing an optimisation problem "backend"

On loading a model, there is no solver backend, only the input dataset.
The backend is generated when a user calls `build()` on their model.
By default this will call back to [Pyomo](https://www.pyomo.org/) to build the model and send it off to the solver given by the user in the run configuration `#!yaml config.solve.solver`.

Pyomo has the benefit of _mutable_ input parameters, which allows you to [update parameter values in your optimisation problem](backend_interface.md) without having to re-build any Pyomo objects.
However, it is otherwise a relatively memory and time-consuming library for building optimisation problems.

Since larger models tend to require a commercial solver to successfully complete in a reasonable amount of time (see our [solver comparison for justification](solver.md)), we have also introduced a direct interface to the Gurobi solver Python API.
Our tests show that this reduces peak memory consumption and time to solution compared to using the Pyomo backend with Gurobi as the solver in both cases.
If you have access to a Gurobi license, this does not require any extra effort on your part, besides having to:

1. Ensure the Gurobi Python library is installed in your Calliope environment:

    === "`pixi`"

        ```shell
        pixi add gurobi::gurobi
        ```

    === "`conda`"

        ```shell
        conda install gurobi::gurobi
        ```

2. Select the Gurobi backend:

    === "In YAML"

        ```yaml
        config.build.backend: gurobi
        ```

    === "In Python"

        ```python
        model.build(backend="gurobi")
        ```

You can still [interface with your optimisation problem](backend_interface.md), but some methods will raise an exception when the Gurobi Python API does not allow for something that the Pyomo API does.

If you do not have access to a commercial solver, there is also a direct interface to the open-source [HiGHS](https://highs.dev/) solver via its Python API ([highspy](https://pypi.org/project/highspy/)), which is installed with Calliope by default.
As with the Gurobi backend, building your optimisation problem this way is faster and less memory-intensive than going via Pyomo.
To use it, select the HiGHS backend in your YAML configuration (`#!yaml config.build.backend: highs`) or at build time (`#!python model.build(backend="highs")`).
Note that the HiGHS backend does not yet support piecewise constraints and cannot un-fix fixed variables without a rebuild.
