
# Model configuration (`config`)

The model configuration specifies the information Calliope needs to initialise, build, and solve the model.
This includes for example the choice of solver with which to actually solve the mathematical optimisation problem. A simple example looks like this:

```yaml
config:
  init:
    name: 'My energy model'
    time_data_path: 'timeseries_data'
    time_subset: ['2005-01-01', '2005-01-05']
  build:
    mode: plan
  solve:
    solver: cbc
```

The configuration is grouped into three top-level items:

* The `init` configuration items are used when you initialise your model (`calliope.Model(...)`).
* The `build` configuration items are used when you build your optimisation problem (`calliope.Model.build(...)`).
* The `solve` configuration items are used when you solve your optimisation problem (`calliope.Model.solve(...)`).

At each of these stages you can override what you have put in your YAML file (or if not in your YAML file, [the default that Calliope uses][model-configuration-schema]).
You do this by providing additional keyword arguments on calling `calliope.Model` or its methods. E.g.,:

```python
# Overriding `config.init` items in `calliope.Model`
model = calliope.Model("path/to/model.yaml", time_subset=["2005-01", "2005-02"])
# Overriding `config.build` items in `calliope.Model.build`
model.build(ensure_feasibility=True)
# Overriding `config.solve` items in `calliope.Model.solve`
model.solve(save_logs="path/to/logs/dir")
```

None of the configuration options are _required_ as there is a default value for them all, but you will likely want to set `init.name`, `init.calliope_version`, `init.time_data_path`, `build.mode`, and `solve.solver`.

To test your model pipeline, `config.init.time_subset` is a good way to limit your model size by slicing the time dimension to a smaller range.

## Deep-dive into some key configuration options

### `config.build.ensure_feasibility`

For a model to find a feasible solution, supply must always be able to meet demand.
To avoid the solver failing to find a solution because your constraints do not enable all demand to be met, you can ensure feasibility:

```yaml
config.build.ensure_feasibility: true
```

This will create an `unmet_demand` decision variable in the optimisation, which can pick up any mismatch between supply and demand, across all carriers.
It has a very high cost associated with its use, so it will only appear when absolutely necessary.

!!! note
    When ensuring feasibility, you can also set a [big M value](https://en.wikipedia.org/wiki/Big_M_method) (`parameters.bigM`). This is the "cost" of unmet demand.
    It is possible to make model convergence very slow if bigM is set too high.
    Default bigM is 1x10$^9$, but should be close to the maximum total system cost that you can imagine.
    This is perhaps closer to 1x10$^6$ for urban scale models and can be as low as 1x10$^4$ if you have re-scaled your data in advance.

### `config.build.mode`

In the `build` section we have `mode`.
A model can run in `plan`, `operate`, or `spores` mode.
In `plan` mode, capacities are determined by the model, whereas in `operate` mode, capacities are fixed and the system is operated with a receding horizon control algorithm.
In `spores` mode, the model is first run in `plan` mode, then run `N` number of times to find alternative system configurations with similar monetary cost, but maximally different choice of technology capacity and location (node).

In most cases, you will want to use the `plan` mode.
In fact, you can use a set of results from using `plan` model to initialise both the `operate` (`config.build.operate_use_cap_results`) and `spores` modes.

### `config.solve.solver`

Possible options for solver include `glpk`, `gurobi`, `cplex`, and `cbc`.
The interface to these solvers is done through the Pyomo library. Any [solver compatible with Pyomo](https://pyomo.readthedocs.io/en/6.5.0/solving_pyomo_models.html#supported-solvers) should work with Calliope.

For solvers with which Pyomo provides more than one way to interface, the additional `solver_io` option can be used.
In the case of Gurobi, for example, it is usually fastest to use the direct Python interface:

```yaml
config:
  solve:
    solver: gurobi
    solver_io: python
```

!!! note
    While explicitly setting the non-default `solver_io: python` is faster for Gurobi, the opposite is currently true for CPLEX, which runs faster with the default `solver_io`.

We tend to test using `cbc` but it is not available to install into your Calliope mamba environment on Windows.
Therefore, we recommend you install GLPK when you are first starting out with Calliope (`mamba install glpk`).
