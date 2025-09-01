# Running a model in Python

The most basic way to run a model programmatically from within a Python interpreter is to create a [calliope.Model][] instance with a given `model.yaml` configuration file, and then call its [calliope.Model.build][] followed by [calliope.Model.solve][] methods:

```python
import calliope
model = calliope.Model('path/to/model.yaml')
model.build()
model.solve()
```

!!! note
    If the model definition is not specified (i.e. `model = Model()`), an error is raised.
    See the example models introduced in the [examples and tutorials](examples/overview.md) section for information on instantiating a simple model without specifying a custom model configuration.

Other ways to load a model in Python are:

* Passing an [calliope.AttrDict][] or standard Python dictionary to the [calliope.Model][] constructor, with the same nested format as the YAML model configuration (top-level keys: `config`, `parameters`, `nodes`, `techs`, etc.).
* Loading a previously saved model from a NetCDF file with `#!python model = calliope.read_netcdf('path/to/saved_model.nc')`.
This can either be a pre-processed model saved before its `build` method was called - which will include input data only - or a completely solved model, which will include input and result data.

After instantiating the [calliope.Model][] object, and before calling the `build()` method, it is possible to manually inspect and adjust the configuration of the model.
The pre-processed inputs are all held in the xarray Dataset `model.inputs`.

After the model has been solved, an xarray Dataset containing results (`model.results`) can be accessed.
At this point, the model can be saved with either [calliope.Model.to_csv][] or [calliope.Model.to_netcdf][], which saves all inputs and results, and is equivalent to the corresponding `--save` options of the command-line tool.

!!! example
    An example of running in an interactive Python session, which also demonstrates some of the analysis possibilities after running a model, is given in the [tutorials](../examples/overview.md).
    You can download and run the embedded notebooks on your own machine (if both Calliope and the Jupyter Notebook are installed).

## Applying a scenario or override when running in Python

There are two ways to override a base model when running in Python which are analogous to the [use of the command-line tool](running-cli.md):

1. By setting the `scenario` argument, e.g.:

    ```python
    model = calliope.Model('model.yaml', scenario='milp')
    ```

2. By passing the `override_dict` argument, which is a Python dictionary, a [calliope.AttrDict][], or a YAML string of overrides:

    ```python
    model = calliope.Model(
        'model.yaml',
        override_dict={'config.solve.solver': 'gurobi'}
    )
    ```

!!! note
    Both `scenario` and `override_dict` can be defined at once.
    They will be applied in order, such that scenarios are applied first, followed by dictionary overrides.
    Therefore, the `override_dict` can be used to override scenarios.

## Tracking progress

When running Calliope in the command line, logging of model pre-processing and solving occurs automatically.
Interactively, for example in a Jupyter notebook, you can enable verbose logging by setting the log level using [calliope.set_log_verbosity][] immediately after importing the Calliope package.
By default, [calliope.set_log_verbosity][] also sets the log level for the solver to `DEBUG`, which allows you to view the solver status.
This can be disabled by `#!python calliope.set_log_verbosity(level, include_solver_output=False)`.
Possible log levels are (from least to most verbose):

1. `CRITICAL`: only show critical errors.
2. `ERROR`: only show errors.
3. `WARNING`: show errors and warnings (default level).
4. `INFO`: show errors, warnings, and informative messages. Calliope uses the INFO level to show a message at each stage of pre-processing, sending the model to the solver, and post-processing, including timestamps.
5. `DEBUG`: SOLVER logging, with heavily verbose logging of a number of function outputs. Only for use when troubleshooting failing runs or developing new functionality in Calliope.
