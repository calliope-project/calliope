# Running a model

There are essentially three ways to run a Calliope model:

1. With the `calliope run` command-line tool (see the [CLI reference][cli-reference] for full details, or a basic explanation below).
2. By programmatically creating and running a model from within other Python code, or in an interactive Python session.
3. By generating and then executing scripts with the `#!shell calliope generate_runs` command-line tool, which is primarily designed for running many scenarios on a high-performance cluster.

!!! example
    Refer to the [examples and tutorials section](examples/index.md) for a more practical look at how to run a Calliope model.

## Running with the command-line tool

We can easily run a model after creating it (see [creating-a-model][]), saving results to a single NetCDF file for further processing.

```shell
$ calliope run testmodel/model.yaml --save_netcdf=results.nc
```

The `calliope run` command takes the following options:

* `--save_netcdf={filename.nc}`: Save complete model, including results, to the given NetCDF file. This is the recommended way to save model input and output data into a single file, as it preserves all data fully, and allows later reconstruction of the Calliope model for further analysis.
* `--save_csv={directory name}`: Save results as a set of CSV files to the given directory. This can be handy if the modeler needs results in a simple text-based format for further processing with a tool like Microsoft Excel.
* `--debug`: Run in debug mode, which prints more internal information, and is useful when troubleshooting failing models.
* `--scenario={scenario}` and `--override_dict={yaml_string}`: Specify a scenario, or one or several overrides, to apply to the model, or apply specific overrides from a YAML string (see below for more information).
* `--help`: Show all available options.

Multiple options can be specified, for example, saving NetCDF, CSV, and HTML plots simultaneously.

```shell
$ calliope run testmodel/model.yaml --save_netcdf=results.nc --save_csv=outputs
```

!!! warning
    The command-line tool does not save results by default -- the modeller must specify one of the `-save` options.


### Applying a scenario or override on the command line

!!! note "See also"
    See the [scenarios-and-overrides][] section for details on how to define scenarios.

The `--scenario` option can be used in three different ways:

* It can be given the name of a scenario defined in the model configuration, as in `--scenario=my_scenario`
* It can be given the name of a single override defined in the model configuration, as in `--scenario=my_override`
* It can be given a comma-separated string of several overrides defined in the model configuration, as in `--scenario=my_override_1,my_override_2`

In the latter two cases, the given override(s) is used to implicitly create a "scenario" on-the-fly when running the model.
This allows quick experimentation with different overrides without explicitly defining a scenario combining them.

Assuming we have specified an override called ``milp`` in our model configuration, we can apply it to our model with:

```shell
$ calliope run testmodel/model.yaml --scenario=milp --save_netcdf=results.nc
```

Note that if both a scenario and an override with the same name exist (such as ``milp`` in the above example), Calliope will raise an error, as it will not be clear which one the user wishes to apply.

It is also possible to use the `--override_dict` option to pass a YAML string that will be applied after anything applied through `--scenario`:

```shell
$ calliope run testmodel/model.yaml --override_dict="{'model.time_subset': ['2005-01-01', '2005-01-31']}" --save_netcdf=results.nc
```

## Running in Python

The most basic way to run a model programmatically from within a Python interpreter is to create a [calliope.Model][] instance with a given `model.yaml` configuration file, and then call its [calliope.Model.build][] followed by [calliope.Model.solve][] methods:

```python
import calliope
model = calliope.Model('path/to/model.yaml')
model.build()
model.solve()
```

!!! note
    If the model definition is not specified (i.e. `model = Model()`), an error is raised.
    See the example models introduced in the [examples-and-tutorials][] section for information on instantiating a simple model without specifying a custom model configuration.

Other ways to load a model in Python are:

* Passing an [calliope.AttrDict][] or standard Python dictionary to the [calliope.Model][] constructor, with the same nested format as the YAML model configuration (top-level keys: `config`, `parameters`, `nodes`, `techs`, etc.).
* Loading a previously saved model from a NetCDF file with `#!python model = calliope.read_netcdf('path/to/saved_model.nc')`.
This can either be a pre-processed model saved before its `build` method was called - which will include input data only - or a completely solved model, which will include input and result data.

After instantiating the [calliope.Model][] object, and before calling the `build()` method, it is possible to manually inspect and adjust the configuration of the model.
The pre-processed inputs are all held in the xarray Dataset `model.inputs`.

After the model has been solved, an xarray Dataset containing results (`model.results`) can be accessed.
At this point, the model can be saved with either [calliope.Model.to_csv][] or [calliope.Model.to_netcdf][], which saves all inputs and results, and is equivalent to the corresponding `--save` options of the command-line tool.

!!! example
    An example of running in an interactive Python session, which also demonstrates some of the analysis possibilities after running a model, is given in the [tutorials][examples-and-tutorials].
    You can download and run the embedded notebooks on your own machine (if both Calliope and the Jupyter Notebook are installed).

### Applying a scenario or override when running in Python

There are two ways to override a base model when running in Python which are analogous to the use of the command-line tool (see the [applying-a-scenario-or-override-on-the-command-line][] section):

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

### Tracking progress

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

## Generating scripts for many model runs

Scripts to simplify the creation and execution of a large number of Calliope model runs are generated with the `calliope generate_runs` command-line tool.
More detail on this is available in the section [Generating scripts to repeatedly run variations of a model][generating-scripts-to-repeatedly-run-variations-of-a-model].

## Improving solution times

Large models will take time to solve.
The easiest is often to just let a model run on a remote device (another computer, or a high performance computing cluster) and forget about it until it is done.
However, if you need results *now*, there are ways to improve solution time.

Details on strategies to improve solution times are given in the [troubleshooting][] section.

## Debugging failing runs

What will typically go wrong, in order of decreasing likelihood:

* The model is improperly defined or missing data.
Calliope will attempt to diagnose some common errors and raise an appropriate error message.
* The model is consistent and properly defined but infeasible.
Calliope will be able to construct the model and pass it on to the solver, but the solver (after a potentially long time) will abort with a message stating that the model is infeasible.
* There is a bug in Calliope causing the model to crash either before being passed to the solver, or after the solver has completed and when results are passed back to Calliope.

Calliope provides help in diagnosing all of these model issues. For details, see the [troubleshooting][] section.
