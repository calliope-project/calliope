# Running a model

There are essentially three ways to run a Calliope model:

1. With the `calliope run` command-line tool (see below, the more detailed explanation in [Running in the command line](../basic/running-cli.md) and the [CLI reference](../reference/cli.md)).
2. By programmatically creating and running a model from within other Python code, or in an interactive Python session (see [Running in Python](../basic/running-python.md)).
3. By generating and then executing scripts with the `#!shell calliope generate_runs` command-line tool, which is primarily designed for running many scenarios on a high-performance cluster (see the [Generating scripts](../advanced/scripts.md) section in the advanced docs).

!!! example
    Refer to the [examples and tutorials section](examples/overview.md) for a more practical look at how to run a Calliope model.

## Easiest to start: using the command line interface

We can easily run a model defined in `model.yaml` and save results to a single NetCDF file, `results.nc`, for further processing:

```shell
$ calliope run model.yaml --save_netcdf=results.nc
```

An alternative is to save results to CSV files in the directory `results_directory`:

```shell
$ calliope run model.yaml --save_csv=results_directory
```

This can be useful for further processing in a tool like Excel, but because of the more than two-dimensional nature of many of Calliope's inputs and results, can be quite unwieldy.

For more details, including how to apply a scenario or override, see the [documentation on running in the command line](../basic/running-cli.md).

## Improving solution times

Large models will take time to solve.
The easiest is often to just let a model run on a remote device (another computer, or a high performance computing cluster) and forget about it until it is done.
However, if you need results *now*, there are ways to improve solution time.

Details on strategies to improve solution times are given in the [troubleshooting](../troubleshooting.md) section.

## Debugging failing runs

These things will typically go wrong, in order of decreasing likelihood:

* The model is improperly defined or missing data.
Calliope will attempt to diagnose some common errors and raise an appropriate error message.
* The model is consistent and properly defined but infeasible.
Calliope will be able to construct the model and pass it on to the solver, but the solver (after a potentially long time) will abort with a message stating that the model is infeasible.
* There is a bug in Calliope causing the model to crash either before being passed to the solver, or after the solver has completed and when results are passed back to Calliope.

Calliope provides help in diagnosing all of these model issues. For details, see the [troubleshooting](../troubleshooting.md) section.
