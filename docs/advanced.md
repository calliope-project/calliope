# Advanced features

Once you're comfortable with [building](building.md), [running](running.md) and [analysing](analysing.md) one of the built-in example models, you may want to explore Calliope's advanced functionality. With these features, you will be able to build and run complex models in no time.

## Time resolution adjustment

Models have a default timestep length (defined implicitly by the timesteps of the model's time series data). This default resolution can be adjusted by specifying time resolution adjustment in the model configuration, for example:

```yaml
config:
    init:
        time_resample: 6H
```

In the above example, this would resample all time series data to 6-hourly timesteps.
Any [pandas-compatible rule describing the target resolution](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html) can be used.

## Generating scripts to repeatedly run variations of a model

[Scenarios and overrides][scenarios-and-overrides] can be used to run a given model multiple times with slightly changed settings or constraints.

This functionality can be used together with the `calliope generate_runs` and `calliope generate_scenarios` command-line tools to generate scripts that run a model many times over in a fully automated way, for example, to explore the effect of different technology costs on model results.

`calliope generate_runs`, at a minimum, must be given the following arguments:

* the model configuration file to use
* the name of the script to create
* `--kind`: Currently, three options are available. windows creates a Windows batch (.bat) script that runs all models sequentially, bash creates an equivalent script to run on Linux or macOS, bsub creates a submission script for a LSF-based high-performance cluster, and sbatch creates a submission script for a SLURM-based high-performance cluster.
* `--scenarios`: A semicolon-separated list of scenarios (or overrides/combinations of overrides) to generate scripts for, for example, scenario1;scenario2 or override1,override2a;override1,override2b. Note that when not using manually defined scenario names, a comma is used to group overrides together into a single model -- in the above example, override1,override2a would be applied to the first run and override1,override2b be applied to the second run

A fully-formed command generating a Windows batch script to run a model four times with each of the scenarios "run1", "run2", "run3", and "run4":

```shell
calliope generate_runs model.yaml run_model.bat --kind=windows --scenarios "run1;run2;run3;run4"
```

Optional arguments are:

* `--cluster_threads`: specifies the number of threads to request on a HPC cluster
* `--cluster_mem`: specifies the memory to request on a HPC cluster
* `--cluster_time`: specifies the run time to request on a HPC cluster
* `--additional_args`: A text string of any additional arguments to pass directly through to `calliope run` in the generated scripts, for example, `--additional_args="--debug"`.
* `--debug`: Print additional debug information when running the run generation script.

An example generating a script to run on a bsub-type high-performance cluster, with additional arguments to specify the resources to request from the cluster:

```shell
calliope generate_runs model.yaml submit_runs.sh --kind=bsub --cluster_mem=1G --cluster_time=100 --cluster_threads=5  --scenarios "run1;run2;run3;run4"
```

Running this will create two files:

* `submit_runs.sh`: The cluster submission script to pass to bsub on the cluster.
* `submit_runs.array.sh`: The accompanying script defining the runs for the cluster to execute.

In all cases, results are saved into the same directory as the script, with filenames of the form `out_{run_number}_{scenario_name}.nc` (model results) and `plots_{run_number}_{scenario_name}.html` (HTML plots), where `{run_number}` is the run number and `{scenario_name}` is the name of the scenario (or the string defining the overrides applied). On a cluster, log files are saved to files with names starting with log_ in the same directory.

Finally, the `calliope generate_scenarios` tool can be used to quickly generate a file with scenarios definition for inclusion in a model, if a large enough number of overrides exist to make it tedious to manually combine them into scenarios. Assuming that in model.yaml a range of overrides exist that specify a subset of time for the years 2000 through 2010, called "y2000" through "y2010", and a set of cost-related overrides called "cost_low", "cost_medium" and "cost_high", the following command would generate scenarios with combinations of all years and cost overrides, calling them "run_1", "run_2", and so on, and saving them to scenarios.yaml:

```shell
calliope generate_scenarios model.yaml scenarios.yaml y2000;y2001;y2002;2003;y2004;y2005;y2006;2007;2008;y2009;2010 cost_low;cost_medium;cost_high --scenario_name_prefix="run_"
```
