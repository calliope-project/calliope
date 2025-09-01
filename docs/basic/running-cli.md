# Running a model in the command line

The basic syntax is:

```shell
$ calliope run testmodel/model.yaml --save_netcdf=results.nc
```

The `calliope run` command takes the following options:

* `--save_netcdf={filename.nc}`: Save complete model, including results, to the given NetCDF file. This is the recommended way to save model input and output data into a single file, as it preserves all data fully, and allows later reconstruction of the Calliope model for further analysis.
* `--save_csv={directory name}`: Save results as a set of CSV files to the given directory. This can be handy if the modeler needs results in a simple text-based format for further processing with a tool like Microsoft Excel.
* `--debug`: Run in debug mode, which prints more internal information, and is useful when troubleshooting failing models.
* `--scenario={scenario}` and `--override_dict={yaml_string}`: Specify a scenario, or one or several overrides, to apply to the model, or apply specific overrides from a YAML string (see below for more information).
* `--help`: Show all available options.

Multiple options can be specified, for example, saving NetCDF and CSV simultaneously.

```shell
$ calliope run testmodel/model.yaml --save_netcdf=results.nc --save_csv=outputs
```

!!! warning
    The command-line tool does not save results by default -- the modeller must specify one of the `-save` options.

## Applying a scenario or override on the command line

!!! note "See also"
    See the [Scenarios and overrides](scenarios.md) section for details on how to define scenarios.

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
$ calliope run testmodel/model.yaml --override_dict="{'init.subset.timesteps': ['2005-01-01', '2005-01-31']}" --save_netcdf=results.nc
```
