# Analysing a model

Calliope inputs and results are designed for easy handling.
Whatever software you prefer to use for data processing, either the NetCDF or CSV output options should provide a path to importing your Calliope results.

!!! example
    Refer to the [examples and tutorials section](examples/index.md) section for a more practical look at how to analyse model results.

## Accessing model data and results

A model which solved successfully has two primary [xarray.Dataset][]s with data of interest:

* `model.inputs`: contains all input data, such as renewable resource capacity factors
* `model.results`: contains all results data, such as dispatch decisions and installed capacities.
  It also includes [results calculated in postprocessing](./reference/api/postprocess.md), such as levelised cost of electricity (LCOE) and capacity factor.

Both of these are [xarray.Dataset][]s and can be further processed with Python.

Data is indexed over a subset of the core Calliope dimensions, e.g. `techs` (technologies), `nodes`, `timesteps`.
Not all combinations of the dimensions items will contain data; if you did not define your `PV` technology at node `X1` then there will be no data for `#!python model.inputs.flow_cap.sel(techs="PV", nodes="X1")`.
In fact, there is likely to be more empty (`NaN`) data points than filled ones.
In Python you can quickly "densify" your data to look at only filled data points: `model.inputs.flow_cap.to_series().dropna()`

!!! note
    On [saving to CSV][calliope.Model.to_csv], each data variable is saved to its own file with all empty data points removed.

## Reading solutions

Calliope provides functionality to read a previously-saved model from a single NetCDF file:

```python
solved_model = calliope.read_netcdf('my_saved_model.nc')
```

Once loaded, the input and results data can be accessed as above (i.e., `solved_model.inputs` and `solved_model.results`).

!!! warning
    Calliope's NetCDF files can be processed with any other tool that can deal with NetCDF.
    However, certain model attributes are serialised on saving the model to ensure the convention is followed.
    To view the model as intended, use Calliope functionality to read the NetCDF.

!!! info "See also"
    The [xarray][] documentation should be consulted for further information on dealing with Datasets.
