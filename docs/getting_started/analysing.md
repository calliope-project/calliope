# Analysing a model

Calliope inputs and results are designed for easy handling.
Whatever software you prefer to use for data processing, either the NetCDF or CSV output options should provide a path to importing your Calliope results.

!!! example
    Refer to the [examples and tutorials section](../examples/index.md) section for a more practical look at how to analyse model results.

## Easiest approach: Using Calligraph

The most straightforward approach to analysing model results is to use [Calligraph](https://calligraph.readthedocs.io/), our separate tool for visualising Calliope results:

<video controls>
    <source src="https://spontaneous-choux-e05fa1.netlify.app/calligraph.mp4" type="video/mp4">
</video>

After having run your model and saved its results to NetCDF (e.g., `calliope run my_model.yaml --save_netcdf=results.nc`), you can open the resulting NetCDF file with Calligraph simply by running:

```shell
$ calligraph results.nc
```

This will open up Calligraph's interactive in-browser interface.
For more, see the [Calligraph documentation](https://calligraph.readthedocs.io/).

## Accessing model data and results within Python

A model which solved successfully has two primary [xarray.Dataset][]s with data of interest:

* `model.inputs`: contains all input data, such as renewable resource capacity factors.
* `model.results`: contains all results data, such as dispatch decisions and installed capacities.
  It also includes [results calculated in postprocessing][postprocessed-statistics], such as levelised cost of electricity (LCOE) and capacity factor.

Both of these are an  [xarray.Dataset][] and can be further processed with Python.

Data is indexed over a subset of the core Calliope dimensions, e.g. `techs` (technologies), `nodes`, `timesteps`.
Not all combinations of the dimensions items will contain data; if you did not define your `PV` technology at node `X1` then there will be no data for `#!python model.inputs.flow_cap.sel(techs="PV", nodes="X1")`.
In fact, there is likely to be more empty (`NaN`) data points than filled ones.
In Python you can quickly "densify" your data to look at only filled data points: `model.inputs.flow_cap.to_series().dropna()`

!!! note
    On [saving to CSV][calliope.Model.to_csv], each data variable is saved to its own file with all empty data points removed.

### Dimensions and broadcasting

Each parameter is only stored over the dimensions it was actually defined over, so that Calliope keeps its model data as small as possible.
For example, if you set `flow_cap_max` to the same value for every technology, defining it only at the technology level, then `#!python model.inputs.flow_cap_max` will be indexed over `techs` alone. This means that it will _not_ have a `nodes` dimension.

As soon as you override the value for one technology at one node, a `nodes` dimension appears.
Calliope only broadcasts a parameter up to the full set of dimensions it needs when it builds the optimisation problem, so the inputs and results you save reflect as minimal a form as possible.

This means the dimensions of a saved array can change depending on how you defined your data.
If your post-processing relies on a parameter always having, say, a `nodes` dimension, you should broadcast the data to the dimensions you expect on your end when you read it in.

The cleanest target to broadcast over is `#!python model.inputs.definition_matrix`, a boolean array over `[nodes, techs, carriers]` that is `True` exactly for the valid technology/carrier combinations at each node.
Broadcasting against it therefore only adds dimensions where a technology is actually defined:

```python
# Give flow_cap_max a nodes dimension
model.inputs.flow_cap_max.broadcast_like(model.inputs.definition_matrix).where(
    model.inputs.definition_matrix
)
```

!!! tip
    Broadcasting is easiest to do in [xarray][], so we recommend saving to NetCDF ([calliope.Model.to_netcdf][]) if your post-processing script will be doing broadcasting after reading the data back in.

!!! info "See also"
    If you would rather have Calliope produce an output over a fixed set of dimensions for you, you can define a [post-processed result](../basic/postprocessing.md) in your math with an explicit `foreach`, and reference the parameter in its expression.
    For example, to get `flow_cap_max` indexed over a known set of dimensions:

    ```yaml
    postprocessed:
      flow_cap_max_known_dims:
        foreach: [nodes, techs, carriers] # `results_flow_cap_max_known_dims.csv` will be indexed over these dimensions
        where: flow_cap_max
        equations:
          - expression: flow_cap_max
    ```

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

## Visualisation

You can visualise Calliope results with:

* Calligraph: See [the Calligraph documentation](https://calligraph.readthedocs.io/).
* Python: Refer to the [example notebooks](../examples/index.md) for some ideas on how to visualise directly within Python.
* Any tool: Save your model results to CSV or NetCDF ([calliope.Model.to_csv][] or [calliope.Model.to_netcdf][]), then further process and analyse them elsewhere.
