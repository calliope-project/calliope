
# Helper functions

For [`where` strings](syntax.md#where-strings) and [`expression` strings](syntax.md#where-strings), there are many helper functions available to use, to allow for more complex operations to be undertaken within the string.
Their functionality is detailed in the [helper function API page](../reference/api/helper_functions.md).
Here, we give a brief summary.
Some of these helper functions require a good understanding of their functionality to apply, so make sure you are comfortable with them before using them.

## inheritance

using `inheritance(...)` in a `where` string allows you to grab a subset of technologies / nodes that all share the same [`template`](../creating/templates.md) in the technology's / node's `template` key.
If a `template` also inherits from another `template` (chained inheritance), you will get all `techs`/`nodes` that are children along that inheritance chain.

So, for the definition:

```yaml
templates:
  techgroup1:
    template: techgroup2
    flow_cap_max: 10
  techgroup2:
    base_tech: supply
techs:
  tech1:
    template: techgroup1
  tech2:
    template: techgroup2
```

`inheritance(techgroup1)` will give the `[tech1]` subset and `inheritance(techgroup2)` will give the `[tech1, tech2]` subset.

## any

Parameters are indexed over multiple dimensions.
Using `any(..., over=...)` in a `where` string allows you to check if there is at least one non-NaN value in a given dimension (akin to [xarray.DataArray.any][]).
So, `any(cost, over=[nodes, techs])` will check if there is at least one non-NaN tech+node value in the `costs` dimension (the other dimension that the `cost` decision variable is indexed over).

## defined

Similar to [any](syntax.md#any), using `defined(..., within=...)` in a `where` string allows you to check for non-NaN values along dimensions.
In the case of `defined`, you can check if e.g., certain technologies have been defined within the nodes or certain carriers are defined within a group of techs or nodes.

So, for the definition:

```yaml
techs:
  tech1:
    base_tech: conversion
    carrier_in: electricity
    carrier_out: heat
  tech2:
    base_tech: conversion
    carrier_in: [coal, biofuel]
    carrier_out: electricity
nodes:
  node1:
    techs: {tech1}
  node2:
    techs: {tech1, tech2}
```

`defined(carriers=electricity, within=techs)` would yield a list of `[True, True]` as both technologies define electricity.

`defined(techs=[tech1, tech2], within=nodes)` would yield a list of `[True, True]` as both nodes define _at least one_ of `tech1` or `tech2`.

`defined(techs=[tech1, tech2], within=nodes, how=all)` would yield a list of `[False, True]` as only `node2` defines _both_ `tech1` and `tech2`.

## sum

Using `sum(..., over=)` in an expression allows you to sum over one or more dimension of your component array (be it a parameter, decision variable, or global expression).

## select_from_lookup_arrays

Some of our arrays in [`model.inputs`][calliope.Model.inputs] are not data arrays, but "lookup" arrays.
These arrays are used to map the array's index items to other index items.
For instance when using [time clustering](../advanced/time.md#time-clustering), the `lookup_cluster_last_timestep` array is used to get the timestep resolution and the stored energy for the last timestep in each cluster.
Using `select_from_lookup_arrays(..., dim_name=lookup_array)` allows you to apply this lookup array to your data array.

## get_val_at_index

If you want to access an integer index in your dimension, use `get_val_at_index(dim_name=integer_index)`.
For example, `get_val_at_index(timesteps=0)` will get the first timestep in your timeseries, `get_val_at_index(timesteps=-1)` will get the final timestep.
This is mostly used when conditionally applying a different expression in the first / final timestep of the timeseries.

It can be used in the `where` string (e.g., `timesteps=get_val_at_index(timesteps=0)` to mask all other timesteps) and the `expression string` (via [slices](syntax.md#slices) - `storage[timesteps=$first_timestep]` and `first_timestep` expression being `get_val_at_index(timesteps=0)`).

## roll

We do not use for-loops in our math.
This can be difficult to get your head around initially, but it means that to define expressions of the form `var[t] == var[t-1] + param[t]` requires shifting all the data in your component array by N places.
Using `roll(..., dimension_name=N)` allows you to do this.
For example, `roll(storage, timesteps=1)` will shift all the storage decision variable objects by one timestep in the array.
Then, `storage == roll(storage, timesteps=1) + 1` is equivalent to applying `storage[t] == storage[t - 1] + 1` in a for-loop.

## default_if_empty

We work with quite sparse arrays in our models.
So, although your arrays are indexed over e.g., `nodes`, `techs` and `carriers`, a decision variable or parameter might only have one or two values in the array, with the rest being NaN.
This can play havoc with defining math, with `nan` values making their way into your optimisation problem and then killing the solver or the solver interface.
Using `default_if_empty(..., default=...)` in your `expression` string allows you to put a placeholder value in, which will be used if the math expression unavoidably _needs_ a value.
Usually you shouldn't need to use this, as your `where` string will mask those NaN values.
But if you're having trouble setting up your math, it is a useful function to getting it over the line.

!!! note
    Our internally defined parameters, listed in the `Parameters` section of our [pre-defined base math documentation][base-math] all have default values which propagate to the math.
    You only need to use `default_if_empty` for decision variables and global expressions, and for user-defined parameters.

## where

[Where strings](syntax.md#where-strings) only allow you to apply conditions across the whole expression equations.
Sometimes, it's necessary to apply specific conditions to different components _within_ the expression.
Using `where(<math_component>, <condition>)` helper function enables this,
where `<math_component>` is a reference to a parameter, variable, or global expression and `<condition>` is a reference to an array in your model inputs that contains only `True`/`1` and `False`/`0`/`NaN` values.
`<condition>` will then be applied to `<math_component>`, keeping only the values in `<math_component>` where `<condition>` is `True`/`1`.

This helper function can also be used to _extend_ the dimensions of a `<math_component>`.
If the `<condition>` has any dimensions not present in `<math_component>`, `<math_component>` will be [broadcast](https://tutorial.xarray.dev/fundamentals/02.3_aligning_data_objects.html#broadcasting-adjusting-arrays-to-the-same-shape) to include those dimensions.

!!! note
    `Where` gets referred to a lot in Calliope math.
    It always means the same thing: applying [xarray.DataArray.where][].
