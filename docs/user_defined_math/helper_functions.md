
# Helper functions

For [`where` strings](syntax.md#where-strings) and [`expression` strings](syntax.md#where-strings), there are many helper functions available to use, to allow for more complex operations to be undertaken within the string.
Their functionality is detailed in the [helper function API page](../reference/api/helper_functions.md).
Here, we give a brief summary.
Helper functions generally require a good understanding of their functionality, so make sure you are comfortable with them beforehand.

## any

Parameters are indexed over multiple dimensions.
Using `any(..., over=...)` in a `where` string allows you to check if there is at least one non-NaN value in a given dimension (akin to [xarray.DataArray.any][]).
So, `any(cost, over=[nodes, techs])` will check if there is at least one non-NaN tech+node value in the `costs` dimension (the other dimension that the `cost` decision variable is indexed over).

## defined

Similar to [any](#any), using `defined(..., within=...)` in a `where` string allows you to check for non-NaN values along dimensions.
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

Using `sum(..., over=)` in an expression allows you to sum over one or more dimensions of your component array (be it a parameter, decision variable, or global expression).

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

## group_sum

Summing over a group of one or more dimension members in a memory-efficient way may be necessary when setting constraints.
For instance, if setting upper flow limits on groups of transmission lines into / out of a set of nodes, or limiting outflow from different types of power plants.
`group_sum(<math_component>, <groupby_array>, <group_dimension>)` allows you to sum over groups of dimension members in a math component.
In the `groupby_array`, you match math component dimension members to members of a new `group_dimension` (e.g. `(node, tech)` combinations to types of `polluting_power_plants`: `(GBR, ocgt): high_particulate_emissions`, `(FRA, ccgt): low_particulate_emissions`).
Once completed, this helper function will return the math component indexed over its original dimensions _minus_ the groupby dimensions _plus_ the new grouper dimension (e.g., `[techs, nodes, carriers, timesteps]` → `[polluting_power_plants, carriers, timesteps]`).
You will need to account for this accordingly in you math `foreach` and other expression components.

!!! note
    If you want to sum over a time period on a datetime dimension, consider using the `group_datetime` convenience helper function.

## group_datetime

When working with timeseries data, you may need to constrain a variable over a time period (e.g. hours, days, weeks).
For instance, a demand may be flexible to be met at any point in a day provided the total daily demand is met.
Or, you may need to constrain the amount of resource a thermal power plant can use each month.
To achieve this, you can use the `group_datetime(<math_component>, <datetime_dimension>, <grouping_period>)` helper function.

For example, `group_datetime(flow_in, timesteps, date)` will return the `flow_in` decision variable summed over dates.
It will therefore be indexed over `[techs, nodes, carriers, date]` instead of `[techs, nodes, carriers, timesteps]` and you will need to account for this accordingly in your math expression.

!!! note
    Only a summation over the given period is possible with this helper function.
    If you want to get e.g., a maximum value per month then you will need to create a new decision variable indexed over `months` and then create a constraint per timestep per month that will effectively set that decision variable to the maximum value over all timesteps in the month.
    This can be quite memory intensive if you want to achieve it for days/weeks as you will be indexing over timesteps * days / weeks, which is a very large array.

## sum_next_n

Use the `sum_next_n(<math_component>, <dimension>, <rolling_horizon_window>)` to sum over a rolling window in a given dimension.
This can be useful in demand-side management constraints, where demand can be shifted within a limited window, e.g. 4 hours.
It can also be useful in tracking startup/shutdown periods when optimising with unit commitment.
