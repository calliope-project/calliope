# Data definitions (`data_definitions`)

Some data is not indexed over [technologies](techs.md) / [nodes](nodes.md).
This data can be defined under the top-level key `parameters`.
This could be a single value:

```yaml
data_definitions:
  my_param: 10
```

or (equivalent):

```yaml
data_definitions:
  my_param:
    data: 10
```

which can then be accessed in the model inputs `model.inputs.my_param` and used in [any math you add](../user_defined_math/index.md) as `my_param`.

Or, it can be indexed over one or more model dimension(s):

```yaml
data_definitions:
  my_indexed_param:
    data: 100
    index: monetary
    dims: costs
  my_multiindexed_param:
    data: [2, 10]
    index: [[monetary, electricity], [monetary, heat]]  # (1)!
    dims: [costs, carriers]
```

1. The length of the inner index lists is equal to the length of `dims`.
The length of the outer list is equal to the length of `data`.

which can be accessed in the model inputs and [any math you add](../user_defined_math/index.md), e.g., `model.inputs.my_multiindexed_param.sel(costs="monetary")` and `my_multiindexed_param`.

You can also index over a new dimension:

```yaml
data_definitions:
  my_indexed_param:
    data: 100
    index: my_index_val
    dims: my_new_dim
```

Which will add the new dimension `my_new_dim` to your model: `model.inputs.my_new_dim` which you could choose to build a math component over:
`foreach: [my_new_dim]`.

!!! warning
    The `parameter` section should not be used for large datasets (e.g., indexing over the time dimension) as it will have a high memory overhead when loading the data.

## Parameters versus lookups

When you specify data through `data_definitions`, you may be populating either a parameter or what Calliope calls a "lookup". A lookup is essentially a "helper parameter" with non-numeric values, for example, a string or a boolean (True/False) value.

Whether your data definition becomes a parameter or a lookup depends on the defined model math [see the documentation on user-defined math](user_defined_math/customise.md) for more on how to create new parameters and lookups.

## Broadcasting data along indexed dimensions

If you want to set the same data for all index items, you can set the `init` [configuration option](config.md) `broadcast_param_data` to True and then use a single value in `data`:

=== "Without broadcasting"

    ```yaml
    my_indexed_param:
      data: [1, 1, 1, 1]
      index: [my_index_val1, my_index_val2, my_index_val3, my_index_val4]
      dims: my_new_dim
    ```

=== "With broadcasting"

    ```yaml
    my_indexed_param:
      data: 1  # All index items will take on this value
      index: [my_index_val1, my_index_val2, my_index_val3, my_index_val4]
      dims: my_new_dim
    ```

!!! warning
    The danger of broadcasting is that you maybe update `index` as a scenario override without realising that the data will be broadcast over this new index.
    E.g., if you start with `!#yaml {data: 1, index: monetary, dims: costs}` and update it with `!#yaml {index: [monetary, emissions]}` then the `data` value of `1` will be set for both `monetary` and `emissions` index values.
