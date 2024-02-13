# Indexed parameters (`parameters`)

Some data is not indexed over [technologies](techs.md) / [nodes](nodes.md).
This data can be defined under the top-level key `parameters`.
This could be a single value:

```yaml
parameters:
  my_param: 10
```

or (equivalent):

```yaml
parameters:
  my_param:
    data: 10
```

which can then be accessed in the model inputs `model.inputs.my_param` and used in [any math you add](../user_defined_math/index.md) as `my_param`.

Or, it can be indexed over one or more model dimension(s):

```yaml
parameters:
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
parameters:
  my_indexed_param:
    data: 100
    index: my_index_val
    dims: my_new_dim
```

Which will add the new dimension `my_new_dim` to your model: `model.inputs.my_new_dim` which you could choose to build a math component over:
`foreach: [my_new_dim]`.

!!! warning
    The `parameter` section should not be used for large datasets (e.g., indexing over the time dimension) as it will have a high memory overhead on loading the data.
