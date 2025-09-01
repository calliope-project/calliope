# Nodes (`nodes`)

## Understanding node-level parameters

`techs` is the only required parameter in a node.
This can be an empty dictionary (`techs: {}`), which you may use if your node is just a junction for transmission technologies (which you [**do not define in the `techs` of a node**](techs.md#transmission-technologies) - rather, you define them as separate technologies that connect a `link_from` node with a `link_to` node).

!!! info "See also"

    See the [`techs` page](techs.md#understanding-tech-level-parameters) for the different formats in which you can define parameters, which holds for node-level parameters too.

## Arbitrary per-node data

Nodes can have arbitrary parameter data assigned which will be available in the optimisation problem, indexed over the `nodes` dimension (see `custom_node_parameter` in the example below).

They can also have parameters that use the [indexed parameter syntax](parameters.md) to define node+other dimension data (`custom_node_flow_out_max` in the example below)
In the below example, `custom_node_flow_out_max` at `region1` could be used to create [your own math](../user_defined_math/index.md) constraint that limits the total outflow of the carriers electricity and gas at that node.

```yaml
nodes:
  region1:
    ...
    custom_node_parameter: 100
    custom_node_flow_out_max:
      data: [1000, 2000]
      index: [electricity, gas]
      dims: carriers
    ...
```

### (De)activating nodes

In an [override](scenarios.md) you may want to remove a node entirely from the model.
The easiest way to do this is to set `active: false`.
The resulting input dataset won't feature that node in any way.
You can even do this to deactivate [technologies](techs.md) at a node.
Conversely, setting `active: true` in an override will lead to the node reappearing.

!!! note
    When deactivating nodes, any transmission technologies that link to that node will also be deactivated.
