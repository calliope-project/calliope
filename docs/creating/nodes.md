
# Nodes (`nodes`)

A model can specify any number of nodes.
These nodes are linked together by transmission technologies.
By consuming a carrier in one node and outputting it in another, linked node, transmission technologies allow resources to be drawn from the system at a different node from where they are brought into it.

The `nodes` section specifies each node:

```yaml
nodes:
    region1:
        latitude: 40
        longitude: -2
        available_area: 100
        node_flow_out_max:
            data: [1000, 2000]
            index: [electricity, gas]
            dims: carriers
        techs:
            unmet_demand_power:
            demand_power:
            ccgt:
                constraints:
                    flow_cap_max: 30000
```

For technologies to be installable at a node, they must be listed under `techs` for that node.
As seen in the example above, each allowed tech must be listed, and can optionally specify additional node-specific parameters (constraints or costs).
If given, node-specific parameters supersede any group constraints a technology defines in the `techs` section for that node.

Nodes can optionally specify geographic coordinates (`latitude` and `longitude`) which are used in visualisation or to compute distances along transmission links.
Nodes can also have any arbitrary parameter assigned which will be available in the optimisation problem, indexed over the `nodes` dimension.
They can also have parameters that use the [indexed parameter syntax](parameters.md) to define node+other dimension data.
In the above example, `node_flow_out_max` at `region1` could be used to create a [custom math](../custom_math/index.md) constraint that limits the total outflow of the carriers electricity and gas at that node.

## Understanding node-level parameters

`techs` is the only required parameter in a node.
This can be an empty dictionary (`techs: {}`), which you may use if your node is just a junction for transmission technologies (which you [**do not define in the `techs` of a node**](techs.md#transmission-technologies) - rather, you define them as separate technologies that connect `from` one node `to` another node).

Everything else about defining parameters can be found on our [`techs` page](techs.md#understanding-tech-level-parameters).
