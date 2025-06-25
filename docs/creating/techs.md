
# Technologies (`techs`)

The `techs` section in the model configuration specifies all of the model's technologies.

Calliope allows a modeller to define technologies with arbitrary characteristics by defining one of the abstract base techs as `base_tech`.
This establishes the basic characteristics in the optimisation model (decision variables and constraints) applied to the technology:

* `supply`: Draws from a source to produce a carrier.
* `demand`: Consumes a carrier to supply to an external sink.
* `storage`: Stores a carrier.
* `transmission`: Transmits a carrier from one node to another.
* `conversion`: Converts from one carrier to another.

??? info "Sharing configuration with templates"

    To share definitions between technologies and/or nodes, you can use configuration templates (the `template` key).
    This allows a technology/node to inherit definitions from [`template` definitions](yaml.md#reusing-definitions-through-templates).
    Note that `template` is different to setting a `base_tech`.
    Setting a base_tech does not entail any configuration options being inherited;
    `base_tech` is only used when building the optimisation problem (i.e., in the `math`).

The following example shows the definition of a `ccgt` technology, i.e. a combined cycle gas turbine that delivers electricity:

```yaml
ccgt:
  name: 'Combined cycle gas turbine'
  color: '#FDC97D'  # (1)!
  base_tech: supply
  carrier_out: power
  source_use_max: .inf # (2)!
  flow_out_eff: 0.5
  flow_cap_max: 40000  # kW
  lifetime: 25
  cost_interest_rate:
    data: 0.10  # (3)!
    index: monetary
    dims: costs
  cost_flow_cap:
    data: 750  # USD per kW
    index: monetary
    dims: costs
  cost_flow_in:
    data: 0.02  # USD per kWh
    index: monetary
    dims: costs
```

1. This is an example of when using quotation marks is important.
Without them, the colour code would be interpreted as a YAML comment!
2. the period at the start of `.inf` will ensure it is read in as a `float` type.
3. Costs require us to explicitly define data in the [indexed parameter](parameters.md) format so that we can define the cost class (in this case: `monetary`).

Each technology must specify an abstract base technology and its carrier (`carrier_out` in the case of a `supply` technology).
Specifying a `color` and a `name` is optional but useful when you want to [visualise or otherwise report your results](../analysing.md).

The rest of the data for the technology is used in the optimisation problem: to set constraints and to link the technology to the objective function (via costs).
In the above example, we have a capacity limit `flow_cap_max`, conversion efficiency `flow_out_eff`, the life time (used in [levelised cost calculations](./reference/api/postprocess.md)), and the resource available for consumption `source_use_max`.
In the above example, the source is set to infinite via `inf`.

The parameters starting with `costs_` give costs for the technology.
Calliope uses the concept of "cost classes" to allow accounting for more than just monetary costs.
The above example specifies only the `monetary` cost class, but any number of other classes could be used, for example `co2` to account for emissions.
Additional cost classes can be created simply by adding them to the definition of costs for a technology.

??? info "Costs in the objective function"
    By default, all defined cost classes are used in the objective function, i.e., the default objective is to minimize total costs.
    Limiting the considered costs can be achieved by [customising the in-built objective function](../user_defined_math/customise.md) to only focus on e.g. monetary costs (`[monetary] in costs`),
    or updating the `objective_cost_weights` indexed parameter to have a weight of `0` for those cost classes you want to be ignored, e.g.:

    ```yaml
    parameters:
      objective_cost_weights:
        data: [1, 0]
        index: [monetary, co2_emissions]
        dims: costs
    ```

## Transmission technologies

You will see in [`nodes`](nodes.md) page that you make it possible for investment in technologies at nodes by specifying the technology name under the node key.
You cannot do this with transmission technologies since they span two nodes.
Instead, you associate transmission technologies with nodes in `techs`:

```yaml
techs:
  ac_transmission:
    link_from: region1  # (1)!
    link_to: region2
    flow_cap_max: 100
    ...
```

1. The region you specify in `link_from` or `link_to` is interchangeable unless you set the parameter `one_way: true`.
In that case, flow along the transmission line is only allowed from the `link_from` node to the `link_to` node.

## Understanding tech-level parameters

### Required parameters

There are _required_ parameters according to the technology `base_tech`:

* `supply`: `base_tech` and `carrier_out`.
* `demand`: `base_tech` and `carrier_in`.
* `storage`: `base_tech` and `carrier_out` and `carrier_in`.
* `transmission`: `base_tech` and `carrier_out`, `carrier_in`, `link_to`, and `link_from`.
* `conversion`: `base_tech` and `carrier_out` and `carrier_in`.

For `storage` and `transmission`, it may seem like unnecessary repetition to define both `carrier_out` and `carrier_in` as they are likely the same value.
However, doing so makes it much easier to process your model definition in Calliope!

### Pre-defined parameters

There is a long list of pre-defined parameters that we use in our [base math][base-math].
These are listed in full with descriptions and units in our [model definition reference page][model-definition-schema].
These parameters come with strict types, default values, and other attributes used in internal Calliope processing.
Therefore they should always be your first port of call.
However, if you want to add your own parameters, that is also possible.

### Adding your own parameter

You can also add any new parameter you like, which will then be available to use in any [math you want to additionally apply](../user_defined_math/index.md).
The only requirements we apply are that it _cannot_ start with an underscore or a number.

We also have a check for any parameter starting with `cost_`.
These _must_ define a cost class, e.g.,:

```yaml
techs:
  tech1:
    cost_custom:
      data: 1
      index: monetary
      dims: costs
```

If you forget to use the [indexed parameter](parameters.md) format for a parameter starting with `cost_` then our YAML schema will raise an error.
For example, this is not valid and will create an error:

```yaml
techs:
  tech1:
    cost_custom: 1
```

### Using the indexed parameter format

The [indexed parameter](parameters.md) format allows you to add dimensions to your data.
By defining just a data value, the resulting parameter will only be indexed over the `techs` dimension (+ optionally the `nodes` dimension if you provide a new value for it at a [node](nodes.md)).
By using the indexed parameter format, you can add new dimensions.
We saw this above with `costs`, but you can add _any_ dimension _except_ `nodes`.

!!! example

    ```yaml
    techs:
      tech1:
          source_use_equals:
            data: [15, 5]
            index: ["2005-01-01 12:00:00", "2005-01-01 13:00:00"]
            dims: timesteps
    ```

    ```yaml
    techs:
      tech1:
        flow_cap_max:
          data: [10, 100]
          index: ["electricity", "heat"]
          dims: carriers
    ```

    ```yaml
    techs:
      tech1:
        cost_flow_cap:
          data: [4, 8]
          index: [["electricity", "monetary"], ["heat", "monetary"]]
          dims: [carriers, costs]
    ```

    ```yaml
    techs:
      tech1:
        source_use_equals:
          data: [15, 5]
          index: ["foo", "bar"]
          dims: my_new_dimension
    ```

### (De)activating techs

In an [override](scenarios.md) you may want to remove a technology entirely from the model.
The easiest way to do this is to set `active: false`.
The resulting input dataset won't feature that technology in any way.
You can even do this to deactivate technologies at specific [nodes](nodes.md) and to deactivate nodes entirely.
Conversely, setting `active: true` in an override will lead to the technology reappearing.
