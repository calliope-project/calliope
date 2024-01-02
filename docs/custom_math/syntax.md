# Math syntax

## foreach lists

If the math component is indexed over sets (e.g., `techs`, `nodes`, `timesteps`), then you need to define a `foreach` list of those sets.
If the component is dimensionless, no `foreach` list needs to be defined.

For example, `#!yaml foreach: [nodes, techs]` will build the component over all `nodes` and `techs` in the model.

The available sets in Calliope are: `nodes`, `techs`, `carriers`, `carrier_tiers`, `costs`, `timesteps`.
If using [time clustering and inter-cluster storage][time_clustering], there is also a `datesteps` set available.
If you want to build over your own custom set, you will need to add it to the calliope model dataset before building the optimisation problem, e.g. as a new top-level parameter.

## where strings

`Where` strings allow you to define math that applies to only a subset of your data or of the models you are running.
They are made up of a series of statements combined with logical operators.
These statements can be one of the following:

1. Checking the existence of set items in an input parameter.
When checking the existence of an input parameter it is possible to first sum it over one or more of its dimensions; if at least one value on the summed dimension(s) is defined, then it will be considered defined in the remaining dimensions.

    ??? examples annotate

        - If you want to apply a constraint across all `nodes` and `techs`, but only for node+tech combinations where the `flow_out_eff` parameter has been defined, you would include `flow_out_eff`.
        - If you want to apply a constraint over `techs` and `timesteps`, but only for combinations where the `source_max` parameter has at least one `node` with a value defined, you would include `any(resource, over=nodes)`.  (1)

    1.  I'm a [helper function][helper-functions]; read more below!

1. Checking the value of a configuration option or an input parameter.
Checks can use any of the operators: `>`, `<`, `=`, `<=`, `>=`.
Configuration options are any that are defined in `config.build`, where you can define your own options to access in the `where` string.

    ??? examples annotate

        - If you want to apply a constraint only if the configuration option `config.build.cyclic_storage` is _True_, you would include `config.cyclic_storage=True` (`True`/`False` is case insensitive).
        - If you want to apply a constraint across all `nodes` and `techs`, but only where the `flow_eff` parameter is less than 0.5, you would include `flow_eff<0.5`.
        - If you want to apply a constraint only for the first timestep in your timeseries, you would include `timesteps=get_val_at_index(dim=timesteps, idx=0)`. (1)
        - If you want to apply a constraint only for the last timestep in your timeseries, you would include `timesteps=get_val_at_index(dim=timesteps, idx=-1)`. (1)

    1.  I'm a [helper function][helper-functions]; read more below!

1. Checking the `parent` of a technology (`storage`, `supply`, etc.) or its inheritance chain (if using `tech_groups` and the `inherit` parameter).

    ??? examples

        - If you want to create a decision variable across only `storage` technologies, you would include `parent=storage`.
        - If you want to apply a constraint across only your own `rooftop_supply` technologies (e.g., you have defined `rooftop_supply` in `tech_groups` and your technologies `pv` and `solar_thermal` define `#!yaml inherit: rooftop_supply`), you would include `inheritance(rooftop_supply)`.

1. Subsetting a set.
The sets available to subset are always [`nodes`, `techs`, `carriers`] + any additional sets defined by you in [`foreach`][foreach-lists].

    ??? examples annotate

        - If you want to filter `nodes` where any of a set of `tech`s are defined: `defined(techs=[tech1, tech2], within=nodes, how=any)` (1).

    1. I'm a [helper function][helper-functions]; read more below!

To combine statements you can use the operators `and`/`or`.
You can also use the `not` operator to negate any of the statements.
These operators are case insensitive, so "and", "And", "AND" are equivalent.
You can group statements together using the `()` brackets.
These statements will be combined first.

??? examples

    - If you want to apply a constraint for `storage` technologies if the configuration option `cyclic_storage` is activated and it is the last timestep of the series: `parent=storage and config.cyclic_storage=True and timesteps=get_val_at_index(dim=timesteps, idx=-1)`.
    - If you want to create a decision variable for the input carriers of conversion technologies: `carrier_in and parent=conversion`
    - If you want to apply a constraint if the parameter `source_unit` is `energy_per_area` or the parameter `area_use_per_flow_cap` is defined: `source_unit=energy_per_area or area_use_per_flow_cap`.
    - If you want to apply a constraint if the parameter `flow_out_eff` is less than or equal to 0.5 and `source_use` has been defined, or `flow_out_eff` is greater than 0.9 and `source_use` has not been defined: `(flow_out_eff<=0.5 and source_use) or (flow_out_eff>0.9 and not source_use)`.

Combining `foreach` and `where` will create an n-dimensional boolean array.
Wherever index items in this array are _True_, your component `expression(s)` will be applied.

## expression strings

As with where strings, expression strings are a series of math terms combined with operators.
The terms can be input parameters, decision variables, global expressions, or numeric values that you define on-the-fly.

If you are defining a `global expression` or `objective`, then the available expression string operators are: `+`, `-`, `*`, `/`, and `**` ("to the power of").
These expressions are applied using standard operator precedence (BODMAS/PEMDAS, see [this wiki](https://en.wikipedia.org/wiki/Order_of_operations) for more info).

If you are defining a `constraint`, then you also need to define a comparison operator: `<=`, `>=`, or `==`.

??? examples

    - If you want to limit all technology outflow to be less than 200 units: `flow_out <= 200`.
    - If you want to create a global expression which is the storage level minus a parameter defining a minimum allowed storage level: `storage - storage_cap * min_storage_level`.
    - If you want to set the outflow of a specific technology `my_tech` to equal all outflows of a specific carrier `my_carrier` at each node: `flow_out[techs=my_tech] == sum(flow_out[carriers=my_carrier], over=techs)`.
    - If you want inflows at a node `my_node` to be at least as much as the inflows in the previous timestep: `flow_in[nodes=my_node] >= roll(flow_in[nodes=my_node], timesteps=1)`.

### Slicing data

You do not need to define the sets of math components in expressions, unless you are actively "slicing" them.
Behind the scenes, we will make sure that every relevant element of the defined `foreach` sets are matched together when applying the expression (we [merge the underlying xarray DataArrays](https://docs.xarray.dev/en/stable/user-guide/combining.html)).
Slicing math components involves appending the component with square brackets that contain the slices, e.g. `flow_out[carriers=electricity, nodes=[A, B]]` will slice the `flow_out` decision variable to focus on `electricity` in its `carriers` dimension and only has two nodes (`A` and `B`) on its `nodes` dimension.
To find out what dimensions you can slice a component on, see your input data (`model.inputs`) for parameters and the definition for decision variables in your loaded math dictionary (`model.math.variables`).

### Helper functions

As with [`where` strings][where-strings], there are many [helper function](https://calliope.readthedocs.io/en/latest/api_reference/#calliopebackendhelper_functions-math-formulation-helper-functions) available to use in expressions, to allow for more complex operations to be undertaken.
Some of these helper functions require a good understanding of their functionality to apply, so make sure you are comfortable with them before using them.

## equations

Equations are combinations of [expression-strings][] and [where-strings][].
You define one or more equations for your model components.
A different `where` string associated with each equation expression allows you to slightly alter the expression for different component members.
You define equations as lists of dictionaries:

``` yaml
equations:
  - where: ...
    expression: ...
  - where: ...
    expression: ...
```

If you are supplying only one equation, you do not need to define a `where` string:

```yaml
equations:
  - expression: ...
```

!!! note

    `where` strings within equations are appended to your top-level `where` string, e.g.:

    ```yaml
    where: storage_cap_max
    equations:
      - where: flow_in_eff > 0.5  # <- this will be parsed as "storage_cap_max and flow_in_eff > 0.5"
      expression: ...
      - where: flow_in_eff <= 0.5  # <- this will be parsed as "storage_cap_max and flow_in_eff <= 0.5"
      expression: ...
    ```

??? examples

    - Divide by efficiency if efficiency is larger than zero, otherwise set the variable to zero:

    ```yaml
    equations:
      - where: flow_eff > 0
        expression: flow_out / flow_out_eff == flow_in
      - where: flow_eff = 0
        expression: flow_out == 0
    ```

    - Limit flow by storage_cap, if it is defined, otherwise by flow_cap:

    ```yaml
    equations:
      - where: storage_cap
        expression: flow_out <= 0.5 * storage_cap
      - where: not storage_cap
        expression: flow_out <= 0.9 * flow_cap
    ```

!!! warning

    You have to be careful when setting up different `where` strings to avoid clashes, where different expressions are valid for the same component member.
    We will raise errors when this happens, and if your `where` strings become too restrictive and so miss a component member that needs an expression.

## sub-expressions

For long expressions - or those where a part of the expression might change for different component members according to a specific condition - you can define sub-expressions.
These look similar to equations; they are lists of dictionaries with `where` and `expression` strings.
They are accessed from you main expression(s) by reference to their name prepended with the special `$` character.
For example:

```yaml
equations:
  - expression: flow_out <= $adjusted_flow_in
sub_expressions:
  adjusted_flow_in:
    - where: inheritance(storage)
      # main expression becomes `flow_out <= flow_in * flow_eff`
      expression: flow_in * flow_eff
    - where: inheritance(supply)
      # main expression becomes `flow_out <= flow_in * flow_eff * parasitic_eff`
      expression: flow_in * flow_eff * parasitic_eff
    - where: inheritance(conversion)
      # main expression becomes `flow_out <= flow_in * flow_eff * 0.3`
      expression: flow_in * flow_eff * 0.3
```

!!! note

    As with [equations][], `where` strings are mixed in together.
    If you have two equation expressions and three sub-expressions, each with two expressions, you will end up with 2*3*2 = 12 unique `where` strings with linked `expression` strings.

## slices

Similarly to [sub-expressions][], you can use references when [slicing your data][slicing-data], again using the `$` identifier.
Standard slicing only allows for dimensions to reference plain strings or lists of plain strings.
If you want to slice using a "lookup" parameter, you will need to provide it within the `slices` sub-key, e.g.:

If you define a lookup parameter "lookup_techs" as:

```yaml
parameters:
  lookup_techs:
    data: True
    index: [tech_1, tech_2]
    dimensions: [techs]
```

Then the following slice will select only the `tech_1` and `tech_2` members of `flow_out`:

```yaml
equations:
  - expression: sum(flow_out[carriers=electricity, techs=$tech_ref]) <= flow_in[carriers=heat] * 0.6
slices:
  tech_ref:
    - expression: lookup_techs
```