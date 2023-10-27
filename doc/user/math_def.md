# Custom math formulation

Since Calliope version 0.7, The math used to build optimisation problems is stored in YAML files.

The same syntax used for the [inbuilt math](https://github.com/calliope-project/calliope/tree/main/calliope/math) can be used to define custom math.
So, if you want to introduce new constraints, decision variables, or objectives, you can do so as part of the collection of YAML files describing your model.

In brief, components of the math formulation are stored under named keys and contain information on the sets over which they will be generated (e.g., for each technology, node, timestep, ...), the conditions under which they will be built in any specific model (e.g., if no `storage` technologies exist in the model, decision variables and constraints associated with them will not be built), and their math expression(s).

In this section, we will describe the [math components][math-components] and the formulation syntax in more detail.
Whenever we refer to a "math component" it could be a:
    - decision variable (something you want the optimisation model to decide on the value of).
    - global expression (a mixture of decision variables and input parameters glued together with math).
    - constraint (a way to limit the upper/lower bound of a decision variable using other decision variables/parameters/global expressions).
    - objective (the expression whose value you want to minimise/maximise in the optimisation).

At the end of the section you will find a full reference for the allowed key:value pairs in your custom math YAML file.

!!! note

    Although we have tried to make a generalised syntax for all kinds of custom math, our focus was on reimplementing the base math.
    Unfortunately, we cannot guarantee that your math will be possible to implement.

!!! warning

    When writing custom math, remember that Calliope is a _linear_ modelling framework. It is possible that your desired math will create a nonlinear optimisation problem.
    Usually, the solver will provide a clear error message when this is the case, although it may not be straightforward to pinpoint what part of your math is the culprit.


## Math syntax

### foreach lists

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

!!! examples:

        * If you want to apply a constraint across all `nodes` and `techs`, but only for node+tech combinations where the `flow_eff` parameter has been defined, you would include `flow_eff`.
        * If you want to apply a constraint over `techs` and `timesteps`, but only for combinations where the `source_max` parameter has at least one `node` with a value defined, you would include `any(resource, over=nodes)` (1).
        { .annotate }

        1. I'm a [helper function][helper-functions]; read more below!

1. Checking the value of a configuration option or an input parameter.
Checks can use any of the operators: `>`, `<`, `=`, `<=`, `>=`.
Configuration options are any that are defined in `config.build`, where you can define your own options to access in the `where` string.

!!! examples:

    * If you want to apply a constraint only if the configuration option `config.build.cyclic_storage` is _True_, you would include `run.cyclic_storage=True` (`True`/`False` is case insensitive).
    * If you want to apply a constraint across all `nodes` and `techs`, but only where the `flow_eff` parameter is less than 0.5, you would include `flow_eff<0.5`.
    * If you want to apply a constraint only for the first timestep in your timeseries, you would include `timesteps=get_val_at_index(dim=timesteps, idx=0)` (1).
    { .annotate }
    * If you want to apply a constraint only for the last timestep in your timeseries, you would include `timesteps=get_val_at_index(dim=timesteps, idx=-1)` (1).
    { .annotate }

    1. I'm a [helper function][helper-functions]; read more below!

1. Checking the inheritance chain of a technology up to and including the [][abstract_base_tech_definitions].
This uses a [helper function][helper-functions], which we go into more detail on below.

!!! examples:

    * If you want to create a decision variable across only `storage` technologies, you would include `inheritance(storage)`.
    * If you want to apply a constraint across only your own `rooftop_supply` technologies (e.g., you have assigned the tech_group `rooftop_supply` as the parent of your technologies `pv` and `solar_thermal`), you would include `inheritance(rooftop_supply)`.

1. Subsetting a set.
The sets available to subset are always [`nodes`, `techs`, `carriers`, `carrier_tiers`] + any additional sets defined by you in [`foreach`][foreach-lists].

!!! examples:

    * If you want to create a decision variable for each `carrier` in the model but only if they are _output_ carriers of technologies, you would include `[out, out_2, out_3] in carrier_tiers`.
    * If you want to filter `nodes` where any of a set of `tech`s are defined: `defined(techs=[tech1, tech2], within=nodes, how=any)` (1).
    { .annotate }

    1. I'm a [helper function][helper-functions]; read more below!

To combine statements you can use the operators `and`/`or`.
You can also use the `not` operator to negate any of the statements.
These operators are case insensitive, so "and", "And", "AND" are equivalent.
You can group statements together using the `()` brackets.
These statements will be combined first.

!!! examples:

    * If you want to apply a constraint for `storage` technologies if the configuration option `cyclic_storage` is activated and it is the last timestep of the series: `inheritance(storage) and run.cyclic_storage=True and timesteps=get_val_at_index(dim=timesteps, idx=-1)`.
    * If you want to create a decision variable for the input carriers of conversion technologies: `[in] in carrier_tiers and inheritance(conversion)`
    * If you want to apply a constraint if the parameter `resource_unit` is `energy_per_area` or the parameter `resource_area_per_energy_cap` is defined: `resource_unit=energy_per_area or resource_area_per_energy_cap`.
    * If you want to apply a constraint if the parameter `flow_eff` is less than or equal to 0.5 and `resource` has been defined, or `flow_eff` is greater than 0.9 and `resource` has not been defined: `(flow_eff<=0.5 and resource) or (flow_eff>0.9 and not resource)`.

Combining `foreach` and `where` will create an n-dimensional boolean array.
Wherever index items in this array are _True_, your component `expression(s)` will be applied.

### expression strings

As with where strings, expression strings are a series of math terms combined with operators.
The terms can be input parameters, decision variables, global expressions, or numeric values that you define on-the-fly.

If you are defining a `global expression` or `objective`, then the available expression string operators are: `+`, `-`, `*`, `/`, and `**` ("to the power of").
These expressions are applied using standard operator precedence (BODMAS/PEMDAS, see [this wiki](https://en.wikipedia.org/wiki/Order_of_operations) for more info).

If you are defining a `constraint`, then you also need to define a comparison operator: `<=`, `>=`, or `==`.

!!! examples:

    * If you want to limit all technology outflow to be less than 200 units: `flow_out <= 200`.
    * If you want to create a global expression which is the storage level minus a parameter defining a minimum allowed storage level: `storage - storage_cap * min_storage_level`.
    * If you want to set the outflow of a specific technology `my_tech` to equal all outflows of a specific carrier `my_carrier` at each node: `flow_out[techs=my_tech] == sum(flow_out[carriers=my_carrier], over=techs)`.
    * If you want inflows at a node `my_node` to be at least as much as the inflows in the previous timestep: `flow_in[nodes=my_node] >= roll(flow_in[nodes=my_node], timesteps=1)`.

#### Slicing data

You do not need to define the sets of math components in expressions, unless you are actively "slicing" them.
Behind the scenes, we will make sure that every relevant element of the defined `foreach` sets are matched together when applying the expression.
Slicing math components involves appending the component with square brackets that contain the slices, e.g. `flow_out[carriers=electricity, nodes=[A, B]]` will slice the `flow_out` decision variable to focus on `electricity` in its `carriers` dimension and only has two nodes (`A` and `B`) on its `nodes` dimension.
To find out what dimensions you can slice a component on, see your input data (`model.inputs`) for parameters and the definition for decision variables in your loaded math dictionary (`model.math.variables`).

#### Helper functions

As with [`where` strings][where-strings], there are many [helper function][helper-functions] available to use in expressions, to allow for more complex operations to be undertaken.
Some of these helper functions require a good understanding of their functionality to apply, so make sure you are comfortable with them before using them.

### `equations`

Equations are combinations of [][expression-strings] and [][where-strings].
You define one or more equations for your model components.
A different `where` string associated with each equation expression allows you to slightly alter the expression for different component members.
You define equations as lists of dictionaries:

    ```yaml
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

!!! examples

    * Divide by efficiency if efficiency is larger than zero, otherwise set the variable to zero:

      ```yaml
      equations:
        - where: flow_eff > 0
          expression: flow_out / flow_out_eff == flow_in
        - where: flow_eff = 0
          expression: flow_out == 0
      ```

    * Limit flow by storage_cap, if it is defined, otherwise by flow_cap:

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

### `sub-expressions`

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

    As with [][equations], `where` strings are mixed in together.
    If you have two equation expressions and three sub-expressions, each with two expressions, you will end up with 2*3*2 = 12 unique `where` strings with linked `expression` strings.

### slices

    Similarly to [][sub-expressions], you can use references when [slicing your data][slicing-data], again using the `$` identifier.
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

## Math components

Here, we will briefly introduce each of the math components you will need to build an optimisation problem.
A more detailed description of the math YAML syntax is provided in the rendered [][math-schema].

### Decision variables

Decision variables (also known as `variables`) are why you are here in the first place.
They are the unknown quantities whose values will decide the value of the objective you are trying to minimise/maximise under the bounds set by the constraints.
These include the output capacity of technologies, the per-timestep flow of carriers into and out of technologies or along transmission lines, and storage content in each timestep.
A decision variable in Calliope math looks like this:

    ```yaml
    variables:
      storage_cap:
        description: "The upper limit on carriers that can be stored by a `supply_plus` or `storage` technology in any timestep."
        unit: carrier_unit
        foreach: [nodes, techs]
        where: "include_storage=True"
        domain: real  # optional; defaults to real.
        bounds:
          min: storage_cap_min
          max: storage_cap_max
        active: true
    ```

1. It needs a unique name.
1. Ideally, it has a long-form `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, this becomes an un-indexed variable.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this decision variable.
1. It can define a domain to set a binary or integer variable (in either case, domain becomes `integer`).
1. It requires a minimum and maximum bound, which can be:
    a. a numeric value:

        ```yaml
        variables:
          flow_out:
          ...
            bounds:
              min: 0
              max: .inf
        ```
    b. a reference to an input parameter, where each valid member of the component will get a different value (see example above).
    If a value for a valid component member is undefined in the referenced parameter, the decision variable will be unbounded for this member.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

### Global Expressions

Global expressions are those combinations of decision variables and input parameters that you want access to in multiple constraints / objectives in the model.
You will also receive the result of the global expression as a numeric value in your optimisation results, without having to do any additional post-processing.

For instance, total costs are global expressions as the cost associated with a technology is not a _constraint_, but rather a linear combination of decision variables and parameters (e.g., `storage_cap * cost_storage_cap`).
To not clutter the objective function with all combinations of variables and parameters, we define a separate global expression:

    ```yaml
    cost:
      description: "The total annualised costs of a technology, including installation and operation costs."
      unit: cost
      foreach: [nodes, techs, costs]
      where: "cost_investment OR cost_var"
      equations:
        - expression: $cost_investment + $cost_var_sum
      sub_expressions:
        cost_investment:
          - where: "cost_investment"
            expression: cost_investment
          - where: "NOT cost_investment"
            expression: "0"
        cost_var_sum:
          - where: "cost_var"
            expression: sum(cost_var, over=timesteps)
          - where: "NOT cost_var"
            expression: "0"
      active: true
    ```
Global expressions are by no means necessary to include, but can make more complex linear expressions easier to keep track of and can reduce post-processing requirements.

1. It needs a unique name.
1. Ideally, it has a long-form `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, this becomes an un-indexed variable.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this decision variable.
1. It has [][equations] (and, optionally, [][sub-expressions] and [][slices]) with corresponding lists of `where`+`expression` dictionaries.
The equation expressions do _not_ have comparison operators; those are reserved for [][constraints]
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

### Constraints

[Decision variables][decision-variables] / [global expressions][global-expressions] need to be constrained or included in the model objective.
Constraining these math components is where you introduce the realities of the system you are modelling.
This includes limits on things like the maximum area use of tech (there's only so much rooftop available for roof-mounted solar PV), and links between in/outflows such as how much carrier is consumed by a technology to produce each unit of output carrier.

    ```yaml
    set_storage_initial:
      description: "Fix the relationship between carrier stored in a `storage` technology at the start and end of the whole model period."
      foreach: [nodes, techs]
      where: "storage AND storage_initial AND config.cyclic_storage=True"
      equations:
        - expression: storage[timesteps=$final_step] * ((1 - storage_loss) ** timestep_resolution[timesteps=$final_step]) == storage_initial * storage_cap
      slices:
        final_step:
          - expression: get_val_at_index(timesteps=-1)
      active: true
    ```

1. It needs a unique name.
1. Ideally, it has a long-form `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, this becomes an un-indexed variable.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this decision variable.
1. It has [][equations] (and, optionally, [][sub-expressions] and [][slices]) with corresponding lists of `where`+`expression` dictionaries.
The equation expressions _must_ have comparison operators.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

### Objectives

With your constrained decision variables and a global expression that binds these variables to costs, you need an objective to minimise/maximise:

    ```yaml
    objectives:
      minmax_cost_optimisation:
        description: >
            Minimise the total cost of installing and operation all technologies in the system.
            If multiple cost classes are present (e.g., monetary and co2 emissions), the weighted sum of total costs is minimised.
            Cost class weights can be defined in the top-level parameter `objective_cost_class`.
        equations:
          - where: "any(cost, over=[nodes, techs, costs])"
            expression: sum(sum(cost, over=[nodes, techs]) * objective_cost_class, over=costs) + $unmet_demand
          - where: "NOT any(cost, over=[nodes, techs, costs])"
            expression: $unmet_demand
        sub_expressions:
          unmet_demand:
            - where: "config.ensure_feasibility=True"
              expression: sum(sum(unmet_demand - unused_supply, over=[carriers, nodes])  * timestep_weights, over=timesteps) * bigM
            - where: "NOT config.ensure_feasibility=True"
              expression: "0"
        sense: minimise
        active: true
    ```

1. It needs a unique name.
1. Ideally, it has a long-form `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `where` string, but no `foreach` (it is a single value you need to minimise/maximise).
Without a `where` string, the objective will be activated.
1. It has [][equations] (and, optionally, [][sub-expressions] and [][slices]) with corresponding lists of `where`+`expression` dictionaries.
These expressions do _not_ have comparison operators.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

!!! warning

    You can only have one objective activated in your math.
    If you have loaded multiple, you can deactivate unwanted ones using `active: false`, or you can set your top-level `where` string on each that leads to only one being valid for your particular problem.

## Introducing custom math to your model

Once you have read this page and the corresponding [][math-schema], you'll be ready to introduce custom math to your model.
You can find examples of custom math that we have put together in [these example notebooks][TODO: fill me].

Whenever you introduce your own math, it will be applied on top of the [base math][link to base math].
Therefore, you can include base math overrides as well as add new math.
For example, if you want to introduce a timeseries parameter to limiting maximum storage capacity:

    ```yaml
    storage_max:
      equations:
        - expression: storage <= storage_cap<span style="color: green;"> * time_varying_parameter<span style="color: green;">
    ```

The other elements of the `storage_max` constraints have not changed (`foreach`, `where`, ...), so we do not need to define them again when overriding.
