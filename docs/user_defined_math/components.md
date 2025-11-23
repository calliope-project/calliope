
# Math components

Here, we will briefly introduce each of the math components you will need to build an optimisation problem.
A more detailed description of the math YAML syntax is provided on the [math syntax](syntax.md) page and in the [math formulation schema][model-math-schema].

## Dimensions

Dimensions define the possible index sets over which model components can be defined.
When you add custom math, you may need to define new dimensions that are not part of Calliope's built-in dimensions.

```yaml
dimensions:
  breakpoints:
    description: Breakpoints for piecewise linear constraints
    dtype: string
    ordered: false
    iterator: breakpoint
```

1. It needs a unique name (`breakpoints` in the above example).
1. Ideally, it has a long-form `title` and `description` added.
These are not required, but are useful metadata for later reference.
1. It can specify a `dtype` (data type) for the dimension items: `string`, `datetime`, `date`, `float`, or `integer` (default: `string`).
1. It can be marked as `ordered: true` if the order of dimension items is meaningful (e.g., chronological time). Default is `false`.
1. It requires an `iterator` string that defines the symbol to use in LaTeX math formulation for this dimension (e.g., `breakpoint` for breakpoints, `timestep` for timesteps).
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

!!! warning
    Dimension defined in math are metadata only.
    You still need to provide the actual dimension values through your model's data tables or YAML configuration.

## Parameters

Parameters are input data that your math formulation will reference.
While Calliope has many built-in parameters, you may need to define additional parameters for custom math.

```yaml
parameters:
  monthly_cost:
    description: Cost per unit capacity to apply per month
    default: 0
    unit: $\frac{\text{cost}}{\text{energy}}$
    resample_method: first
```

1. It needs a unique name (`monthly_cost` in the above example).
1. Ideally, it has a long-form `title`, `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
Unit information can be given in LaTeX math format for prettier rendering in the documentation.
1. It can have a `default` value (numeric: integer or float) that will be used where the parameter is not defined in the input data.
The default is `NaN` if not specified.
1. It can specify a `resample_method` (`mean`, `sum`, or `first`) which determines how to aggregate data if resampling is applied over any of the parameter's dimensions.
Default is `first`.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

!!! warning
    Parameters defined in math are metadata only.
    You still need to provide the actual parameter values through your model's data tables or YAML configuration.

## Lookups

Lookups allow you to create derived arrays, by selecting or transforming existing parameters, and allow you to mask arrays.
This is useful when you need to reference data under different index sets or apply complex conditions that have been computed in a preprocessing step.

```yaml
lookups:
  remote_node:
    description: For each transmission node pair at node A, get corresponding node B
    dtype: string
    default: .nan
    resample_method: first
  outages:
    description: Boolean timeseries where `True` indicates an outage
    dtype: bool
    default: False
    resample_method: first
```

1. It needs a unique name (`remote_node` in the above example).
1. Ideally, it has a long-form `title` and `description` added.
   These are not required, but are useful metadata for later reference.
1. It can specify a `dtype` (data type): `float`, `string`, `bool`, `datetime`, or `date` (default: `string`).
1. It can have a `default` value that will be used where the lookup is not defined in the input data.
The default can be a string, numeric value, boolean, or `NaN` (default: `NaN`).
1. It can specify a `resample_method` (`mean`, `sum`, or `first`) for data aggregation if resampling is applied. Default is `first`.
1. It can optionally define `one_of` as a list of allowed values that the lookup can take.
   This is used for pre-validation only.
1. It can use `pivot_values_to_dim` to pivot lookup values into a new dimension, converting the array to boolean values.
   For example, if a lookup over `techs` has values `[electricity, gas]` and `pivot_values_to_dim: carriers`, it becomes a boolean array over `[techs, carriers]`.
1. It can be deactivated by setting `active: false`.

!!! warning
    Lookups defined in math are metadata only.
    You still need to provide the actual lookup values through your model's data tables or YAML configuration.

## Decision variables

Decision variables (called `variables` in Calliope) are the unknown quantities whose values can be chosen by the optimisation algorithm while optimising for the chosen objective (e.g. cost minimisation) under the bounds set by the constraints.
These include the output capacity of technologies, the per-timestep flow of carriers into and out of technologies or along transmission lines, and storage content in each timestep.
A decision variable in Calliope math looks like this:

```yaml
variables:
--8<-- "src/calliope/math/base.yaml:variable"
```

1. It needs a unique name (`storage_cap` in the example above).
1. Ideally, it has a long-form name (`title`), a `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, it becomes an un-indexed variable.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this decision variable.
1. It can define a domain to turn it into a binary or integer variable (in either of those cases, domain becomes `integer`).
1. It requires a minimum and maximum bound, which can be:
    1. a numeric value:
    ```yaml
    variables:
      flow_out:
      ...
        bounds:
          min: 0
          max: .inf
    ```
    1. a reference to an input parameter, where each valid member of the variable (i.e. each value of the variable for a specific combination of indices) will get a different value based on the values of the referenced parameters (see example above).
    If a value for a valid variable member is undefined in the referenced parameter, the decision variable will be unbounded for this member.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.
1. It can take on a `default` value that will be used in math operations to avoid `NaN` values creeping in.
The default value should be set such that it has no impact on the optimisation problem if it is included (most of the time, this means `NaN`).

## Global Expressions

Global expressions are those combinations of decision variables and input parameters (_and_ other global expressions!) that you want access to in multiple constraints / objectives in the model.
You will also receive the result of the global expression as a numeric value in your optimisation results, without having to do any additional post-processing.

For instance, total costs are global expressions as the cost associated with a technology is not a _constraint_, but rather a linear combination of decision variables and parameters (e.g., `storage_cap * cost_storage_cap`).
To not clutter the objective function with all combinations of variables and parameters, we define a separate global expression:

```yaml
global_expressions:
--8<-- "src/calliope/math/base.yaml:expression"
```

Global expressions are by no means necessary to include, but can make more complex linear expressions easier to keep track of and can reduce post-processing requirements.

1. It needs a unique name (`cost` in the above example).
1. Ideally, it has a long-form name (`title`), a `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, it becomes an un-indexed expression.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this expression.
1. It has [equations](syntax.md#equations) (and, optionally, [sub-expressions](syntax.md#sub-expressions) and [slices](syntax.md#slices)) with corresponding lists of `where`+`expression` dictionaries.
The equation expressions do _not_ have comparison operators; those are reserved for [constraints](#constraints)
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.
1. It can take on a `default` value that will be used in math operations to avoid `NaN` values creeping in.
The default value should be set such that it has no impact on the optimisation problem if it is included (most of the time, this means `NaN`).
1. It can have an `order` defined to reprioritise its addition to the optimisation problem.
   This is often necessary when adding new global expressions that you will use in other global expressions since they will need to be defined _before_ the other global expressions that refer to them.

??? example "Re-ordering global expressions"
    If tracking a new cost as a global expression - `cost_operation_monthly` - we will want to add it to the overall  `cost` global expression so it is tracked in the objective:

    ```yaml
    global_expressions:
      # Add our new expression.
      cost_operation_monthly:
        ...

      # Add our new expression to the overall `cost` expression.
      # This only requires updating the where string and main equation in `cost`; the rest will be inherited from the   pre-defined math.
      cost:
        where: "cost_operation_monthly OR cost_investment_annualised OR cost_operation_variable OR cost_operation_fixed"
        equations:
          - expression: >-
            cost_investment_annualised +
            $cost_operation_sum +
            cost_operation_fixed +
            cost_operation_monthly
    ```

    For `cost` to know that `cost_operation_monthly` exists, the latter needs to be defined first.
    Although we have defined `cost_operation_monthly` in the above example, because `cost` is being _updated_ from the pre-defined math, its order will reflect its position in the  pre-defined math, i.e. higher than `cost_operation_monthly`.
    To bring the order of `cost_operation_monthly` higher we can define `order` as a suitably small number.
    Usually, a negative number is used (e.g. `-1`) to ensure we move it to before _all_ pre-defined global expressions:

    ```yaml
    global_expressions:
      cost_operation_monthly:
        order: -1
        ...
      cost:
        ...
    ```

## Constraints

[Decision variables](#decision-variables) / [global expressions](#global-expressions) need to be constrained or included in the model objective.
Constraining these math components is where you introduce the realities of the system you are modelling.
This includes limits on things like the maximum area use of tech (there's only so much rooftop available for roof-mounted solar PV), and links between in/outflows such as how much carrier is consumed by a technology to produce each unit of output carrier.
Here is an example:

```yaml
constraints:
--8<-- "src/calliope/math/base.yaml:constraint"
```

1. It needs a unique name (`set_storage_initial` in the above example).
1. Ideally, it has a long-form `description` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, it becomes an un-indexed constraint.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this constraint.
1. It has [equations](syntax.md#equations) (and, optionally, [sub-expressions](syntax.md#sub-expressions) and [slices](syntax.md#slices)) with corresponding lists of `where`+`expression` dictionaries.
The equation expressions _must_ have comparison operators.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

## Piecewise constraints

If you have non-linear relationships between two decision variables, you may want to represent them as a [piecewise linear function](https://en.wikipedia.org/wiki/Piecewise_linear_function).
The most common form of a piecewise function involves creating special ordered sets of type 2 (SOS2), set of binary variables that are linked together with specific constraints.

!!! note
    You can find a fully worked-out example in our [piecewise linear tutorial][defining-piecewise-linear-constraints].

Because the formulation of piecewise constraints is so specific, the math syntax differs from all other modelling components by having `x` and `y` attributes that need to be specified:

```yaml
piecewise_constraints:
  sos2_piecewise_flow_out:
    description: Set outflow to follow capacity according to a piecewise curve.
    foreach: [nodes, techs, carriers]
    where: piecewise_x AND piecewise_y
    x_expression: flow_cap
    x_values: piecewise_x
    y_expression: flow_out
    y_values: piecewise_y
    active: true
```

1. It needs a unique name (`sos2_piecewise_flow_out` in the above example).
1. Ideally, it has a long-form `description` added.
This is not required, but is useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, it becomes an un-indexed constraint.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this constraint.
1. It has `x` and `y` [expression strings](syntax.md#expression-strings) (`x_expression`, `y_expression`).
1. It has `x` and `y` parameter references (`x_values`, `y_values`).
This should be a string name referencing an input parameter that contains the `breakpoints` dimension.
The values given by this parameter will be used to set the respective (`x` / `y`) expression at each breakpoint.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

The component attributes combine to describe a piecewise curve that links the `x_expression` and `y_expression` according to their respective values in `x_values` and `y_values` at each breakpoint.

!!! note
    If the non-linear function you want to represent is convex, you may be able to avoid SOS2 variables, and instead represent it using [constraint components](#constraints).
    You can find an example of this in our [piecewise linear costs custom math example][piecewise-linear-costs].

!!! warning
    This approximation of a non-linear relationship may improve the representation of whatever real system you are modelling, but it will come at the cost of a more difficult model to solve.
    Indeed, introducing piecewise constraints may mean your model can no longer reach a solution with the computational resources you have available.

## Objectives

With your constrained decision variables and a global expression that binds these variables to costs, you need an objective to minimise/maximise. The default, pre-defined objective is `min_cost_optimisation` and looks as follows:

```yaml
objectives:
--8<-- "src/calliope/math/base.yaml:objective"
```

1. It needs a unique name.
1. Ideally, it has a long-form `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `where` string, but no `foreach` (it is a single value you need to minimise/maximise).
Without a `where` string, the objective will be activated.
1. It has [equations](syntax.md#equations) (and, optionally, [sub-expressions](syntax.md#sub-expressions) and [slices](syntax.md#slices)) with corresponding lists of `where`+`expression` dictionaries.
These expressions do _not_ have comparison operators.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

!!! warning
    You can only have one objective function activated in your math.
    If you have defined multiple objective functions, you can deactivate unwanted ones using `active: false`, or you can set your top-level `where` string on each that leads to only one being valid for your particular problem.

## Postprocessed expressions

Postprocessed expressions allow you to compute additional results after the model has been solved.
These are similar to [global expressions](#global-expressions) but are calculated using the optimised values of decision variables and global expressions, rather than being part of the optimisation problem itself.

```yaml
postprocessed:
  annual_energy:
    description: Total annual energy output
    unit: $\frac{\text{energy}}{\text{year}}$
    foreach: [nodes, techs, carriers]
    where: flow_out
    equations:
      - expression: sum(flow_out, over=timesteps) * timestep_resolution
```

1. It needs a unique name (`annual_energy` in the above example).
1. Ideally, it has a long-form `title`, `description` and a `unit` added.
   These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
   Without a `foreach`, it becomes an un-indexed expression.
   Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this expression.
1. It has [equations](syntax.md#equations) (and, optionally, [sub-expressions](syntax.md#sub-expressions) and [slices](syntax.md#slices)) with corresponding lists of `where`+`expression` dictionaries.
1. It can be deactivated so that it does not appear in the results by setting `active: false`.
1. It can take on a `default` value that will be used in calculations to avoid `NaN` values creeping in.
   This is most useful when using one postprocessed expression as an input to another postprocessed expression.
1. It can have an `order` defined to reprioritise its calculation.
  This is often necessary when adding new postprocessed expressions that you will use in other postprocessed expressions since they will need to be calculated _before_ the other postprocessed expressions that refer to them.

!!! note
    Postprocessed expressions are calculated _after_ the optimisation is complete and do not affect the optimisation problem.
    They are useful for deriving additional metrics, aggregating results, or computing performance indicators from the optimised solution.

??? example "Re-ordering postprocessed expressions"
    If you want to calculate capacity factors as a postprocessed expression and then use those to calculate weighted averages:

    ```yaml
    postprocessed:

      # Use capacity factor in another postprocessed expression
      weighted_capacity_factor:
        foreach: [techs]
        equations:
          - expression: sum(capacity_factor * flow_cap, over=nodes) / sum(flow_cap, over=nodes)

      # Defined after, but order forces capacity factor to be calculated _first_
      capacity_factor:
        order: -1
        foreach: [nodes, techs]
        where: flow_cap
        equations:
          - expression: sum(flow_out, over=timesteps) / (flow_cap * timestep_resolution)
    ```

## Checks

Checks are validation rules that ensure your model configuration is valid before building the optimisation problem.
They can raise errors (which stop model building) or warnings (which alert you to potential issues but allow the model to continue).

```yaml
checks:
  unbounded_flow_cap_cost:
    where: cost_flow_cap<0
    message: >
      Technologies with negative `cost_flow_cap` must have `flow_cap_max` defined
      to avoid unbounded optimization problems.
    errors: raise
```

1. It needs a unique name (`unbounded_flow_cap_cost` in the above example).
1. It requires a `where` string that defines the condition to check.
   If this condition evaluates to `True` for any model components, the check is triggered.
1. It requires a `message` string that will be shown to users when the check is triggered.
   This should clearly explain what the issue is and how to fix it.
1. It has an `errors` field that determines the severity (default: `raise`):
    - `raise`: Will raise an error and stop model building.
    - `warn`: Will issue a warning but allow model building to continue.
1. It can be deactivated by setting `active: false`.

Checks are evaluated after the model definition is loaded but before the backend optimization problem is built.
This allows you to catch configuration issues early and provide helpful error messages to users.
