
# Math components

Here, we will briefly introduce each of the math components you will need to build an optimisation problem.
A more detailed description of the math YAML syntax is provided on the [math syntax](syntax.md) page and in the [math formulation schema][math-formulation-schema].

## Decision variables

Decision variables (called `variables` in Calliope) are the unknown quantities whose values can be chosen by the optimisation algorithm while optimising for the chosen objective (e.g. cost minimisation) under the bounds set by the constraints.
These include the output capacity of technologies, the per-timestep flow of carriers into and out of technologies or along transmission lines, and storage content in each timestep.
A decision variable in Calliope math looks like this:

```yaml
variables:
--8<-- "src/calliope/math/plan.yaml:variable"
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
The default value should be set such that it has no impact on the optimisation problem if it is included (most of the time, this means setting it to zero).

## Global Expressions

Global expressions are those combinations of decision variables and input parameters that you want access to in multiple constraints / objectives in the model.
You will also receive the result of the global expression as a numeric value in your optimisation results, without having to do any additional post-processing.

For instance, total costs are global expressions as the cost associated with a technology is not a _constraint_, but rather a linear combination of decision variables and parameters (e.g., `storage_cap * cost_storage_cap`).
To not clutter the objective function with all combinations of variables and parameters, we define a separate global expression:

```yaml
global_expressions:
--8<-- "src/calliope/math/plan.yaml:expression"
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
The default value should be set such that it has no impact on the optimisation problem if it is included (most of the time, this means setting it to zero).

## Constraints

[Decision variables](#decision-variables) / [global expressions](#global-expressions) need to be constrained or included in the model objective.
Constraining these math components is where you introduce the realities of the system you are modelling.
This includes limits on things like the maximum area use of tech (there's only so much rooftop available for roof-mounted solar PV), and links between in/outflows such as how much carrier is consumed by a technology to produce each unit of output carrier.
Here is an example:

```yaml
constraints:
--8<-- "src/calliope/math/plan.yaml:constraint"
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
--8<-- "src/calliope/math/plan.yaml:objective"
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
