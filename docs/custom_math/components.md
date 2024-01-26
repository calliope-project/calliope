
# Math components

Here, we will briefly introduce each of the math components you will need to build an optimisation problem.
A more detailed description of the math YAML syntax is provided on the [math syntax](syntax.md) page and in the [math formulation schema][math-formulation-schema].

## Decision variables

Decision variables (called `variables` in Calliope) are the unknown quantities whose values can be chosen by the optimisation algorithm while optimising for the chosen objective (e.g. cost minimisation) under the bounds set by the constraints.
These include the output capacity of technologies, the per-timestep flow of carriers into and out of technologies or along transmission lines, and storage content in each timestep.
A decision variable in Calliope math looks like this:

```yaml
variables:
--8<-- "src/calliope/math/base.yaml:variable"
```

1. It needs a unique name (`storage_cap` in the example above).
1. Ideally, it has a long-form `description` and a `unit` added.
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

## Global Expressions

Global expressions are those combinations of decision variables and input parameters that you want access to in multiple constraints / objectives in the model.
You will also receive the result of the global expression as a numeric value in your optimisation results, without having to do any additional post-processing.

For instance, total costs are global expressions as the cost associated with a technology is not a _constraint_, but rather a linear combination of decision variables and parameters (e.g., `storage_cap * cost_storage_cap`).
To not clutter the objective function with all combinations of variables and parameters, we define a separate global expression:

```yaml
global_expressions:
--8<-- "src/calliope/math/base.yaml:expression"
```

Global expressions are by no means necessary to include, but can make more complex linear expressions easier to keep track of and can reduce post-processing requirements.

1. It needs a unique name (`cost` in the above example).
1. Ideally, it has a long-form `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, it becomes an un-indexed expression.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this expression.
1. It has [equations](syntax.md#equations) (and, optionally, [sub-expressions](syntax.md#sub-expressions) and [slices](syntax.md#slices)) with corresponding lists of `where`+`expression` dictionaries.
The equation expressions do _not_ have comparison operators; those are reserved for [constraints](#constraints)
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

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
1. Ideally, it has a long-form `description` and a `unit` added.
These are not required, but are useful metadata for later reference.
1. It can have a top-level `foreach` list and `where` string.
Without a `foreach`, it becomes an un-indexed constraint.
Without a `where` string, all valid members (according to the `definition_matrix`) based on `foreach` will be included in this constraint.
1. It has [equations](syntax.md#equations) (and, optionally, [sub-expressions](syntax.md#sub-expressions) and [slices](syntax.md#slices)) with corresponding lists of `where`+`expression` dictionaries.
The equation expressions _must_ have comparison operators.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

## Objectives

With your constrained decision variables and a global expression that binds these variables to costs, you need an objective to minimise/maximise. The default built-in objective is `min_cost_optimisation` and looks as follows:

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