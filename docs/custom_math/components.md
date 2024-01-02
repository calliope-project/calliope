
# Math components

Here, we will briefly introduce each of the math components you will need to build an optimisation problem.
A more detailed description of the math YAML syntax is provided in the [math formulation schema][math-formulation-schema].

## Decision variables

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

## Global Expressions

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
1. It has [equations][] (and, optionally, [sub-expressions][] and [slices][]) with corresponding lists of `where`+`expression` dictionaries.
The equation expressions do _not_ have comparison operators; those are reserved for [constraints][]
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

## Constraints

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
1. It has [equations][] (and, optionally, [sub-expressions][] and [slices][]) with corresponding lists of `where`+`expression` dictionaries.
The equation expressions _must_ have comparison operators.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

## Objectives

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
1. It has [equations][] (and, optionally, [sub-expressions][] and [slices][]) with corresponding lists of `where`+`expression` dictionaries.
These expressions do _not_ have comparison operators.
1. It can be deactivated so that it does not appear in the built optimisation problem by setting `active: false`.

!!! warning

    You can only have one objective activated in your math.
    If you have loaded multiple, you can deactivate unwanted ones using `active: false`, or you can set your top-level `where` string on each that leads to only one being valid for your particular problem.