# Mixed Integer Linear Programming (MILP) example model

This example is based on the Urban scale example model, but with an override to introduce binary and integer variables.
This override is applied from the `scenarios.yaml` file:

```yaml
--8<-- "src/calliope/example_models/urban_scale/scenarios.yaml:milp"
```

!!! note
    MILP functionality can be easily applied, but convergence is slower as a result of integer/binary variables.
    It is recommended to use a commercial solver (e.g. Gurobi, CPLEX) if you wish to utilise these variables outside this example model.

## Model definition

We will only discuss the components of the model definition that differ from the urban scale example model.
Refer to that tutorial page for more information on this model.

### Units

The capacity of a technology is usually a continuous decision variable, which can be within the range of 0 and `flow_cap_max` (the maximum capacity of a technology).
In this model, we introduce a unit limit on the CHP instead:

```yaml
--8<-- "src/calliope/example_models/urban_scale/scenarios.yaml:chp"
```

A unit maximum allows a discrete, integer number of CHP to be purchased, each having a capacity of `flow_cap_per_unit`.
`flow_cap_max` and `flow_cap_min` are now ignored, in favour of `units_max` or `units_min`.

A useful feature unlocked by introducing this is the ability to set a minimum operating capacity which is *only* enforced when the technology is operating.
In the LP model, `flow_out_min_relative` would force the technology to operate at least at that proportion of its maximum capacity at each time step.
In this model, the newly introduced `flow_out_min_relative` of 0.2 will ensure that the output of the CHP is 20% of its maximum capacity in any time step in which it has a _non-zero output_.

### Purchase cost

The boiler does not have a unit limit, it still utilises the continuous variable for its capacity. However, we have introduced a `purchase` cost:

```yaml
--8<-- "src/calliope/example_models/urban_scale/scenarios.yaml:boiler"
```

By introducing this, the boiler is now associated with a binary decision variable.
It is 1 if the boiler has a non-zero `flow_cap` (i.e. the optimisation results in investment in a boiler) and 0 if the capacity is 0.

The purchase cost is applied to the binary result, providing a fixed cost on purchase of the technology, irrespective of the technology size.
In physical terms, this may be associated with the cost of pipework, land purchase, etc.
The purchase cost is also imposed on the CHP, which is applied to the number of integer CHP units in which the solver chooses to invest.

### Asynchronous flow in/out

The heat pipes which distribute thermal energy in the network may be prone to dissipating heat in an "unphysical" way.
I.e. given that they have distribution losses associated with them, in any given timestep a link could produce and consume energy simultaneously.
It would therefore lose energy to the atmosphere, but have a net energy transmission of zero.

This allows e.g. a CHP facility to overproduce heat to produce more cheap electricity, and have some way of dumping that heat.
The `async_flow_switch` binary variable (triggered by the `force_async_flow` parameter) ensures this phenomenon is avoided:

```yaml
--8<-- "src/calliope/example_models/urban_scale/scenarios.yaml:heat_pipes"
```

Now, only one of `flow_out` and `flow_in` can be non-zero in a given timestep.
This constraint can also be applied to storage technologies, to similarly control charge/discharge.

---

To try loading and solving the model yourself, see the accompanying notebook [here][running-the-milp-example-model]
