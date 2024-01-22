# Run modes

Calliope can leverage different methods to solve your optimisation problem.
By default, it is designed to find a system configuration with the lowest combined cost to invest in and then operate technologies, with complete knowledge of what the future holds.
This is a method that is known as "perfect foresight" optimisation.
We refer to it in our model as [`plan` mode][plan-mode].

In addition to perfect foresight optimisation, we have a [receding horizon "operate" optimisation mode][operate-mode] and [our "spores" mode][spores-mode] to generate alternative system configurations that are within a small deviation of the optimal cost that is computed in `plan` mode. Read on to find out more about each of these run modes.

## Plan mode

In `plan` mode, the user defines upper and lower boundaries for technology capacities and the model decides on an optimal system configuration.
In this configuration, the total cost of investing in technologies and then using them to meet demand in every _timestep_ (e.g., every hour) is as low as possible.

We scale investment costs so they are equivalent to time-varying (e.g., fuel and maintenance) costs by using _annualisation_.
With annualisation, we imagine having to take out a loan to pay for the technology investment.
The amount we pay per year for the technology is then the annual loan repayment.
This loan repayment is affected by its interest rate and the loan period.

$$
\frac{\text{investment cost} \times{} \text{interest rate} \times{} (1 + \text{interest rate})^\text{loan period}}{(1 + \text{interest rate})^\text{loan period} - 1}
$$

For instance, if we have a technology which will cost 2 million EUR to build and we take out a loan with a 10% interest rate over 25 years, then the annual cost of this technology will be:

$$
\frac{2 \times{} 0.1 \times{} 1.1^25}{1.1^25 - 1} = 0.22 \text{million EUR}
$$

In Calliope, we define interest rate and loan period using the parameters `cost_interest_rate` and `lifetime`, respectively.

## Operate mode

In `operate` mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm.
This is sometimes known as a `dispatch` model - we're only concerned with the _dispatch_ of technologies whose capacities are already fixed.

There are two main reasons to run in `operate` mode:

1. You can assess how well your system performs when it doesn't have perfect foresight over the entire time period.
Are your storage devices used appropriately when

To specify a valid `operate` mode model, capacities for all technologies at all locations must be defined.
This can be done by specifying `flow_cap`, `storage_cap`, `area_use`, `source_cap`, and `purchased_units` as _input parameters_.
These will not clash with the decision variables of the same name that are found in `plan` mode as `operate` mode will deactivate those decision variables.

Operate mode runs a model with a receding horizon control algorithm.
This requires two additional configuration options to be defined:

```yaml
config.build:
    operate_horizon: 48h  # (1)!
    operate_window: 24h
```

1. This is a pandas frequency string.
You can use any [frequency aliases](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases) to define your horizon and window.

`horizon` specifies how far into the future the control algorithm optimises in each iteration.
`window` specifies how many of the hours within `horizon` are actually kept in the results.
In the above example, decisions on how to operate for each 24-hour window are made by optimising over 48-hour horizons (i.e., the second half of each optimisation run is discarded).
For this reason, `horizon` must always be equal to or larger than `window`.

!!! warning
    You must define **all** your technology capacities as input parameters for the model to run successfully.

## SPORES mode

!!! warning
    SPORES mode has not yet been re-implemented in Calliope v0.7.

`SPORES` refers to Spatially-explicit Practically Optimal REsultS.
This run mode allows a user to generate any number of alternative results which are within a certain range of the optimal cost.
It follows on from previous work in the field of `modelling to generate alternatives` (MGA), with a particular emphasis on alternatives that vary maximally in the spatial dimension.
This run mode was developed for and implemented in a `study on the future Italian energy system <https://doi.org/10.1016/j.joule.2020.08.002>`_.

As an example, if you wanted to generate 10 SPORES, all of which are within 10% of the optimal system cost, you would define the following in your model configuration:

```yaml
config.build.mode: spores
config.solve:
    # The number of SPORES to generate:
    spores_number: 10
    # The cost class to optimise against when generating SPORES:
    spores_score_cost_class: spores_score
    # The initial system cost to limit the SPORES to fit within:
    spores_cost_max: .inf
    # The cost class to constrain to be less than or equal to `spores_cost_max`:
    spores_slack_cost_group: monetary
parameters:
    # The fraction above the cost-optimal cost to set the maximum cost during SPORES:
    slack: 0.1
```

You will now also need a `spores_score` cost class in your model.
The `spores_score` is the cost class against which the model optimises in the generation of SPORES.
The recommended approach is to initialise it in your model definition for all technologies and locations that you want to limit within the scope of finding alternatives.
Technologies at locations with higher scores will be penalised in the objective function, so are less likely to be chosen.
In the [national scale example model][national-scale-example-model], this would look something like:

```yaml
tech_groups:
    add_spores_score:
        inherit: cost_dim_setter
        cost_flow_cap:
            data: [null, null]
            index: ["monetary", "spores_score"]
            dims: costs
        cost_interest_rate:
            data: [0.1, 1]
            index: ["monetary", "spores_score"]
            dims: costs

techs:
    ccgt:
        inherit: add_spores_score
        cost_flow_cap.data: [750, 0]
    csp:
        inherit: add_spores_score
        cost_flow_cap.data: [1000, 0]
    battery:
        inherit: add_spores_score
        cost_flow_cap.data: [null, 0]
    region1_to_region2:
        inherit: add_spores_score
        cost_flow_cap.data: [10000, 0]
```

!!! note
    We use and recommend using 'spores_score' to define the cost class that you will now optimise against.
    However, it is user-defined, allowing you to choose terminology that best fits your use-case.
