# Pre-defined math

As of Calliope version 0.7, the math used to build optimisation problems is stored in YAML files.
Our pre-defined math is a re-implementation of the formerly hardcoded math formulation in this YAML format.

Calliope follows a strict order of priority when applying math: **base math -> mode math -> extra math**.

* **Base math**, defined in `config.init.base_math`, is _always_ applied to any Calliope problem.
By default, it is a 'snapshot' [capacity planning problem][base-math] with perfect foresight.
* **Mode math**, activated via `config.build.mode`, is reserved for special cases that require additional processing within Calliope.
This includes [operate](#operate-mode) and [spores](#spores-mode) modes.
* **Extra math**, activated via `config.build.extra_math`, is for additional formulations that can be _optionally_ loaded on top of the base math.
For instance, the [inter-cluster storage][inter-cluster-storage-math] extra math allows you to track storage levels in technologies more accurately when you are using timeseries clustering in your model.

To load optional pre-defined math on top of the base math, you can reference it by name in your model configuration.

!!! example
    In the example below, Calliope will first apply all the [`base`][base-math] pre-defined math, then the [`storage_inter_cluster`][inter-cluster-storage-math] pre-defined math.

    ```yaml
    config:
    build:
        extra_math: [storage_inter_cluster]
    ```

All pre-defined math YAML files can be found in [`math` directory of the Calliope source code](https://github.com/calliope-project/calliope/blob/main/src/calliope/math/storage_inter_cluster.yaml).

If you want to introduce new constraints, decision variables, or objectives, you can do so as part of the collection of YAML files describing your model.
See the [user-defined math](../user_defined_math/index.md) section for an in-depth guide to applying your own math.

## Build modes

Calliope can leverage different methods to solve your optimisation problem.
By default, it is designed to find a system configuration with the lowest combined cost to invest in and then operate technologies, with complete knowledge of what the future holds.
This is a method that is known as "perfect foresight" optimisation.
We refer to it in our model as [`base` mode](#base-mode).

In addition to perfect foresight optimisation, we have a [receding horizon "operate" optimisation mode](#operate-mode) and [our "spores" mode](#spores-mode) to generate alternative system configurations that are within a small deviation of the optimal cost that is computed in `base` mode. Read on to find out more about each of these run modes.

### Base mode

In `base` mode, the user defines upper and lower boundaries for technology capacities and the model decides on an optimal system configuration.
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

### Operate mode

In `operate` mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm.
This is sometimes known as a `dispatch` model - we're only concerned with the _dispatch_ of technologies whose capacities are already fixed.

There are two main reasons to run in `operate` mode:

1. You can assess how well your system performs when it doesn't have perfect foresight over the entire time period.
Are your storage devices used appropriately when

To specify a valid `operate` mode model, capacities for all technologies at all locations must be defined.
This can be done by specifying `flow_cap`, `storage_cap`, `area_use`, `source_cap`, and `purchased_units` as _input parameters_.
These will not clash with the decision variables of the same name that are found in `base` mode as `operate` mode will deactivate those decision variables.

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

### SPORES mode

`SPORES` refers to Spatially-explicit Practically Optimal REsultS.
This run mode allows a user to generate any number of alternative results which are within a certain range of the optimal cost.
It follows on from previous work in the field of `modelling to generate alternatives` (MGA), with a particular emphasis on alternatives that vary maximally in the spatial dimension.
This run mode was developed for and first implemented in a [study on the future Italian energy system](https://doi.org/10.1016/j.joule.2020.08.002). We later expanded it to enable greater [computational efficiency and versatility](https://doi.org/10.1016/j.apenergy.2023.121002) in what kind of alternative results are prioritised.

As an example, if you wanted to generate 10 SPORES, all of which are within 10% of the optimal system cost, you would define the following in your model configuration:

```yaml
config.build.mode: spores
# The number of SPORES to generate:
config.solve.spores.number: 10
# The fraction above the cost-optimal cost to set the maximum cost during SPORES:
parameters.spores_slack: 0.1
```

To get a glimpse of how the results generated via SPORES compare to simple cost optimisation, check out our documentation
on [comparing run modes](../examples/modes.py).

#### Limiting the search for alternatives to specific technologies

By default, all technologies at all nodes will be subject to SPORES scoring.
This means that they are all equally likely to be removed from the system when generating alternative system designs.
You may want to instead focus on specific technologies when searching for alternatives.
To do so, set a _tracking parameter_.
This will be a parameter you set to `True` for all technologies that you want SPORES scores to be applied to.
Any technology _not_ being tracked will not be penalised in the optimisation for having a non-zero capacity.

!!! example

    ```yaml
    config.solve.spores.tracking_parameter: my_tracking_parameter
    parameters:
      my_tracking_parameter: # defines which techs are going to be subject to SPORES scoring
        data: [true, true, true]
        index: [ccgt, csp, battery]
        dims: techs
    ```

    Or, at the technology level:

    ```yaml
    config.solve.spores.tracking_parameter: my_tracking_parameter
    techs:
      ccgt:
        my_tracking_parameter: true
      csp:
        my_tracking_parameter: true
      battery:
        my_tracking_parameter: true
      pv:
        my_tracking_parameter: false # will not be tracked
    ```

#### Saving results per SPORE

Optimisation runs can be resource intensive.
You should not lose all your results if the optimisation fails part way through your SPORES runs.
To mitigate this, you can _save results per SPORE run_ to capture results up to any point of failure.

!!! example

    ```yaml
    config.build.mode: spores
    # The number of SPORES to generate:
    config.solve.spores:
      number: 10
      save_per_spore_path: results/spores
    # The fraction above the cost-optimal cost to set the maximum cost during SPORES:
    parameters.spores_slack: 0.1
    ```

    Here, 11 result files will be present in the results, 1 for the baseline (non-SPORES) run + 10 SPORES runs.
    Each SPORE run will be labelled `spore_<run number>.nc`, e.g. `results/spores/spore_10.nc`.

!!! note
    - The `save_per_spore_path` directory path will be considered as relative to the current working directory unless given as an absolute path.
    - Even if you choose to save results per SPORE, they will also be stored in memory.
      Then, following the successful completion of all SPORES runs, all results will be concatenated and available in the `model.results` dataset.

#### Skipping the baseline optimisation run

The `baseline` run does not set any SPORES scores or a system cost slack.
It is equivalent to running the model in [`base`](#base-mode) mode.
If you already have a model with baseline results, you don't need to re-run that optimisation.
Instead, you can _skip the baseline run_.

!!! example

    ```py
    import calliope

    # This model already has results from running in `base` mode.
    model = calliope.read_netcdf(...)

    model.build(mode="spores")
    model.solve(spores={"use_latest_results": True, "number": 10})
    ```

### Continuing from an existing set of SPORES

You may already have a set of SPORES results and want to create more.
You can do so by _continuing from the most recent results_.
This is useful to further explore the option space, restart a run that was stopped prematurely, or to update the SPORE objective from this point onwards.

!!! example

    1. Extending from 5 to 10 SPORES.

    ```python
    import calliope

    # This model already has results from running in `plan` mode.
    model = calliope.Model(...)

    model.build(mode="spores")
    model.solve(spores={"number": 5}) # `model.results` will now have 5 SPORES run results

    # `model.results` will now have an additional 5 SPORES run results (6-10)
    model.solve(spores={"use_latest_results": True, "number": 10})
    ```

    2. Extending from 5 to 10 SPORES and updating the cost slack and SPORES algorithm.

    ```python
    import calliope
    import xarray as xr

    # This model already has results from running in `plan` mode.
    model = calliope.Model(...)

    model.build(mode="spores")
    model.solve(spores={"number": 5}) # `model.results` will now have 5 SPORES run results

    model.backend.update_parameter("spores_slack", xr.DataArray(0.3))

    # `model.results` will now have an additional 5 SPORES run results (6-10)
    model.solve(spores={"use_latest_results": True, "number": 10, "scoring_algorithm": "random"})
    ```

    3. Restarting after premature failure, assuming [results were being saved per SPORE](#saving-results-per-spore).

    ```python
    import calliope
    import xarray as xr

    # Load the model from scratch to get access to input data
    m_init = calliope.Model(...)

    # Load the most recent SPORE run to continue from.
    m_most_recent = calliope.read_netcdf("/path/to/spore/runs/spore_3.nc")

    # Create a new calliope model that merges the input and results data
    m_rerun = calliope.Model(
        inputs=m_init.inputs, results=m_most_recent.results, **m_most_recent.dump_all_attrs()
    )

    # Run use the latest SPORE results as the starting point
    m_rerun.build()
    m_rerun.solve(spores={"use_latest_results": True}, force=True)
    ```
