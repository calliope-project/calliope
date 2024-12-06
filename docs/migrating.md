# Migrating from v0.6 to v0.7

In Calliope v0.7 we have made many user-facing changes which will mean you cannot load a model definition used in v0.6.

On this page we will list the main changes to help you understand these changes and migrate your existing models to work in v0.7.

We believe these changes will make your life easier in the long run.
Some might seem like steps back, as you have to write _more_ YAML for the same definition.
However, the resulting definition should be easier to understand when you come back to it in the future, and the changes made to model definition have made the internal code much easier - leading to (hopefully) fewer bugs!

Since v0.7 is in a pre-release phase, if there are changes that you don't agree with or bugs when you try implementing something, please [raise an issue](https://github.com/calliope-project/calliope/issues/new/choose) or [start/join a discussion thread](https://github.com/calliope-project/calliope/discussions) on our GitHub repository.

## Changes

### Flat technology definition

We have removed the distinction between `essentials`, `constraints` and `costs`.
Instead, you define all your technology parameters at the same level.

=== "v0.6"

    ```yaml
    ccgt:
      essentials:
        name: 'Combined cycle gas turbine'
        color: '#E37A72'
        parent: supply
        carrier_out: power
      constraints:
        resource: inf
        energy_eff: 0.5
        energy_cap_max: 40000
        energy_cap_max_systemwide: 100000
        energy_ramping: 0.8
        lifetime: 25
      costs:
        monetary:
          interest_rate: 0.10
          energy_cap: 750
          om_con: 0.02
    ```
=== "v0.7"

    ```yaml
    ccgt:
      name: 'Combined cycle gas turbine'
      color: '#E37A72'
      base_tech: supply
      carrier_out: power
      source_use_max: inf
      flow_out_eff: 0.5
      flow_cap_max: 40000
      flow_cap_max_systemwide: 100000
      flow_ramping: 0.8
      lifetime: 25
      cost_interest_rate:
        data: 0.10
        index: monetary
        dims: costs
      cost_flow_cap:
        data: 750
        index: monetary
        dims: costs
      cost_flow_in:
        data: 0.02
        index: monetary
        dims: costs
    ```

### `file=`/`df=` → `data_tables` section

`file=/df=` parameter values as references to timeseries data is replaced with loading tabular data at the top-level using the `data_tables` key.

Assuming you have these two files:

demand_file.csv:
```shell
,node1,node2
2005-01-01 00:00,100,100  # (1)!
2005-01-01 00:00,200,200
...
```

1. We're using positive numbers here which reflects our change to [positive values for demand data](#negative-→-positive-demand-and-carrier-consumption-values).

supply_file.csv:
```shell
,node1,node2
2005-01-01 00:00,10,20
2005-01-01 00:00,1,2
...
```

=== "v0.6"

    ```yaml
    techs:
      demand_tech:
        constraints:
          resource: file=demand_file.csv
          force_resource: true
      supply_tech:
        constraints:
          resource: file=supply_file.csv
          force_resource: false
    ```

=== "v0.7"

    ```yaml
    data_tables:
      demand_data:
        data: demand_file.csv
        rows: timesteps
        columns: nodes
        add_dims:
          techs: demand_tech
          parameters: sink_equals

      supply_data:
        data: supply_file.csv
        rows: timesteps
        columns: nodes
        add_dims:
          techs: supply_tech
          parameters: source_max
    ```

!!! info "See also"
    [`data_tables` introduction](creating/data_tables.md); [`data_tables` tutorial][loading-tabular-data].

### Negative → positive demand and carrier consumption values

Demand data are now strictly _positive_ numbers and so are the values of the `carrier_con` (now `flow_in`) decision variable.

### `model.run()` → `model.build()` + `model.solve()`

When running in Python, building and solving your optimisation problem have been split into two steps:

1. `model.build()` creates the in-memory Python objects that define optimisation problem components (decision variables, constraints, the objective function, ...).
This creates the [calliope.Model.backend][] object, which you can query and use to [tweak the optimisation problem](advanced/backend_interface.md) before sending it to the solver.

2. `model.solve()` sends the built optimisation problem to the configured solver and, provided an optimal solution is available, creates the [calliope.Model.results][] object which contains the optimal results.

### `model` and `run` → `config.init`/`.build`/`.solve`

The model configuration is now split based on the stages of going from a model definition to solving your Calliope model:

* All the options in `config.init` are applied when you create your model (`calliope.Model(...)`).
* All the options in `config.build` are applied when you build your optimisation problem (`calliope.Model.build(...)`).
* All the options in `config.solve` are applied when you solve your optimisation problem (`calliope.Model.solve(...)`).

This split means you can change configuration options on-the-fly if you are working in Python by adding them as keyword arguments in your calls:

=== "YAML config"

    ```yaml
    config:
      init:
        time_subset: ["2005-01", "2005-02"]
      build:
        mode: plan
      solve:
        solver: glpk
    ```

=== "Python"

    ```python
    import calliope
    model = calliope.Model(time_subset=["2005-01", "2005-02"])
    model.build(mode="plan")
    model.solve(solver="glpk")
    ```

!!! info "See also"
    [Introduction to configuring your model](creating/config.md).

### `locations` → `nodes`

`locations` (abbreviated to `locs` in the Calliope data dimensions) has been renamed to `nodes` (no abbreviation).
This allows us to not require an abbreviation and is a disambiguation from the [pandas.DataFrame.loc][] and [xarray.DataArray.loc][] methods.

### `parent` and `tech_groups` → `base_tech` and `templates`

Technology `parent` inheritance has been renamed to `base_tech`, which is fixed to be one of [`demand`, `supply`, `conversion`, `transmission`, `storage`].

The `tech_groups` functionality has been removed in favour of a new, more flexible, `templates` functionality.

=== "v0.6"

    ```yaml
    tech_groups:
      supply_interest_rate:
        essentials:
          parent: supply
        costs:
          monetary:
            interest_rate: 0.1
      conversion_interest_rate:
        essentials:
          parent: conversion
        costs:
          monetary:
          interest_rate: 0.1
    techs:
      supply_tech:
        essentials:
          parent: supply_interest_rate
      conversion_tech:
        essentials:
          parent: supply_conversion_rate
    ```

=== "v0.7"

    ```yaml
    templates:
      common_interest_rate:
        cost_interest_rate:
          data: 0.1
          index: monetary
          dims: costs
    techs:
      supply_tech:
        base_tech: supply
        template: common_interest_rate
      conversion_tech:
        base_tech: conversion
        template: common_interest_rate
    ```

### `costs.monetary.flow_cap` → `cost_flow_cap`

We have changed the nesting structure for defining technology costs so they are flatter and can leverage our multi-dimensional parameter definition.

=== "v0.6"

    ```yaml
    techs:
      supply_tech:
        costs:
          monetary:
            interest_rate: 0.1
            energy_cap: 10
            om_con: 0.05
          emissions:
            om_con: 0.1
    ```

=== "v0.7"

    ```yaml
    techs:
      supply_tech:
        cost_interest_rate:
          data: 0.1
          index: monetary
          dims: costs
        cost_flow_cap:
          data: 10
          index: monetary
          dims: costs
        cost_flow_in:
          data: [0.05, 0.1]
          index: [monetary, emissions]
          dims: costs
    ```

### `links` → transmission links defined in `techs`

The top-level key `links` no longer exists.
Instead, links are defined as separate transmission technologies in `techs`, including `to`/`from` keys:

=== "v0.6"

    ```yaml
    techs:
      ac_transmission:
        essentials:
          parent: transmission
        constrains:
          energy_cap_max: 10
      dc_transmission:
        essentials:
          parent: transmission
        constrains:
          energy_cap_max: 5
    links:
      X1,X2:
        techs:
          ac_transmission:
          dc_transmission:
    ```

=== "v0.7"

    ```yaml
    techs:
      x1_to_x2_ac_transmission:
        from: X1
        to: X2
        base_tech: transmission
        flow_cap_max: 10
      x1_to_x2_dc_transmission:
        from: X1
        to: X2
        base_tech: transmission
        flow_cap_max: 5
    ```

!!! note
    You can use [`templates`](creating/yaml.md#reusing-definitions-through-templates) to minimise duplications in the new transmission technology definition.

### Renaming parameters/decision variables without core changes in function

You may have a already noticed new parameter names being referenced in examples of other changes.
We have renamed parameters to improve clarity in their function and to make it clear that although Calliope is designed to model energy systems, its flow representation is suitable for modelling any kind of flow (water, waste, etc.).

Here are the main changes to parameter/decision variable names that are not linked to changes in functionality (those are detailed elsewhere on this page):

* `energy`/`carrier` → `flow`, e.g. `energy_cap_max` is now `flow_cap_max` and `energy_cap` is now `flow_cap`.
* `prod`/`con` → `out`/`in`, e.g., `carrier_prod` is now `flow_out`.
* `om_prod`/`con` → `cost_flow_out`/`in`.
* `resource` → `source_use` (for things entering the model) and `sink_use` (for things leaving the model).
* `resource_area` → `area_use`.
* `energy_cap_min_use` → `flow_out_min_relative` (i.e., the value is relative to `flow_cap`).
* `parasitic_eff` →  `flow_out_parasitic_eff`.
* `force_asynchronous_prod_con` → `force_async_flow`.
* `cost_var` → `cost_operation_variable`.
* `exists` → `active`.

!!! info "See also"
    [Our full list of internally defined parameters][model-definition-schema];
    the `Parameters` section of our [pre-defined base math documentation][base-math].

### Renaming / moving configuration options

Along with [changing the YAML hierarchy of model configuration](#model-and-run-→-configinitbuildsolve), we have changed the name of configuration options, mainly to create a flat YAML hierarchy or to group settings alphabetically:

* `model.subset_time` → `config.init.time_subset`
* `model.time: {function: resample, function_options: {'resolution': '6H'}}` → `config.init.time_resample`
* `run.operation.window` → `config.build.operate_window`
* `run.operation.horizon` → `config.build.operate_horizon`
* `run.operation.use_cap_results` → `config.build.operate_use_cap_results`

We have also moved some _data_ out of the configuration and into the [top-level `parameters` section](creating/parameters.md):

* `run.objective_options.cost_class` → `parameters.objective_cost_weights`
* `run.bigM` → `parameters.bigM`

!!! info "See also"
    [Our full list of internally defined configuration options][model-configuration-schema].

### `force_resource` → `source_use_equals` / `sink_use_equals`

Instead of defining the binary trigger `force_resource` to enforce the production/consumption of a resource into/out of the system, you can define the data using the new `_equals` parameters `source_use_equals` / `sink_use_equals`.
`source_use_equals` is used to force the use of a resource in a `supply` technology.
`sink_use_equals` is used to force the amount of resource that a `demand` technology must consume.

If you want these resource uses to be upper or lower bounds, use the equivalent `_max`/`_min` parameters.

You can find an example of this change [above](#filedf-→-data_tables-section).

### `units` + `purchased` → `purchased_units`

We have rolled the integer decision variable `units` and the binary `purchased` into one decision variable `purchased_units`.
To achieve the same functionality for `purchased`, set `purchased_units_max: 1`.

### `cost_investment` → `cost_investment_annualised` + `cost_operation_fixed`

Investment costs are split out into the component caused by annual operation and maintenance (`cost_operation_fixed`) and an annualised equivalent of the initial capital investment (`cost_investment_annualised`).
`cost_investment` still exists in the model results and represents the initial capital investment, i.e., without applying the economic depreciation rate.

### Explicitly triggering MILP and storage decision variables/constraints

In v0.6, we inferred that a mixed-integer linear model was desired based on the user defining certain parameters.
For example, defining `units_max` would trigger the integer `units` decision variable.
Defining the `purchase` cost parameter would trigger the binary `purchased` decision variable.

Now, you need to explicitly set the method using `cap_method`:

=== "v0.6"

    ```yaml
    techs:
      supply_tech:
        constraints:  # triggers `units` integer variable
          units_max: 4
          energy_cap_per_unit: 300
          energy_cap_min_use: 0.2
      conversion_tech:  # triggers `purchased` integer variable
        costs:
          monetary:
            purchase: 2000
    ```

=== "v0.7"

    ```yaml
    techs:
      supply_tech:
        units_max: 4
        flow_cap_per_unit: 300
        flow_in_min_relative: 0.2
        cap_method: integer  # triggers the `purchased_units` integer variable
      conversion_tech:
        cost_purchase:
          data: 2000
          index: monetary
          dims: costs
        cap_method: integer  # triggers the `purchased_units` integer variable
    ```

To include a storage buffer in non-`storage` technologies, you also need to explicitly enable it.
To do so, use `include_storage: true` - simply defining e.g. `storage_cap_max` and expecting storage decision variables to be triggered is not enough!

!!! note
    You do not need to enable storage with `include_storage` in `storage` technologies!

### Structure of input and output data within a Calliope model

The concatenated `loc::tech` and `loc::tech::carrier` sets have been removed.
Model components are now indexed separately over `nodes`, `techs`, and `carriers` (where applicable).
Although primarily an internal change, this affects the xarray dataset structure and hence how users access data in `model.inputs` and `model.results`.

For example:

=== "v0.6"

    ```python
    model.inputs.energy_cap_max.loc[{"loc_techs": "X::pv"}]
    ```

=== "v0.7"

    ```python
    model.inputs.flow_cap_max.loc[{"nodes": "X", "techs": "pv"}]
    ```

!!! note
    This change is functionally equivalent to first calling `model.get_formatted_array("energy_cap_max")` in v0.6, which is no longer necessary in v0.7.

### Defining node coordinates

Only geographic coordinates are now allowed (we have [removed x/y coordinates](#xy-coordinates)) and they can be defined directly as `latitude`/`longitude`.

=== "v0.6"

    ```yaml
    nodes:
      X1:
        coordinates:
          lat: 1
          lon: 2
    ```

=== "v0.7"

    ```yaml
    nodes:
      X1:
        latitude: 1
        longitude: 2
    ```

### Distance units

Distances between nodes along transmission links will be automatically derived according to the nodes' [geographic coordinates](#defining-node-coordinates), if the user does not set a `distance`.
This was also the case in v0.6.
The change in v0.7 is that we default to deriving _kilometres_, not _metres_.
If you prefer to keep your distance units in _metres_, set the configuration option: `#!yaml config.init.distance_unit: m`

### Operate mode inputs

* To set the capacities in operate mode, you no longer need to set the `_max` constraints for your technologies (`area_use_max`, `flow_cap_max`, etc.); you can specify the decision variables as parameters directly.
Therefore, you can define e.g. `flow_cap` as one of your technology parameters.
This is because the additional math applied in operate mode deactivates the decision variables with the same names, paving the way for the parameters to be used in the math formulation instead.

    === "v0.6"

        ```yaml
        techs:
          tech1:
            constraints:
            energy_cap_max: 1  # will be translated internally to `energy_cap` by Calliope
            storage_cap_max: 1  # will be translated internally to `storage_cap` by Calliope
        ```

    === "v0.7"

        ```yaml
        techs:
          tech1:
            flow_cap: 1
            storage_cap: 1
        ```

* Operate horizon and window periods are based on Pandas time frequencies, not integer number of timesteps.
Therefore, `24H` is equivalent to `24` in v0.6 if you are using hourly resolution, but is equivalent to `12` in v0.6 if you are using 2-hourly resolution:

    === "v0.6"

        ```yaml
        model:
          time: {function: resample, function_options: {'resolution': '6H'}}
          operation:
            window: 2
            horizon: 4
        ```

    === "v0.7"

        ```yaml
        config:
          init:
            time_resample: 6H
          build:
            operate_window: 12H
            operate_horizon: 24H
        ```

!!! warning
    Although we know that `operate` mode works on our example models, we have not introduced thorough tests for it yet - proceed with caution!

### Per-technology cyclic storage

The configuration option to set cyclic storage globally (`run.cyclic_storage`) has been moved to a parameter at the technology level.
With this change, you can decide if a specific storage technology (or [technology with a storage buffer](#storage-buffers-in-all-technology-base-classes)) has cyclic storage enforced or not.
As in v0.6, cyclic storage defaults to being _on_ (`cyclic_storage: true`).

=== "v0.6"

    ```yaml
    run:
      cyclic_storage: true
    ```

=== "v0.7"

    ```yaml
    techs:
      storage_tech_with_cyclic_storage:
        base_tech: storage
        cyclic_storage: true
      supply_tech_without_cyclic_storage:
        base_tech: supply
        include_storage: true
        cyclic_storage: false
    ```

## Removals

### `_equals` constraints

parameters such as `energy_cap_equals` have been removed.
You can reimplement them by setting `_max` and `_min` parameters to the same value.
The benefit of this is that you can switch between fixing the parameter value (previously `_equals`) and having a range of values (different `_min`/`_max` values) by [updating parameters in the build optimisation model](advanced/backend_interface.md).
With `_equals` constraints, it would trigger a completely different mathematical formulation, which you could not then tweak - you had to rebuild the optimisation problem entirely.

=== "v0.6"

    ```yaml
    techs:
      tech1:
        constraints:
          energy_cap_equals: 1
          storage_cap_equals: 2
    ```

=== "v0.7"

    ```yaml
    techs:
      tech1:
        flow_cap_min: 1
        flow_cap_max: 1
        storage_cap_min: 2
        storage_cap_max: 2
    ```

!!! note
    The exception to this is `source_use_equals`/`sink_use_equals`.
    These parameters have been _introduced_, to [replace `force_resource`](#force_resource-→-source_use_equals--sink_use_equals).
    They are in the model because these tend to be timeseries parameters, so we want to avoid the memory overhead of repeating the data in `_min` and `_max` parameters.

### `x`/`y` coordinates

We no longer process cartesian node coordinates.
Instead, you should define your coordinates using [`latitude`/`longitude`](#defining-node-coordinates).

### Comma-separated node definitions

Defining duplicate definitions for nodes by chaining their names in the YAML key (`node1,node2,node3: ...`) is no longer possible.
We are trying to minimise the custom elements of our YAML files which allows us to leverage YAML schemas to validate user inputs and to keep our YAML readers more maintainable.

You can now use [`templates`](#templates-for-nodes) to minimise duplicating key-value pairs in your YAML definitions.

### `supply_plus` and `conversion_plus` technology base classes

We have removed the `supply_plus` and `conversion_plus` base technology classes.

Instead, `supply_plus` can be effectively represented by using `supply` as the base tech and setting [`include_storage: true`](#explicitly-triggering-milp-and-storage-decision-variablesconstraints) in the model definition.

`conversion_plus` can be represented by using `conversion` as the base tech and using lists of carriers in `carrier_in` and/or `carrier_out`.
To reimplement arbitrary links between carrier "tiers" (`in_2`, `out_2` etc.), you can [define your own math](user_defined_math/index.md), which is a simultaneously more powerful and more human-readable way of defining complex conversion technologies.

!!! info "See also"
    [Example of additional math to link carrier flows](examples/urban_scale/index.md#interlude-user-defined-math).

### `carrier` key

We now require `carrier_in` and `carrier_out` to be explicitly defined for all base techs (only `carrier_in` for demand and `carrier_out` for supply technologies).
This means you cannot use the alias `carrier` to define the same inflow/outflow carrier.
We do this because it aligns with the internal Calliope data structure (we were always converting `carrier` to `carrier_in`/`_out`) and it makes it clearer to the user that the carrier is the same.
This is especially important now that you can [define different inflow/outflow carriers for any technology base class](#multiple-carriers-and-different-carriers-inout-in-all-technology-base-classes).

=== "v0.6"

    ```yaml
    techs:
      battery:
        essentials:
          parent: storage
          carrier: electricity
      supply_tech:
        essentials:
          parent: supply
          carrier: electricity
    ```

=== "v0.7"

    ```yaml
    techs:
      battery:
        base_tech: storage
        carrier_in: electricity
        carrier_out: electricity
      supply_tech:
        base_tech: supply
        carrier_out: electricity
    ```

### `carrier_tiers` and `carrier_ratios`

Carrier tiers were only used in `conversion_plus` technologies, yet led to a whole new model dimension.
Additionally, `carrier_ratios` could be easily confused due to the complex nature of their application.
With the [removal of the `conversion_plus` base class](#supply_plus-and-conversion_plus-technology-base-classes), we have simplified how multiple carriers in/out are defined.
To achieve the same functionality as carrier tiers/ratios offered, you will need to apply your own math.

One form of carrier flow interactions _is_ still possible using only the pre-defined math.
This is where there is a choice between inflow/outflow carriers instead of one carrier inflow/outflow _requiring_ the inflow/outflow of another carrier.
You can do this with flow efficiencies indexed over carriers rather than using `carrier_ratios`.

For instance, here's how you represent a reversible heat pump without additional math:

=== "v0.6"

    ```yaml
    techs:
      heat_pump:
        essentials:
          parent: conversion_plus
          carrier_in: electricity
          carrier_out: [heat, cooling]
        constraints:
          carrier_ratios:
            carrier_out:
              heat: 3
              cooling: 2.5
    ```

=== "v0.7"

    ```yaml
    techs:
      heat_pump:
        base_tech: conversion
        carrier_in: electricity
        carrier_out: [heat, cooling]
        flow_out_eff:
          data: [3, 2.5]
          index: [heat, electricity]
          dims: carriers
    ```

!!! info "See also"
    [Example of additional math to link carrier flows](examples/urban_scale/index.md#interlude-user-defined-math);
    [Examples of complex CHP plant operating space math][chp-plants].

### Group constraints

One driving reason to implement our [own math syntax](user_defined_math/index.md) was to replace our "group constraints".
These constraints were becoming more and more complex and it ultimately proved impossible to manage all the different ways users wanted to apply them.
We have re-implemented all these constraints as tested additional math snippets, which you can explore in our [example gallery](user_defined_math/examples/index.md).

### Configuration options

* With the [change in how timeseries data is defined](#filedf-→-data_tables-section), we have removed the reference to a `timeseries_data_path`.
Instead, data table filepaths should always be relative to the `model.yaml` file or they should be absolute paths.
* We have removed `run.relax_constraint` alongside [removing group constraints](#group-constraints).
* We have removed `model.file_allowed`, which many users will not even know existed (it was a largely internal configuration option)!
Instead, it is possible to index any parameter over the time dimension.
It is up to you to ensure the math formulation is set up to handle this change, which may require [tweaking existing math](user_defined_math/customise.md#introducing-additional-math-to-your-model).
* With the [removal of time clustering](#clustering), we have removed `model.random_seed` and `model.time` options.

### Plotting

It is now no longer possible to plot natively with Calliope.
We made this decision due to the wide variety of visualisations that we saw being created outside our plotting module.
It has proven impossible to keep our plotting methods agile given the almost infinite tweaks that libraries like [matplotlib](https://matplotlib.org/) and [plotly](https://plotly.com/) allow.

If you want to achieve some of the same plots that were possible with the Calliope v0.6 plotting module, see our [example notebooks](examples/index.md).

At a later stage, we are planning for a separate visualisation module that will provide similar functionality to the formerly-included plotting.

### Clustering

Time masking and clustering capabilities have been severely reduced.
Time resampling and clustering are now accessible by top-level configuration keys: e.g., `config.init.time_resample: 2H`, `config.init.time_cluster: cluster_file.csv`.
Clustering is simplified to only matching model dates to representative days, with those representative days being in the clustered timeseries.

If you want to masking/cluster data you should now leverage other tools, some of which you can find referenced on our [time adjustment](advanced/time.md#time-clustering) page.
We made this decision due to the complex nature of time clustering.
With our former implementation, we were making decisions about the data that the user should have more awareness of and control over.
It is also a constantly evolving field, but not the focus of Calliope, so we are liable to fall behind on the best-in-class methods.

## Additions

### Storage buffers in all technology base classes

On [removing `supply_plus`](#supply_plus-and-conversion_plus-technology-base-classes), we have opened up the option to have a storage "buffer" for any technology base class.
This enables any flow into the technology to be stored across timesteps as it is in a `storage` technology.
We have not yet enabled this for `demand` technologies, but you could [add your own math](user_defined_math/index.md) to enable it.

!!! warning
    Although our math should be set up to handle a storage buffer for a `conversion` or `transmission` technology, we do not have any direct tests to check possible edge cases.

### Multiple carriers and different carriers in/out in all technology base classes

On [removing `conversion_plus`](#supply_plus-and-conversion_plus-technology-base-classes), we have opened up the option to have different carriers in/out of `storage`/`transmission` technologies, and to define multiple carriers in/out of any technology.
This means you could define different output carriers for a `supply` technology, or a different carrier into a storage technology compared to the carrier that comes out.

!!! warning
    Although our math should be set up to handle multiple carriers and different inflow/outflow carriers for non-conversion technologies, we do not have any direct tests to check possible edge cases.

### `templates` for nodes

The new [`templates` key](creating/yaml.md#reusing-definitions-through-templates) makes up for the [removal of grouping node names in keys by comma separation](#comma-separated-node-definitions).

So, to achieve this result:

```yaml
nodes:
  region1:
    techs:
      battery:
      demand_electricity:
      ccgt:
  region2:
    techs:
      battery:
      demand_electricity:
      ccgt:
  region3:
    techs:
      battery:
      demand_electricity:
      ccgt:
```

We would do:

=== "v0.6"

    ```yaml
    nodes:
      region1,region2,region3:
        techs:
          battery:
          demand_electricity:
          ccgt:
    ```

=== "v0.7"

    ```yaml
    templates:
      standard_tech_list:
        techs:
          battery:
          demand_electricity:
          ccgt:
    nodes:
      region1.template: standard_tech_list
      region2.template: standard_tech_list
      region3.template: standard_tech_list
    ```

### Inflow and outflow efficiencies

Flow efficiencies are now split into inflow (`flow_in_eff`) and outflow (`flow_out_eff`) efficiencies.
This enables different storage charge/discharge efficiencies to be applied.

### Differentiating capacities and efficiencies between carriers

`flow_cap` (formerly `energy_cap`) is indexed in the optimisation problem over `carriers` as well as `nodes` and `techs`.
This allows capacity constraints to be defined separately for inflows and outflows for `conversion` technologies.
For example:

```yaml
techs:
  dual_fuel_coal_plant:
    base_tech: conversion
    carrier_in: [coal, biofuel]
    carrier_out: electricity
    flow_cap_max:
      data: [100, 80]
      index: [coal, biofuel]
      dims: carriers
    flow_in_eff:
      data: [0.5, 0.6]
      index: [coal, biofuel]
      dims: carriers
```

You can "switch off" a constraint for a given carrier by setting its value to `null` in the indexed parameter data or just not referencing it:

=== "Setting a carrier value to `null`"

    ```yaml
    techs:
      dual_fuel_coal_plant:
        base_tech: conversion
        carrier_in: [coal, biofuel]
        carrier_out: electricity
        flow_cap_max:
          data: [100, null]
          index: [coal, biofuel]
          dims: carriers
        flow_in_eff:
          data: [0.5, 0.6]
          index: [coal, biofuel]
          dims: carriers
    ```

=== "Not referencing a carrier"

    ```yaml
    techs:
      dual_fuel_coal_plant:
        base_tech: conversion
        carrier_in: [coal, biofuel]
        carrier_out: electricity
        flow_cap_max:
          data: 100
          index: coal
          dims: carriers
        flow_in_eff:
          data: [0.5, 0.6]
          index: [coal, biofuel]
          dims: carriers
    ```

!!! warning
    If you define multiple inflow/outflow carriers but don't specify the `carriers` dimension in your parameter definitions, the values will be applied to _all_ carriers.
    That is,

    ```yaml
    techs:
      dual_fuel_coal_plant:
        base_tech: conversion
        carrier_in: [coal, biofuel]
        carrier_out: electricity
        flow_cap_max: 100
        flow_in_eff: 0.5
    ```

    Will be interpreted by Calliope as:

    ```yaml
    techs:
      dual_fuel_coal_plant:
        base_tech: conversion
        carrier_in: [coal, biofuel]
        carrier_out: electricity
        flow_cap_max:
          data: [100, 100]
          index: [coal, biofuel]
          dims: carriers
        flow_in_eff:
          data: [0.5, 0.5]
          index: [coal, biofuel]
          dims: carriers
    ```

### Defining parameters outside the scope of `nodes` and `techs`

We now have a top-level key: `parameters` in which you can use our indexed parameter syntax to define any model parameters that you want, without them necessarily being linked to a node/technology.
For instance, to define a parameter that applies over the `carriers` dimension:

```yaml
parameters:
  my_new_param:
    data: [1, 2, 3]
    index: [heat, electricity, waste]
    dims: carriers
```

Or to define a single value that you might use to limit the total emissions in your system:

```yaml
parameters:
  emissions_limit:
    data: 100
    index: emissions
    dims: costs
```

!!! info "See also"
    [Defining parameters when you create your model](creating/parameters.md).

### Indexing parameters over arbitrary dimensions

At the `tech` level, `node` level, and the top-level (via the [`parameters` key](#defining-parameters-outside-the-scope-of-nodes-and-techs)), you can extend the dimensions a parameter is indexed over.

At the tech level, this allows you to define [different values for different carriers](#differentiating-capacities-and-efficiencies-between-carriers).
At any level, it allows you to define a brand new model dimension and your values over those.
For example, if you want to apply some simplified piecewise constraints:

```yaml
parameters:
  cost_flow_cap_piecewise_slopes:
    data: [5, 7, 14]
    index: [0, 1, 2]
    dims: pieces
  cost_flow_cap_piecewise_intercept:
    data: [0, -2, -16]
    index: [0, 1, 2]
    dims: pieces
```

Or if you want to set costs for carrier flows at a node:

```yaml
nodes:
  region1:
    cost_per_flow:
      data: [1, 2]
      index: [[monetary, electricity], [monetary, heat]]
      dims: [costs, carriers]
```

!!! note
    1. Just defining new parameters is not enough to have an effect on the optimisation problem.
    You also need to [define your own math](user_defined_math/index.md).
    2. Because we process your YAML files to create the `nodes` and `techs` dimensions you will find in your Calliope model, you cannot use `nodes`/`techs` as dimensions of indexed parameters under the `nodes` or `techs` keys.
    It _is_ possible to refer to `nodes` and `techs` as dimensions under the top-level `parameters` key.

!!! info "See also"
    [Defining parameters when you create your model](creating/parameters.md).

### Loading non-timeseries tabular data

With the [change in loading timeseries data](#filedf-→-data_tables-section), we have expanded loading of tabular data to allow any data input.
Technically, you can now define all your data in tables (although we would still recommend a mix of YAML and tabular model definition).

!!! info "See also"
    `data_tables` [introduction](creating/data_tables.md) and [tutorial][loading-tabular-data].

### YAML-based math syntax

We have overhauled our internal mathematical formulation to remove the strong link to the Pyomo library.
Now, all components of our internal math are defined in a readable YAML syntax that we have developed.

You can add your own math to update the pre-defined math and to represent the physical system in ways we do not cover in our base math, or to apply new modelling methods and problem types (e.g., pathway or stochastic optimisation)!

When adding your own math, you can add [piecewise linear constraints](user_defined_math/components.md#piecewise-constraints), which is a new type of constraint compared to what could be defined in v0.6.

!!! info "See also"
    Our [pre-defined](pre_defined_math/index.md) and [user-defined](user_defined_math/index.md) math documentation.
