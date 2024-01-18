# Building a model

In short, a Calliope model works like this: **supply technologies** can take a **source** from outside of the modeled system and turn it into a specific **carrier** in the system.
The model specifies one or more **nodes** along with the technologies allowed at those nodes.
**Transmission technologies** can move the same carrier from one node to another, while **conversion technologies** can convert one carrier into another at the same node.
**Demand technologies** remove carriers from the system through a **sink**, while **storage technologies** can store carriers at a specific node.
Putting all of these possibilities together allows a modeller to specify as simple or as complex a model as necessary to answer a given research question.

??? info "Terminology"
    The terminology defined here is used throughout the documentation and the model code and configuration files:

    * **Technology**: a technology that produces, consumes, converts or transports carriers.
    * **Node**: a site which can contain multiple technologies and which may contain other nodes for carrier balancing purposes.
    * **Source**: a source of commodity that can (or must) be used by a technology to introduce carriers into the system.
    * **Sink**: a commodity sink that can (or must) be used by a technology to remove carriers from the system.
    * **Carrier**: a carrier that groups technologies together into the same network, for example `electricity` or `heat`.

    As more generally in constrained optimisation, the following terms are also used:

    * Parameter: a fixed coefficient that enters into model equations.
    * Variable: a variable coefficient (decision variable) that enters into model equations.
    * Constraint: an equality or inequality expression that constrains one or several variables.

!!! example
    Refer to the [examples and tutorials section][examples-and-tutorials] for a more practical look at how to build a Calliope model.

## Files that define a model

Calliope models are defined through YAML files (a format that is both human and computer-readable), and CSV files (a simple tabular format).

??? info "Brief primer on YAML as used in Calliope"

    All configuration files (with the exception of time series data files) are in the YAML format, "a human friendly data serialisation standard for all programming languages".

    Configuration for Calliope is usually specified as `option: value` entries, where `value` might be a number, a text string, or a list (e.g. a list of further settings).

    Calliope allows an abbreviated form for long, nested settings:

    ```yaml
    one:
        two:
            three: x
    ```

    can be written as:

    ```yaml
    one.two.three: x
    ```

    Calliope also allows a special `import:` directive in any YAML file.
    This can specify one or several YAML files to import.
    Data defined in the current and imported file(s) must be mutually exclusive.
    If both the imported file and the current file define the same option, Calliope will raise an exception.

    Certain YAML files _do_ allow overriding, that is the `overrides` you define and then can reference when loading your Calliope model.
    These setting will override any data that match the same name and will add new data if it wasn't already there.
    It will do so by following the entire nesting chain, therefore overriding:

    ```yaml
    one.two.three: x
    four.five.six: x
    ```

    With

    ```yaml
    one.two.four: y
    four.five.six: y

    ```

    Will lead to:

    ```yaml
    one.two:
        three: x
        four: y
    four.five.six: y
    ```

    To _entirely_ replace a nested dictionary you can use our special key `_REPLACE_`.
    Now, using this override:

    ```yaml
    one._REPLACE_:
        four: y
    four.five.six: y
    ```

    Will lead to:

    ```yaml
    one.four: y
    four.five.six: y
    ```

    Using quotation marks (`'` or `"`) to enclose strings is optional, but can help with readability.
    The three ways of setting `option` to `text` below are equivalent:

    ```yaml
    option: "text"
    option: 'text'
    option: text
    ```

    Sometimes, a setting can be either enabled or disabled.
    In this case, the boolean values `true` or `false` are used.
    You can also disable a non-boolean setting by using `null`, which will be converted to `None` in Python.

    Comments can be inserted anywhere in YAML files with the `#` symbol.
    The remainder of a line after `#` is interpreted as a comment.

    See the [YAML website](https://yaml.org/) for more general information about YAML.

It makes sense to collect all files belonging to a model inside a single model directory.
The layout of that directory typically looks roughly like this (`+` denotes directories, `-` files):

```
+ example_model
    + model_config
        - locations.yaml
        - techs.yaml
    + data_sources
        - solar_resource.csv
        - electricity_demand.csv
    - model.yaml
    - scenarios.yaml
```

In the above example, the files `model.yaml`, `locations.yaml` and `techs.yaml` together are the model definition.
This definition could be in one file, but it is more readable when split into multiple.
We use the above layout in the example models.

Inside the `data_sources` directory, tabular data are stored as CSV files.

!!! note
    The easiest way to create a new model is to use the `calliope new` command, which makes a copy of one of the built-in examples models:

    ```shell
    $ calliope new my_new_model
    ```

    This creates a new directory, `my_new_model`, in the current working directory.

    By default, `calliope new` uses the national-scale example model as a template.
    To use a different template, you can specify the example model to use, e.g.: `--template=urban_scale`.

## Model configuration and model definition

We distinguish between the model **configuration** (the options provided to Calliope to do its work) and the model **definition** (your representation of a physical system in YAML).
Model configuration is everything under the top-level YAML key [`config`][model-configuration-config].
Model definition is everything else, under the top-level YAML keys [`parameters`][top-level-parameters-parameters], [`techs`][technologies-techs], [`nodes`][nodes-nodes], [`tech_groups`][technology-tech-groups-and-node-node-groups-inheritance], [`node_groups`][technology-tech-groups-and-node-node-groups-inheritance], and [`data_sources`][loading-tabular-data-data-sources].

It is possible to define alternatives to the model configuration/definition that you can refer to when you initialise your model.
These are defined under the top-level YAML keys [`scenarios` and `overrides`][scenarios-and-overrides].

We will now look at all these top-level keys in turn.

## Model configuration (`config`)

The model configuration specifies the information Calliope needs to initialise, build, and solve the model.
This includes for example the choice of solver with which to actually solve the mathematical optimisation problem. A simple example looks like this:

```yaml
config:
    init:
        name: 'My energy model'
        time_data_path: 'timeseries_data'
        time_subset: ['2005-01-01', '2005-01-05']
    build:
        mode: plan
    solve:
        solver: cbc
```

The configuration is grouped into three top-level items:

* The `init` configuration items are used when you initialise your model (`calliope.Model(...)`).
* The `build` configuration items are used when you build your optimisation problem (`calliope.Model.build(...)`).
* The `solve` configuration items are used when you solve your optimisation problem (`calliope.Model.solve(...)`).

At each of these stages you can override what you have put in your YAML file (or if not in your YAML file, [the default that Calliope uses][config-schema]).
You do this by providing additional keyword arguments on calling `calliope.Model` or its methods. E.g.,:

```python
# Overriding `config.init` items in `calliope.Model`
model = calliope.Model("path/to/model.yaml", time_subset=["2005-01", "2005-02"])
# Overriding `config.build` items in `calliope.Model.build`
model.build(ensure_feasibility=True)
# Overriding `config.solve` items in `calliope.Model.solve`
model.solve(save_logs="path/to/logs/dir")
```

None of the configuration options are _required_ as there is a default value for them all, but you will likely want to set `init.name`, `init.calliope_version`, `init.time_data_path`, `build.mode`, and `solve.solver`.

To test your model pipeline, `config.init.time_subset` is a good way to limit your model size by slicing the time dimension to a smaller range.

### Deep-dive into some key configuration options

#### `config.build.ensure_feasibility`

For a model to find a feasible solution, supply must always be able to meet demand.
To avoid the solver failing to find a solution because your constraints do not enable all demand to be met, you can ensure feasibility:

```yaml
config.build.ensure_feasibility: true
```

This will create an `unmet_demand` decision variable in the optimisation, which can pick up any mismatch between supply and demand, across all carriers.
It has a very high cost associated with its use, so it will only appear when absolutely necessary.

!!! note
    When ensuring feasibility, you can also set a [big M value](https://en.wikipedia.org/wiki/Big_M_method) (`parameters.bigM`). This is the "cost" of unmet demand.
    It is possible to make model convergence very slow if bigM is set too high.
    Default bigM is 1x10$^9$, but should be close to the maximum total system cost that you can imagine.
    This is perhaps closer to 1x10$^6$ for urban scale models and can be as low as 1x10$^4$ if you have re-scaled your data in advance.

#### `config.build.mode`

In the `build` section we have `mode`.
A model can run in `plan`, `operate`, or `spores` mode.
In `plan` mode, capacities are determined by the model, whereas in `operate` mode, capacities are fixed and the system is operated with a receding horizon control algorithm.
In `spores` mode, the model is first run in `plan` mode, then run `N` number of times to find alternative system configurations with similar monetary cost, but maximally different choice of technology capacity and location (node).

In most cases, you will want to use the `plan` mode.
In fact, you can use a set of results from using `plan` model to initialise both the `operate` (`config.build.operate_use_cap_results`) and `spores` modes.

#### `config.solve.solver`

Possible options for solver include `glpk`, `gurobi`, `cplex`, and `cbc`.
The interface to these solvers is done through the Pyomo library. Any [solver compatible with Pyomo](https://pyomo.readthedocs.io/en/6.5.0/solving_pyomo_models.html#supported-solvers) should work with Calliope.

For solvers with which Pyomo provides more than one way to interface, the additional `solver_io` option can be used.
In the case of Gurobi, for example, it is usually fastest to use the direct Python interface:

```yaml
config:
    solve:
        solver: gurobi
        solver_io: python
```

!!! note
    The opposite is currently true for CPLEX, which runs faster with the default `solver_io`.

We tend to test using `cbc` but it is not available to install into your Calliope mamba environment on Windows.
Therefore, we recommend you install GLPK when you are first starting out with Calliope (`mamba install glpk`).

## Model definition

### Top-level parameters (`parameters`)

Some data is not indexed over technologies / nodes (as will be described in more detail below).
This data can be defined under the top-level key `parameters`.
This could be a single value:

```yaml
parameters:
    my_param: 10
```

or (equivalent):

```yaml
    parameters:
        my_param:
            data: 10
```

which can then be accessed in the model inputs `model.inputs.my_param` and used in [custom math][custom-math-formulation] as `my_param`.

Or, it can be indexed over one or more model dimension(s):

```yaml
parameters:
    my_indexed_param:
        data: 100
        index: monetary
        dims: costs
    my_multiindexed_param:
        data: [2, 10]
        index: [[monetary, electricity], [monetary, heat]]  # (1)!
        dims: [costs, carriers]
```

1. The length of the inner index lists is equal to the length of `dims`.
The length of the outer list is equal to the length of `data`.

which can be accessed in the model inputs and [custom math][custom-math-formulation], e.g., `model.inputs.my_multiindexed_param.sel(costs="monetary")` and `my_multiindexed_param`.

You can also index over a new dimension:

```yaml
parameters:
    my_indexed_param:
        data: 100
        index: my_index_val
        dims: my_new_dim
```

Which will add the new dimension `my_new_dim` to your model: `model.inputs.my_new_dim` which you could choose to build a math component over:
`foreach: [my_new_dim]`.

!!! warning
    The `parameter` section should not be used for large datasets (e.g., indexing over the time dimension) as it will have a high memory overhead on loading the data.

### Technologies (`techs`)

Calliope allows a modeller to define technologies with arbitrary characteristics by defining one of the abstract base techs as `base_tech`.
This establishes the basic characteristics in the optimisation model (decision variables and constraints) applied to the technology:

* `supply`: Draws from a source to produce a carrier.
* `demand`: Consumes a carrier to supply to an external sink.
* `storage`: Stores a carrier.
* `transmission`: Transmits a carrier from one node to another.
* `conversion`: Converts from one carrier to another.

!!! info "Sharing configuration through inheritance"
    To share definitions between technologies and/or nodes, you can use configuration inheritance (the `inherit` key).
    This allows a technology/node to inherit definitions from `tech_group`/`node_group` definitions.
    Note that `inherit` is different to setting a `base_tech`.
    Setting a base_tech does not entail any configuration options being inherited.
    It is only used when building the optimisation problem (i.e., in the `math`).

The `techs` section in the model configuration specifies all of the model's technologies.
In our current example, this is in a separate file, `model_config/techs.yaml`, which is imported into the main `model.yaml` file alongside the file for nodes described further below.

```yaml
import:
    - 'model_config/techs.yaml'
    - 'model_config/nodes.yaml'
```

!!! note
    The `import` statement can specify a list of paths to additional files to import (the imported files, in turn, may include further files, so arbitrary degrees of nested configurations are possible).
    The `import` statement can either give an absolute path or a path relative to the importing file.

The following example shows the definition of a `ccgt` technology, i.e. a combined cycle gas turbine that delivers electricity:

```yaml
ccgt:
    name: 'Combined cycle gas turbine'
    color: '#FDC97D'  # (1)!
    base_tech: supply
    carrier_out: power
    source_use_max: .inf # (2)!
    flow_out_eff: 0.5
    flow_cap_max: 40000  # kW
    lifetime: 25
    cost_interest_rate:
        data: 0.10  # (3)!
        index: monetary
        dims: costs
    cost_flow_cap:
        data: 750  # USD per kW
        index: monetary
        dims: costs
    cost_flow_in:
        data: 0.02  # USD per kWh
        index: monetary
        dims: costs
```

1. This is an example of when using quotation marks is important.
Without them, the colour code would be interpreted as a YAML comment!
2. the period at the start of `.inf` will ensure it is read in as a `float` type.
3. Costs require us to explicitly define data in the [top-level parameter][top-level-parameters-parameters] format so that we can define the cost class (in this case: `monetary`).

Each technology must specify an abstract base technology and its carrier (`carrier_out` in the case of a `supply` technology).
Specifying a `color` and a `name` is optional but useful when you want to [visualise or otherwise report your results][analysing-a-model].

The rest of the data for the technology is used in the optimisation problem: to set constraints and to link the technology to the objective function (via costs).
In the above example, we have a capacity limit `flow_cap_max`, conversion efficiency `flow_out_eff`, the life time (used in levelised cost calculations), and the resource available for consumption `source_use_max`.
In the above example, the source is set to infinite via `inf`.

The parameters starting with `costs_` give costs for the technology.
Calliope uses the concept of "cost classes" to allow accounting for more than just monetary costs.
The above example specifies only the `monetary` cost class, but any number of other classes could be used, for example `co2` to account for emissions.
Additional cost classes can be created simply by adding them to the definition of costs for a technology.

??? info "Costs in the objective function"
    By default, all defined cost classes are used in the objective function, i.e., the default objective is to minimize total costs.
    Limiting the considered costs can be achieved by [customising the in-built objective function][introducing-custom-math-to-your-model] to only focus on e.g. monetary costs (`[monetary] in costs`), or updating the `objective_cost_weights` top-level parameter to have a weight of `0` for those cost classes you want to be ignored, e.g.:

    ```yaml
    parameters:
        objective_cost_weights:
            data: [1, 0]
            index: [monetary, co2_emissions]
            dims: costs
    ```

#### Transmission technologies

You will see in the next section that you can associated technologies with nodes, making it possible for investment in those technologies at those nodes.
Since they span two nodes, you associate transmission technologies with nodes in `techs`:

```yaml
techs:
    ac_transmission:
        from: region1  # (1)!
        to: region2
        flow_cap_max: 100
        ...
```

1. The region you specify in `from` or `to` is interchangeable unless you set the parameter `one_way: true`.
In that case, flow along the transmission line is only allowed from the `from` region to the `to` region.

### Nodes (`nodes`)

A model can specify any number of nodes.
These nodes are linked together by transmission technologies.
By consuming a carrier in one node and outputting it in another, linked node, transmission technologies allow resources to be drawn from the system at a different node from where they are brought into it.

The `nodes` section specifies each node:

```yaml
nodes:
    region1:
        latitude: 40
        longitude: -2
        available_area: 100
        node_flow_out_max:
            data: [1000, 2000]
            index: [electricity, gas]
            dims: carriers
        techs:
            unmet_demand_power:
            demand_power:
            ccgt:
                constraints:
                    flow_cap_max: 30000
```

For technologies to be installable at a node, they must be listed under `techs`.
As seen in the example above, each allowed tech must be listed, and can optionally specify additional node-specific parameters (constraints or costs).
If given, node-specific parameters supersede any group constraints a technology defines in the `techs` section for that node.

Nodes can optionally specify coordinates (`latitude` and `longitude`) which are used in visualisation or to compute distances along transmission links.
Nodes can also have any arbitrary parameter assigned which will be available in the optimisation problem, indexed over the `nodes` dimension.
They can also have parameters that use the [top-level parameter syntax][] to define node+other dimension data.
In the above example, `node_flow_out_max` at `region1` could be used to create a [custom math][custom-math-formulation] constraint that limits the total outflow of the carriers electricity and gas at that node.

### Technology (`tech_groups`) and node (`node_groups`) inheritance

For larger models, duplicate entries can start to crop up and become cumbersome.
To streamline data entry, technologies and nodes can inherit common data from a `tech_group` or `node_group`, respectively.

For example, if we want to set interest rate to `0.1` across all our technologies, we could define:

```yaml
tech_groups:
    interest_rate_setter:
        cost_interest_rate:
            data: 0.1
            index: monetary
            dims: costs
techs:
    ccgt:
        inherit: interest_rate_setter
        ...
    ac_transmission:
        inherit: interest_rate_setter
        ...
```

Similarly, if we want to allow the same technologies at all our nodes:

```yaml
node_groups:
    standard_tech_list:
        techs: {ccgt, battery, demand_power}  # (1)!
nodes:
    region1:
        inherit: standard_tech_list
        ...
    region2:
        inherit: standard_tech_list
        ...
    ...
    region100:
        inherit: standard_tech_list
```

1. this YAML syntax is shortform for:
    ```yaml
    techs:
        ccgt:
        battery:
        demand_power:
    ```

Inheritance chains can also be set up.
That is, groups can inherit from groups.
E.g.:

```yaml
tech_groups:
    interest_rate_setter:
        cost_interest_rate:
            data: 0.1
            index: monetary
            dims: costs
    investment_cost_setter:
        inherit: interest_rate_setter
        cost_flow_cap:
            data: 100
            index: monetary
            dims: costs
        cost_area_use:
            data: 1
            index: monetary
            dims: costs
techs:
    ccgt:
        inherit: investment_cost_setter
        ...
    ac_transmission:
        inherit: interest_rate_setter
        ...
```

Finally, inherited properties can always be overridden by the inheriting component.
This can be useful to streamline setting costs, e.g.:

```yaml
tech_groups:
    interest_rate_setter:
        cost_interest_rate:
            data: 0.1
            index: monetary
            dims: costs
    investment_cost_setter:
        inherit: interest_rate_setter
        cost_interest_rate.data: 0.2  # this will replace `0.1` in the `interest_rate_setter`.
        cost_flow_cap:
            data: null
            index: monetary
            dims: costs
        cost_area_use:
            data: null
            index: monetary
            dims: costs
techs:
    ccgt:
        inherit: investment_cost_setter
        cost_flow_cap.data: 100  # this will replace `null` in the `investment_cost_setter`.
        ...
```


### Scenarios and overrides

You may want to define slightly different data for sensitivity analyses, or to compare the effect of resampling your time dimension to varying degrees.
There's no need to create different models files for each of these, instead you can define overrides in your main model definition.

`overrides` are blocks of YAML that specify configurations that expand or override parts of the base model.
`scenarios` are combinations of any number of such overrides.
Both are specified at the top level of the model configuration, as in this example `model.yaml` file:

```yaml
scenarios:
    high_cost_2005: ["high_cost", "year2005"]
    high_cost_2006: ["high_cost", "year2006"]

overrides:
    high_cost:
        techs.onshore_wind.cost_flow_cap.data: 2000
    year2005:
        model.time_subset: ['2005-01-01', '2005-12-31']
    year2006:
        model.time_subset: ['2006-01-01', '2006-12-31']

config:
    ...
```

Each override is given by a name (e.g. `high_cost`) and any number of model settings - anything in the model configuration can be overridden by an override.
In the above example, one override defines higher costs for an `onshore_wind` technology.
The other two other overrides specify different time subsets, so would run an otherwise identical model over two different periods of timeseries data.

One or several overrides can be applied when [running a model][running-a-model].
Overrides can also be combined into scenarios to make applying them at run-time easier.
Scenarios consist of a name and a list of override names which together form that scenario.

Scenarios and overrides can be used to generate scripts that run a single Calliope model many times, either sequentially, or in parallel on a high-performance cluster
(see the section [Generating scripts to repeatedly run variations of a model][generating-scripts-to-repeatedly-run-variations-of-a-model]).

!!! note
    Overrides can also import other files. This can be useful if many overrides are defined which share large parts of model configuration, such as different levels of interconnection between model zones.
    See our [advanced features section][importing-other-yaml-files-in-overrides] for details.<!--TODO-->


### Loading tabular data (`data_sources`)

We have chosen YAML syntax to define Calliope models as it is human-readable.
However, when you have a large dataset, the YAML files can become large and ultimately not as readable as we would like.
For instance, for parameters that vary in time we would have a list of 8760 values and timestamps to put in our YAML file!

Therefore, alongside your YAML model definition, you can load tabular data from CSV files (or from in-memory [pandas.DataFrame][] objects) under the `data_sources` top-level key.
As of Calliope v0.7.0, this tabular data can be of _any_ kind.
Prior to this, loading from file was limited to timeseries data.

The full syntax from loading tabular data can be found in the associated [schema][data_sources_schema].
In brief it is:

* **source**: path to file or reference name for an in-memory object.
* **rows**: the dimension(s) in your table defined per row.
* **columns**: the dimension(s) in your table defined per column.
* **select**: values within dimensions that you want to select from your tabular data, discarding the rest.
* **drop**: dimensions to drop from your rows/columns, e.g., a "comment" row.
* **add_dimensions**: dimensions to add to the table after loading it in, with the corresponding value(s) to assign to the dimension index.

In this section we will show some examples of loading data and provide the equivalent YAML definition that it would replace.

Loading a simple photovoltaic (PV) tech using a time series of hour-by-hour electricity generation data might look like this:

=== Tabular

    Data in file:

    ```
    ,node1,node2
    2005-01-01 12:00:00,15,20
    2005-01-01 13:00:00,5,11
    ...
    ```

    ```yaml
    data_sources:
        pv_capacity_factor_data:
            source: data_sources/pv_resource.csv
            rows: timesteps
            columns: nodes
            add_dimensions:
                techs: pv
                parameters: source_use_equals
    ```
=== YAML

    ```yaml
    techs:
        pv:
            source_use_equals:
                data: [15, 5, 20, 11, ...]
                index:
                    - ["2005-01-01 12:00:00", "node1"]
                    - ["2005-01-01 13:00:00", "node1"]
                    - ["2005-01-01 12:00:00", "node2"]
                    - ["2005-01-01 13:00:00", "node1"]
                    - ...
                dims: ["timesteps", "nodes"]
    ```

??? note "Getting timestamp formats right"
    By default, Calliope expects time series data in a model to be indexed by ISO 8601 compatible time stamps in the format `YYYY-MM-DD hh:mm:ss`, e.g. `2005-01-01 00:00:00`.
    This can be changed by setting `config.build.time_format` based on [`strftime` directives](https://strftime.org/), which defaults to `"ISO8601"`.

    If you work with your CSV files in Excel, keep a careful eye on the format of the timestamps.
    Excel will automatically update the format to match your operating system default, which is usually _not_ the `ISO8601` format.


Loading technology data from file:

=== Tabular

    Data in file:

    ```
    tech1,base_tech,supply
    tech1,flow_cap_max,100
    tech1,flow_out_eff,0.1
    tech1,area_use_max,500
    tech1,area_use_per_flow_cap,7
    tech2,base_tech,demand
    tech3,base_tech,storage
    tech3,storage_cap_max,200
    tech3,flow_cap_max,100
    ...
    ```

    ```yaml
    data_sources:
        tech_data:
            source: data_sources/tech_data.csv
            rows: [techs, parameters]
    ```
=== YAML

    ```yaml
    techs:
        tech1:
            base_tech: supply
            flow_cap_max: 100
            flow_out_eff: 0.1
            area_use_max: 500
            area_use_per_flow_cap: 7
        tech2:
            base_tech: demand
        tech3:
            base_tech: storage
            storage_cap_max: 200
            flow_cap_max: 100
    ```

Loading technology cost data from file:

=== Tabular

    Data in file:

    ```
    tech1,cost_flow_cap,100
    tech1,cost_area_use,50
    tech1,cost_flow_out,0.2
    tech1,cost_interest_rate,0.1
    tech3,cost_flow_cap,20
    tech3,cost_storage_cap,150
    tech3,cost_interest_rate,0.1
    ...
    ```

    ```yaml
    data_sources:
        tech_data:
            source: data_sources/tech_data.csv
            rows: [techs, parameters]
            add_dimensions:
                costs: monetary
    ```
=== YAML

    ```yaml
    tech_groups:  # (1)!
        cost_setter:
            cost_interest_rate:
                data: 0.1
                index: monetary
                dims: costs
            cost_flow_cap:
                data: null
                index: monetary
                dims: costs
            cost_area_use:
                data: null
                index: monetary
                dims: costs
            cost_flow_out:
                data: null
                index: monetary
                dims: costs
            cost_storage_cap:
                data: null
                index: monetary
                dims: costs

    techs:
        tech1:
            inherit: cost_setter
            cost_flow_cap.data: 100
            cost_area_use.data: 50
            cost_flow_out.data: 0.2
        tech3:
            cost_flow_cap.data: 20
            cost_storage_cap.data: 150
    ```

    1. To limit repetition, we have defined [technology groups][technology-tech-groups-and-node-node-groups-inheritance] for our costs.

#### Loading CSV files vs `pandas` dataframes

To load from CSV, set the filepath in `source` to point to your file.
This filepath can either be relative to your `model.yaml` file (as in the above examples) or an absolute path.

To load from a pandas dataframe, you can specify the `data_source_dfs` dictionary of objects when you initialise your model:

```python
import calliope
import pandas as pd
df1 = pd.DataFrame(...)
df2 = pd.DataFrame(...)

model = calliope.Model("path/to/model.yaml", data_source_dfs={"data_source_1": df1, "data_source_2": df2})
```

And then you point to those dictionary keys in your data source `source`:

```yaml
data_sources:
    ds1:
        source: data_source_1
        ...
    ds2:
        source: data_source_2
        ...
```

!!! note
    As with loading tabular data from CSV, you will need to specify `rows`, `columns`, etc. based on the shape of your dataframe.
    Rows correspond to your dataframe index levels and columns to your dataframe column levels.

    You _cannot_ specify [pandas.Series][] objects.
    Ensure you convert them to dataframes (`to_frame()`) before adding them to your data source dictionary.
