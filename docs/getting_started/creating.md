# Creating a model

## What a basic model needs

You need these basic building blocks in place for the most minimal possible model:

* Basic model configuration
* One carrier
* One `supply` and one `demand` tech for that carrier
* One node
* One timestep

Without specifying anything further, this model will run in the `base` math mode, which means a capacity expansion problem where both the technology capacities (time-independent) and their operation (time-dependent) are decision variables.
In the above example, we would find that the `power_supply` technology, which has a maximum capacity of 100, would get a capacity of 50 in the solution, since that is the demand from `power_demand` in the single timestep.

A minimal model that satisfies the above requirements could be written in a single `model.yaml` file like this:

```yaml
--8<-- "docs/getting_started/minimal_model.yaml"
```

!!! note
    Above, you see another feature of YAML: comment strings.
    Anything after a `#` character is ignored when Calliope reads the model files, so we can use this to leave detailed comments and documentation about the model.
    For example, we can use these comments to specify units: Calliope does not enforce units, but it is good to keep track of them ourselves.

Anything more - additional nodes, more timesteps, more techs (including storage, conversion, transmission, etc), and more carriers - is an extension of this most minimal case.
Since we want to build much larger and more complex models, so let us look at the various building blocks to do so in more detail.

As you will see in the documentation and the examples, there are often different ways of specifying a Calliope model and its data, with their advantages and disadvantages.
For example, in the minimal example above, we are defining the data for `power_demand` directly within the main `model.yaml` file.
This works for a single timestep, but for a real-world model with thousands of timesteps, we would want to read the same data from a tabular file.

## Model directory layout

It makes sense to collect all files belonging to a model inside a single model directory, and to separate different parts of the model into separate files (making use of the `import` top-level key to combine them).
The layout of a Calliope model directory might look like this (`+` denotes directories, `-` files):

```
+ example_model
    + model_definition
        - nodes.yaml
        - techs.yaml
    + data_tables
        - solar_resource.csv
        - electricity_demand.csv
    - model.yaml
    - scenarios.yaml
```

We use the above layout in the example models.
In the above example, the files `model.yaml`, `nodes.yaml` and `techs.yaml` together are the model definition.
Part of `model.yaml` is also taken up by the model configuration.
We will first look at that in more detail, before moving on to the model definition.

## Model configuration

The [model configuration](../basic/config.md) specifies the information Calliope needs to initialise, build, and solve the model.
This includes for example the choice of solver with which to actually solve the mathematical optimisation problem.
Configuration split into `init`, `build`, and `solve`, which mirrors the three main stages through which Calliope creates and runs a model.
A more complete example would look like this:

```yaml
config:
  init:  # Configuration for initialising the model
    name: 'My energy model'
    subset.timesteps: ['2005-01-01', '2005-01-05']  # Run for only a subset of all defined timesteps
    mode: operate  # Run in "operate" mode
  build:  # Configuration for building the model
    backend: pyomo  # Use the (default) Pyomo backend to build the model
  solve:  # Configuration for solving the model
    solver: cbc
```

More details on what is available in the model configuration is in the [model configuration documentation](../basic/config.md).

## Techs

The model's [techs](../basic/techs.md) (technologies) are defined under the `techs` top-level key.
Each technology must specify its `base_tech`, which defines its basic characteristics (i.e., its decision variables and constraints):

* `supply`: Draws from a source to produce a carrier.
* `demand`: Consumes a carrier to supply to an external sink.
* `storage`: Stores a carrier.
* `transmission`: Transmits a carrier from one node to another.
* `conversion`: Converts from one carrier to another.

As explained above, a model must contain at least one `supply` and `demand` tech, whereas the other techs are optional.

Technologies generally need to specify, besides their `base_tech`, at least one `carrier`, and at least one or two constraints like `flow_cap_max` (which constraints the maximum nameplate capacity to produce a flow of the given carrier). A more fully-featured tech definition would look like this:

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

More details on how techs are defined is in the [techs documentation](../basic/techs.md).

## Nodes

The model's [nodes](../basic/nodes.md) are the locations in space where technologies can be placed and are defined under the `nodes` top-level key.
A model can specify any number of nodes.
These nodes can be linked together by transmission technologies.
By consuming a carrier in one node and outputting it in another, linked node, transmission technologies allow resources to be drawn from the system at a different node from where they are brought into it.

The `nodes` section specifies each node. A more complete example looks like this:

```yaml
nodes:
  region1:
    latitude: 40
    longitude: -2
    available_area: 100
    techs:
      demand_power:
      ccgt:
        flow_cap_max: 30000
```

Note that:

* Some of the node-level model specification pertains to the node itself, e.g. the `latitude`, `longitude`, and `available_area`. The former two are used in visualisation, the latter can be used in area-related constraints. These are all optional
* `node_flow_out_max`
* The `techs` key at the node level specifies which techs are installable (available) at that node. As seen in the example above, each allowed tech must be listed (e.g. the case of `demand_power`), and can optionally specify node-specific parameters (e.g. the case of `ccgt`).

If given, node-specific parameters supersede any parameters given at the technology level. In the above example, `flow_cap_max` for the tech `ccgt` in the node `region` will supersede any model-wide value for `ccgt`'s `flow_cap_max` defined in the `techs` top-level key.

More details on how techs are defined is in the [nodes documentation](../basic/nodes.md).

## Transmission techs

A special kind of technology links nodes together so that a carrier can flow from one node to another. These are transmission techs, specified by defining a tech with `base_tech: transmission`.

Each potential "transmission link" between two nodes is specified as a separate technology with `link_from` and `link_to` keys specifying the two nodes it connects:

```yaml
techs:
  ac_transmission:
    base_tech: transmission
    link_from: region1
    link_to: region2
    flow_cap_max: 100
    ...
```

## Data tables

We have chosen YAML syntax to define Calliope models as it is human-readable.
However, when you have a large dataset, the YAML files can quickly become unreadable.
For instance, for parameters that vary in time we would have a list of 8760 values and timestamps to put in our YAML file!

Therefore, alongside your YAML model definition, you can load tabular data from CSV files (or from in-memory [pandas.DataFrame][] objects) using the `data_tables` top-level key.

Recall our simple directory layout above. We have a `data_tables` directory with two tabular CSV files:

```
+ example_model
   ...
    + data_tables
        - solar_resource.csv
        - electricity_demand.csv
    ...
```

To read such a file into our model, we specify the file we want to load and how exactly we want to treat the rows and columns, for example:

```yaml
data_tables:
  pv_capacity_factor_data:
    data: data_tables/solar_resource.csv
    rows: timesteps
    add_dims:
      techs: pv
      parameters: source_use_equals
```

More detail on how to use this powerful feature and how to structure your CSV files is in the [data tables documentation](../basic/data_tables.md).

The [examples and tutorials section](../examples/overview.md) is also particularly useful to see in small example models how this feature works.

## Data definitions

Sometimes we want to define data neither via the node-specific or tech-specific ways outlined above, nor read data in as data tables from separate files.
We can also specify model data via [data definitions](../basic/data_definitions.md), with the `data_definitions` top-level key.
This is particularly useful when making use of more advanced functionality such as [user-defined custom math](../user_defined_math/index.md), where we may want to introduce custom parameters.

More details on how techs are defined is in the [data definitions documentation](../basic/data_definitions.md).

!!! note "Three kinds of parameters: tech-specific, node-specific and indexed parameters"
    Above, we describe how you can directly define data for tech-specific parameters within `techs` and node-specific parameters within `nodes`.
    With data definitions (and of course also with data tables), you can define parameters with any dimensions (including techs/nodes).
    The data definition syntax used for this can also be used in the `nodes` and `techs` keys, for example to define data with additional dimensions.


## Overrides and scenarios

For example, you might want to explore several pre-defined capacity expansion plans in a model of the European power grid.
To do so, you first define a base model, then define one `override` with each alternative grid configuration.

The `scenarios` can combine several `overrides`.
For example, you might also want to explore different future cost developments, and define `overrides` for those.
In your scenarios, you can then combine overrides for a specific realisation of future costs and a specific grid configuration.

Overrides (and the scenarios that reference overrides) can overwrite anything that is defined in the Calliope YAML files: both model configuration and model definition.

For more details on how these are used in practice, refer to the example models, and to the [overrides and scenarios documentation](../basic/scenarios.md).


## Creating a new model from a built-in template

To get a basic model from which to expand, you can use the `calliope new` command, which makes a copy of one of the built-in examples models:

```shell
$ calliope new my_new_model
```

This creates a new directory, `my_new_model`, in the current working directory.

By default, `calliope new` uses the national-scale example model as a template.
To use a different template, you can specify the example model to use, e.g.: `--template=urban_scale`.
