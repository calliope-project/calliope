# Creating a model

In short, a Calliope model works like this: **supply technologies** can take a **source** from outside of the modeled system and turn it into a specific **carrier** in the system.
The model specifies one or more **nodes** along with the technologies allowed at those nodes.
**Transmission technologies** can move the same carrier from one node to another, while **conversion technologies** can convert one carrier into another at the same node.
**Demand technologies** remove carriers from the system through a **sink**, while **storage technologies** can store carriers at a specific node.
Putting all of these possibilities together allows a modeller to specify as simple or as complex a model as necessary to answer a given research question.

??? info "An overview of the Calliope terminology"
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
    Refer to the [examples and tutorials section](../examples/index.md) for a more practical look at how to build a Calliope model.

## Model configuration and model definition

Calliope models are defined through [YAML](yaml.md) files (a format that is both human and computer-readable), and CSV files (a simple tabular format).
In the pages in this section we will take you through each part of setting up your model.

We distinguish between:

- the model **configuration** (the options provided to Calliope to do its work) and
- the model **definition** (your representation of a physical system in YAML).

Model configuration is everything under the top-level YAML key [`config`](config.md).
Model definition is everything else, under the top-level YAML keys [`parameters`](parameters.md), [`techs`](techs.md), [`nodes`](nodes.md), [`templates`](templates.md), and [`data_tables`](data_tables.md).

It is possible to define alternatives to the model configuration/definition that you can refer to when you initialise your model.
These are defined under the top-level YAML keys [`scenarios` and `overrides`](scenarios.md).

We dive into each of these top-level keys in the pages you'll find in the left-hand navigation.

## Structuring your model directory

It makes sense to collect all files belonging to a model inside a single model directory.
The layout of that directory typically looks roughly like this (`+` denotes directories, `-` files):

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

In the above example, the files `model.yaml`, `nodes.yaml` and `techs.yaml` together are the model definition.
This definition could be in one file, but it is more readable when split into multiple.
We use the above layout in the example models.

Inside the `data_tables` directory, tabular data are stored as CSV files.

!!! note
    The easiest way to create a new model is to use the `calliope new` command, which makes a copy of one of the built-in examples models:

    ```shell
    $ calliope new my_new_model
    ```

    This creates a new directory, `my_new_model`, in the current working directory.

    By default, `calliope new` uses the national-scale example model as a template.
    To use a different template, you can specify the example model to use, e.g.: `--template=urban_scale`.

## Next steps to setting up your model

The rest of this section discusses everything you need to know to set up a model:

- An overview of [YAML as it is used in Calliope](yaml.md) - though this comes first here, you can also safely skip it and refer back to it as a reference as questions arise when you go through the model configuration and definition examples.
- More details on the [model configuration](config.md).
- The key parts of the model definition, first, the [technologies](techs.md), then, the [nodes](nodes.md), the locations in space where technologies can be placed.
- How to use [technology and node templates](templates.md) to reduce repetition in the model definition.
- Other important features to be aware of when defining your model: defining [indexed parameters](parameters.md), i.e. parameter which are not indexed over technologies and nodes, [loading tabular data](data_tables.md), and defining [scenarios and overrides](scenarios.md).
