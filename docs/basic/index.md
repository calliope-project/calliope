# Building blocks

This section explains the building blocks that make up a Calliope model in detail: how to define a model, how to run and analyse it, and how to extend it with your own math.

If you are new to Calliope, work through the [getting started](../getting_started/concepts.md) and [examples & tutorials](../examples/index.md) sections first; the pages here go into the detail behind those introductions.

<div class="grid cards" markdown>

-   :fontawesome-solid-cubes:{ .lg .middle } __Defining your model__

    ---

    * [Model configuration](config.md): the `config` section that controls how a model is built and solved.
    * [Technologies](techs.md): defining the `techs` that produce, consume, store, and convert energy.
    * [Nodes](nodes.md): defining the `nodes` at which technologies are placed.
    * [Loading tabular data](data_tables.md): bringing in data from CSV files or dataframes via `data_tables`.
    * [Data definitions](data_definitions.md): how parameters and indexed data are defined.

-   :fontawesome-solid-play:{ .lg .middle } __Running & analysing__

    ---

    * [Modes](modes.md): the available build and solve modes (`plan`, `operate`, `spores`).
    * [Scenarios and overrides](scenarios.md): managing variations of a model.
    * [Running in the command line](running-cli.md): building and solving from the CLI.
    * [Running in Python](running-python.md): building and solving from a Python session.
    * [Postprocessing](postprocessing.md): working with results after solving.

-   :fontawesome-solid-square-root-variable:{ .lg .middle } __User-defined math__

    ---

    * [Defining your own math](../user_defined_math/index.md): extending Calliope's base math.
    * [Math components](../user_defined_math/components.md): variables, constraints, expressions, and objectives.
    * [Math syntax](../user_defined_math/syntax.md): the syntax for writing math expressions.
    * [Helper functions](../user_defined_math/helper_functions.md): functions available within math.
    * [Adding your own math to a model](../user_defined_math/customise.md): applying custom math to a model.

</div>
