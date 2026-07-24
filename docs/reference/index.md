# Reference

Reference material for Calliope: the YAML syntax and command-line interface, the built-in math formulations, the Python API, and the schemas that validate model definitions.

These pages are intended for looking up specifics. For a guided introduction, see [getting started](../getting_started/concepts.md) and the [building blocks](../basic/index.md) section.

<div class="grid cards" markdown>

-   :fontawesome-solid-terminal:{ .lg .middle } __Syntax & CLI__

    ---

    * [YAML as used in Calliope](yaml.md): the YAML conventions used throughout Calliope.
    * [Command line interface](cli.md): reference for the `calliope` command-line tools.

-   :fontawesome-solid-square-root-variable:{ .lg .middle } __Built-in math__

    ---

    * [Built-in base math](../math/base.md): the base mathematical formulation, always applied to a model.
    * [Other built-in math](../math/built_in/index.md): pre-defined mode and extra math.

-   :fontawesome-brands-python:{ .lg .middle } __Python API__

    ---

    * [Model](api/model.md): the core `calliope.Model` class.
    * [Backend](api/backend_model.md): the optimisation backend interface.
    * [Helper functions](api/helper_functions.md): math helper functions.
    * [Example models](api/example_models.md): loading the built-in example models.
    * [AttrDict](api/attrdict.md): the nested-dictionary utility.
    * [Exceptions](api/exceptions.md): Calliope's errors and warnings.
    * [Logging](api/logging.md): configuring Calliope's logging.

-   :fontawesome-solid-table-list:{ .lg .middle } __Schemas__

    ---

    * [Configuration schema](config_schema.md): the `config` validation schema.
    * [Data table schema](data_table_schema.md): the `data_tables` validation schema.
    * [Model definition schema](model_schema.md): the model definition validation schema.
    * [Math schema](math_schema.md): the math validation schema.

</div>
