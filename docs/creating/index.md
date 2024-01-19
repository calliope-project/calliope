# Creating a model

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

## Model configuration and model definition

Calliope models are defined through [YAML][brief-introduction-to-yaml-as-used-in-calliope] files (a format that is both human and computer-readable), and CSV files (a simple tabular format).
In the pages in this section we will take you through each part of setting up your model.

We distinguish between the model **configuration** (the options provided to Calliope to do its work) and the model **definition** (your representation of a physical system in YAML).
Model configuration is everything under the top-level YAML key [`config`][model-configuration-config].
Model definition is everything else, under the top-level YAML keys [`parameters`][top-level-parameters-parameters], [`techs`][technologies-techs], [`nodes`][nodes-nodes], [`tech_groups`][technology-tech_groups-and-node-node_groups-inheritance], [`node_groups`][technology-tech_groups-and-node-node_groups-inheritance], and [`data_sources`][loading-tabular-data-data_sources].

It is possible to define alternatives to the model configuration/definition that you can refer to when you initialise your model.
These are defined under the top-level YAML keys [`scenarios` and `overrides`][scenarios-and-overrides].

We dive into each of these top-level keys in the pages you'll find in the left-hand navigation.

## Structuring your model directory

It makes sense to collect all files belonging to a model inside a single model directory.
The layout of that directory typically looks roughly like this (`+` denotes directories, `-` files):

```
+ example_model
    + model_definition
        - nodes.yaml
        - techs.yaml
    + data_sources
        - solar_resource.csv
        - electricity_demand.csv
    - model.yaml
    - scenarios.yaml
```

In the above example, the files `model.yaml`, `nodes.yaml` and `techs.yaml` together are the model definition.
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


## Brief introduction to YAML as used in Calliope

All model configuration/definition files (with the exception of tabular data files) are in the YAML format, "a human friendly data serialisation standard for all programming languages".

Configuration for Calliope is usually specified as `option: value` entries, where `value` might be a number, a text string, or a list (e.g. a list of further settings).

### Abbreviated nesting

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

### Relative file imports

Calliope also allows a special `import:` directive in any YAML file.
This can specify one or several YAML files to import, e.g.:

```yaml
import:
    - path/to/file_1.yaml
    - path/to/file_2.yaml
```

Data defined in the current and imported file(s) must be mutually exclusive.
If both the imported file and the current file define the same option, Calliope will raise an exception.

As you will see in our [standard model directory structure][structuring-your-model-directory], we tend to store our model definition in separate files.
In this case, our `model.yaml` file tends to have the following `import` statement:

```yaml
import:
    - 'model_definition/techs.yaml'
    - 'model_definition/nodes.yaml'
    - 'scenarios.yaml'
```

This means that we have:

=== "model.yaml"

    ```yaml
    import:
        - 'model_definition/techs.yaml'
        - 'model_definition/nodes.yaml'
        - 'scenarios.yaml'
    config:
        init:
            ...
        build:
            ...
        solve:
            ...
    ```

=== "model_definition/techs.yaml"

    ```yaml
    techs:
        tech1:
            ...
        ...
    ```

=== "model_definition/nodes.yaml"

    ```yaml
    nodes:
        node1:
            ...
        ...
    ```

=== "scenarios.yaml"

    ```yaml
    overrides:
        override1:
        ...
    scenarios:
        scenario1: [override1, ...]
    ...
    ```

Which Calliope will receive as:

    ```yaml
    import:
        - 'model_definition/techs.yaml'
        - 'model_definition/nodes.yaml'
        - 'scenarios.yaml'
    config:
        init:
            ...
        build:
            ...
        solve:
            ...
    techs:
        tech1:
            ...
        ...
    nodes:
        node1:
            ...
        ...
    overrides:
        override1:
        ...
    scenarios:
        scenario1: [override1, ...]
    ...
    ```

!!! note
    * The imported files may include further files, so arbitrary degrees of nested configurations are possible.
    * The `import` statement can either give an absolute path or a path relative to the importing file.

### Overriding one file with another

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

### Data types

Using quotation marks (`'` or `"`) to enclose strings is optional, but can help with readability.
The three ways of setting `option` to `text` below are equivalent:

```yaml
option: "text"
option: 'text'
option: text
```

Without quotations, the following values in YAML will be converted to different Python types:

- Any unquoted number will be interpreted as numeric.
- `true` or `false` values will be interpreted as boolean.
- `.inf` and `.nan` values will be interpreted as the float values `np.inf` and `np.nan`, respectively.
- `null` values will interpreted as `None`.

### Comments

Comments can be inserted anywhere in YAML files with the `#` symbol.
The remainder of a line after `#` is interpreted as a comment.
Therefore, if you have a string with a `#` in it, make sure to use explicit quotation marks.


### Lists and dictionaries

Lists in YAML can be of the form `[...]` or a series of lines starting with `-`.
These two lists are equivalent:

```yaml
key: [option1, option2]
```

```yaml
key:
    - option1
    - option2
```

Dictionaries can be of the form `{...}` or a series of lines _without_ a starting `-`.
These two dictionaries are equivalent:

```yaml
key: {option1: value1, option2: value2}
```

```yaml
key:
    option1: value1
    option2: value2
```

To continue dictionary nesting, you can add more `{}` parentheses or you can indent your lines further.
We prefer to use 2 spaces for indenting (although you'll find that most examples in this documentation are 4 spaces).

We sometimes also ask for lists of dictionaries in Calliope, e.g.:

```
key:
  - option1: value1
    option2: value2
  - option3: value3
    option4: value4
```

Which is equivalent in Python to `{"key": [{"option1": value1, "option2": value2}, {"option3": value3, "option4": value4}]}`.

!!! info "See also"
    See the [YAML website](https://yaml.org/) for more general information about YAML.
