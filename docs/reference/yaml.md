# YAML as used in Calliope

All model configuration/definition files (with the exception of tabular data files) are in the YAML format, "a human friendly data serialisation standard for all programming languages".

## A quick introduction to YAML

Configuration for Calliope is usually specified as `option: value` entries, where `value` might be a number, a text string, or a list (e.g. a list of further settings).

!!! info "See also"
    See the [YAML website](https://yaml.org/) for more general information about YAML.

### Data types

Using quotation marks (`'` or `"`) to enclose strings is optional, but can help with readability.
The three ways of setting `option` to `text` below are equivalent:

```yaml
option1: "text"
option2: 'text'
option3: text
```

Without quotations, the following values in YAML will be converted to different Python types:

- Any unquoted number will be interpreted as numeric (e.g., `1`, `1e6` `1e-10`).
- `true` or `false` values will be interpreted as boolean.
- `.inf` and `.nan` values will be interpreted as the float values `np.inf` (infinite) and `np.nan` (not a number), respectively.
- `null` values will interpreted as `None`.

### Comments

Comments can be inserted anywhere in YAML files with the `#` symbol.
The remainder of a line after `#` is interpreted as a comment.
Therefore, if you have a string with a `#` in it, make sure to use explicit quotation marks.

```yaml
# This is a comment
option1: "text with ##hashtags## needs quotation marks"
```

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
We prefer to use 2 spaces for indenting as this makes the nested data structures more readable than the often-used 4 spaces.

We sometimes also use lists of dictionaries in Calliope, e.g.:

```yaml
key:
  - option1: value1
    option2: value2
  - option3: value3
    option4: value4
```

Which is equivalent in Python to `#!python {"key": [{"option1": value1, "option2": value2}, {"option3": value3, "option4": value4}]}`.

## Calliope's additional YAML features

To make model definition easier, we add some extra features that go beyond regular YAML formatting.

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

As you will see in our [standard model directory structure](../getting_started/creating.md#model-directory-layout), we tend to store our model definition in separate files.
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

### Reusing definitions through templates

For larger models, duplicate entries can start to crop up and become cumbersome.
To streamline data entry, any section can inherit common data from a `template` which is defined in the top-level `templates` section.

???+ example "Example 1: templates in technologies"

    If we want to set interest rate to `0.1` across all our technologies, we could define:

    === "Using templates"

        ```yaml
        templates:
          interest_rate_setter:
            cost_interest_rate:
              data: 0.1
              index: monetary
              dims: costs
        techs:
          ccgt:
            flow_out_eff: 0.5
            template: interest_rate_setter
          ac_transmission:
            flow_out_eff: 0.98
            template: interest_rate_setter
        ```
    === "Without templates"

        ```yaml
        techs:
          ccgt:
            flow_out_eff: 0.5
            cost_interest_rate:
              data: 0.1
              index: monetary
              dims: costs
          ac_transmission:
            flow_out_eff: 0.98
            cost_interest_rate:
              data: 0.1
              index: monetary
              dims: costs
        ```

??? example "Example 2: templates in nodes"

    Similarly, if we want to allow the same technologies at all our nodes:

    === "Using templates"

        ```yaml
        templates:
          standard_tech_list:
            techs: {ccgt, battery, demand_power}
        nodes:
          region1:
            template: standard_tech_list
            latitude: 39
            longitude: -2
          region2:
            template: standard_tech_list
            latitude: 40
            longitude: 0
        ```

    === "Without templates"

        ```yaml
        nodes:
          region1:
            techs:
              ccgt:
              battery:
              demand_power:
            latitude: 39
            longitude: -2
          region2:
            techs:
              ccgt:
              battery:
              demand_power:
            latitude: 40
            longitude: 0
        ```

??? example "Example 3: templates in data tables"

    Storing common options under the `templates` key is also useful for data tables.

    === "Using templates"

        ```yaml
        templates:
          common_data_options:
            rows: timesteps
            columns: nodes
            add_dims:
              inputs: source_use_max
        data_tables:
          pv_data:
            table: /path/to/pv_timeseries.csv
            template: common_data_options
            add_dims:
              techs: pv
          wind_data:
            table: /path/to/wind_timeseries.csv
            template: common_data_options
            add_dims:
              techs: wind
          hydro_data:
            table: /path/to/hydro_timeseries.csv
            template: common_data_options
            add_dims:
              techs: hydro
        ```
    === "Without templates"

        ```yaml
        data_tables:
          pv_data:
            table: /path/to/pv_timeseries.csv
            rows: timesteps
            columns: nodes
            add_dims:
              inputs: source_use_max
              techs: pv
          wind_data:
            table: /path/to/wind_timeseries.csv
            rows: timesteps
            columns: nodes
            add_dims:
              inputs: source_use_max
              techs: wind
          hydro_data:
            table: /path/to/hydro_timeseries.csv
            rows: timesteps
            columns: nodes
            add_dims:
              inputs: source_use_max
              techs: hydro
        ```

Inheritance chains can also be created.
That is, templates can inherit from other templates.

??? example "Example 4: template inheritance chain"

    A two-level template inheritance chain.

    === "Using templates"

        ```yaml
        templates:
          interest_rate_setter:
            cost_interest_rate:
              data: 0.1
              index: monetary
              dims: costs
          investment_cost_setter:
            template: interest_rate_setter
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
            template: investment_cost_setter
            flow_out_eff: 0.5
          ac_transmission:
            template: interest_rate_setter
            flow_out_eff: 0.98
        ```

    === "Without templates"

        ```yaml
        techs:
          ccgt:
            cost_interest_rate:
              data: 0.1
              index: monetary
              dims: costs
            cost_flow_cap:
              data: 100
              index: monetary
              dims: costs
            cost_area_use:
               data: 1
               index: monetary
               dims: costs
            flow_out_eff: 0.5
          ac_transmission:
            cost_interest_rate:
              data: 0.1
              index: monetary
              dims: costs
            cost_flow_cap:
              data: 100
              index: monetary
              dims: costs
            cost_area_use:
               data: 1
               index: monetary
               dims: costs
            flow_out_eff: 0.98
        ```

Template properties can always be overwritten by the inheriting component.
That is, a 'local' value has priority over the template value.
This can be useful to streamline setting costs for different technologies.

??? example "Example 5: overriding template values"

    In this example, a technology overrides a single templated cost.

    === "Using templates"

        ```yaml
        templates:
          interest_rate_setter:
            cost_interest_rate:
              data: 0.1
              index: monetary
              dims: costs
          investment_cost_setter:
            template: interest_rate_setter
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
            template: investment_cost_setter
            cost_flow_cap.data: 100  # this will replace `null` in the `investment_cost_setter`.
        ```

    === "Without templates"

        ```yaml
        techs:
          ccgt:
            cost_interest_rate:
              data: 0.2
              index: monetary
              dims: costs
            cost_flow_cap:
              data: 100
              index: monetary
              dims: costs
            cost_area_use:
              data: null
              index: monetary
              dims: costs
        ```

### Overriding one file with another

Generally, if the imported file and the current file define the same option, Calliope will raise an exception.

However, you can define `overrides` which you can then reference when loading your Calliope model (see [Scenarios and overrides](../basic/scenarios.md)). These `override` settings will override any data that match the same name and will add new data if it wasn't already there.

It will do so by following the entire nesting chain. For example:

```yaml
# Initial configuration
one.two.three: x
four.five.six: x

# Override to apply
one.two.four: y
four.five.six: y
```

The above would lead to:

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
