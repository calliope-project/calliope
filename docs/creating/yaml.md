# Brief introduction to YAML as used in Calliope

All model configuration/definition files (with the exception of tabular data files) are in the YAML format, "a human friendly data serialisation standard for all programming languages".

Configuration for Calliope is usually specified as `option: value` entries, where `value` might be a number, a text string, or a list (e.g. a list of further settings).

## Abbreviated nesting

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

## Relative file imports

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

## Overriding one file with another

While generally, as stated above, if an the imported file and the current file define the same option, Calliope will raise an exception.

However, you can define `overrides` which you can then reference when loading your Calliope model (see [Scenarios and overrides](scenarios.md)). These `override` settings will override any data that match the same name and will add new data if it wasn't already there.

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

## Data types

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
- `.inf` and `.nan` values will be interpreted as the float values `np.inf` (infinite) and `np.nan` (not a number), respectively.
- `null` values will interpreted as `None`.

## Comments

Comments can be inserted anywhere in YAML files with the `#` symbol.
The remainder of a line after `#` is interpreted as a comment.
Therefore, if you have a string with a `#` in it, make sure to use explicit quotation marks.


## Lists and dictionaries

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

Which is equivalent in Python to `{"key": [{"option1": value1, "option2": value2}, {"option3": value3, "option4": value4}]}`.

!!! info "See also"
    See the [YAML website](https://yaml.org/) for more general information about YAML.
