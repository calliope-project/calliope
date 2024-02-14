# Adding your own math to a model

Once you understand the [math components](components.md) and the [formulation syntax](syntax.md), you'll be ready to introduce your own math to a model.

You can find examples of additional math that we have put together in our [math example gallery](examples/index.md).

Whenever you introduce your own math, it will be applied on top of the [base math][base-math].
Therefore, you can include base math overrides as well as add new math.
For example, you may want to introduce a timeseries parameter to the pre-defined `storage_max` constraint to limit maximum storage capacity on a per-timestep basis:

```yaml
storage_max:
  equations:
    - expression: storage <= storage_cap * time_varying_parameter
```

The other elements of the `storage_max` constraints have not changed (`foreach`, `where`, ...), so we do not need to define them again when adding our own twist on the pre-defined math.

When defining your model, you can reference any number of YAML files containing the math you want to add in `config.init`. The paths are relative to your main model configuration file:

```yaml
config:
  init:
    add_math: [my_new_math_1.yaml, my_new_math_2.yaml]
```

You can also define a mixture of your own math and the [pre-defined math](../pre_defined_math/index.md):

```yaml
config:
  init:
    add_math: [my_new_math_1.yaml, storage_inter_cluster, my_new_math_2.md]
```

## Adding your parameters to the YAML schema

Our YAML schema are used to validate user inputs.
The model definition schema includes metadata on all our pre-defined parameters, which you can find rendered in our [reference page][model-definition-schema].

When you add your own math you are likely to be adding new parameters to the model.
You can update the Calliope model definition schema to include your new entries using `calliope.util.schema.update_model_schema`.
This ensures that your parameters have default values attached to them and if you choose to [write your own documentation](#writing-your-own-math-documentation), your parameters will have this metadata added to their descriptions.

Entries in the schema look like this:

```yaml
flow_cap_max:
  $ref: "#/$defs/TechParamNullNumber"  # (1)!
  default: .inf
  x-type: float
  title: Maximum rated flow capacity.
  description: >-
    Limits `flow_cap` to a maximum.
  x-unit: power.
```

1. This is a cross-reference to a much longer schema entry that says the parameter type is either `None`, a simple number, or an indexed parameter dictionary with the `data`, `index`, and `dims` keys.

When you add your own parameters to the schema, you will need to know the top-level key under which the parameter will be found in your YAML definition: [`nodes`](../creating/nodes.md), [`techs`](../creating/techs.md), or [`parameters`](../creating/parameters.md).
As a general rule, if it includes the `techs` dimension, put it under `techs`; if it includes `nodes` but _not_ `techs` then put it under `nodes`; if it includes neither dimension, put it under `parameters`.

The dictionary you supply for each parameter can include the following:

* title (str): Short description of the parameter.
* description (str): Long description of the parameter.
* type (str or array): expected type of entry.
We recommend you use the pre-defined cross-reference `$ref: "#/$defs/TechParamNullNumber"` instead of explicitly using this key, to allow the parameter to be either numeric or an indexed parameter.
If you are adding a cost, you can use the cross reference `$ref: "#/$defs/TechCostNullNumber"`.
If you want to allow non-numeric data (e.g., strings), you would set `type: string` instead of using the cross-reference.
* default (str): default value.
This will be used in generating the optimisation problem.
* x-type (str): type of the non-NaN array entries in the internal calliope representation of the parameter.
This is usually one of `float` or `str`.
* x-unit (str): Unit of the parameter to use in documentation.
* x-operate-param (bool): If True, this parameter's schema data will only be loaded into the optimisation problem if running in "operate" mode.

!!! note

    Schema attributes which start with `x-` are calliope-specific.
    They are not used at all for YAML validation and instead get picked up by us using the utility function `calliope.util.schema.extract_from_schema`.

## Writing your own math documentation

You can write your model's mathematical formulation to view it in a rich-text format (as we do for our [pre-defined math](../pre_defined_math/index.md) in this documentation).
To write a LaTeX, reStructuredText, or Markdown file that includes only the math valid for your model:

```python
model = calliope.Model("path/to/model.yaml")
model.build_math_documentation(include="valid")
model.write_math_documentation(filename="path/to/output/file.[tex|rst|md]")
```

You can then convert this to a PDF or HTML page using your renderer of choice.
We recommend you only use HTML as the equations can become too long for a PDF page.
