# Adding your own math to a model

Once you understand the [math components](components.md) and the [formulation syntax](syntax.md), you'll be ready to introduce your own math to a model.

!!! info
    You can find examples of additional math that we have put together in our [math example gallery](examples/index.md).

Whenever you introduce your own math, it can either be _added_ on top of our [pre-defined math](../pre_defined_math/index.md) or _replace_ it entirely.

## Adding extra math

The simplest way to add math is to extend Calliope's pre-existing formulation by defining a new **extra math** option.
For example, you may want to introduce a timeseries parameter to the pre-defined `storage_max` constraint to limit maximum storage capacity on a per-timestep basis:

```yaml
storage_max:
  equations:
    - expression: storage <= storage_cap * time_varying_parameter
```

This will not change other elements of the `storage_max` constraints (`foreach`, `where`, ...), so we do not need to define them again when adding our own twist on the pre-defined math.

When defining your model, you can reference any number of YAML files containing the math you want to add in `config.init.extra_math`.
Both absolute paths and paths relative to `model.yaml` are valid.

```yaml
config:
  init:
    math_paths:
      my_new_math_1: "my_new_math_1.yaml"
      my_new_math_2: "/home/your_name/Documents/my_new_math_2.yaml"
```

You can then select which math to apply in this model run by specifying it in `config.init.extra_math`.
It is even possible to define a mixture of your math and other [pre-defined math](../pre_defined_math/index.md):

```yaml
config:
  init:
    extra_math: [my_new_math_1, storage_inter_cluster, my_new_math_2]
```

???+ tip
    Always remember Calliope's strict order of priority: **base math -> mode -> extra math**.

Finally, when working in an interactive Python session, you can add math as a dictionary at build time:

```python
model.build(add_math_dict={...})
```

This will be applied after the pre-defined mode math and any extra math listed in `config.init.extra_math`.

!!! note
    When working in an interactive Python session, you can view the final math dictionary that has been applied to build the optimisation problem by inspecting `model.math.build` at any point.

## Re-defining Calliope's pre-defined base math

If you prefer to start from scratch with your math, you can ask Calliope to completely replace our pre-defined **base math** with your own.

```yaml
config:
  init:
    math_paths: {base: your/base_math_file.yaml}
```

This will tell Calliope to overwrite _all_ of our pre-defined `base` math with your file.

You can similarly replace _mode_ math like that used in `operate` mode:

```yaml
config:
  init:
    math_paths: {operate: your/operate_math_file.yaml}
```

!!! danger
    Modes and other pre-defined options such as `operate` and `spores` might not work as expected!

## Adding your parameters to the YAML schema

Our YAML schemas are used to validate user inputs.
The model definition schema includes metadata on all our pre-defined parameters, which you can find rendered in our [reference page][model-definition-schema].

When you add your own math you are likely to be adding new parameters to the model.
You can update the Calliope model definition schema to include your new entries using [`calliope.util.schema.update_model_schema(...)`][calliope.util.schema.update_model_schema].
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
    Schema attributes which start with `x-` are Calliope-specific.
    They are not used at all for YAML validation and instead get picked up by us using the utility function [calliope.util.schema.extract_from_schema][].

!!! warning
    The schema is updated in-place so your edits to it will remain active as long as you are running in the same session.
    You can reset your updates to the schema and return to the pre-defined schema by calling [`calliope.util.schema.reset()`][calliope.util.schema.reset]

## Writing your own math documentation

You can write your model's mathematical formulation to view it in a rich-text format (as we do for our [pre-defined math](../pre_defined_math/index.md) in this documentation).
To write a LaTeX, reStructuredText, or Markdown file that includes only the math valid for your model:

```python
from calliope.postprocess.math_documentation import MathDocumentation

model = calliope.Model("path/to/model.yaml")
model.build()

math_documentation = MathDocumentation(model, include="valid")
math_documentation.write(filename="path/to/output/file.[tex|rst|md]")
```

You can then convert this to a PDF or HTML page using your renderer of choice.
We recommend you only use HTML as the equations can become too long for a PDF page.

!!! note
    You can add interactive elements to your documentation, if you are planning to host them online using MKDocs.
    This includes tabs to flip between rich-text math and the input YAML snippet, and dropdown lists for math component cross-references.
    Just set the `mkdocs_features` argument to `True` in `math_documentation.write`.
    We use this functionality in our [pre-defined math](../pre_defined_math/index.md).
