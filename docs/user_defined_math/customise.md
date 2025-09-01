# Adding your own math to a model

Once you understand the [math components](components.md) and the [formulation syntax](syntax.md), you'll be ready to introduce your own math to a model.

!!! info
    You can find examples of additional math that we have put together in our [math example gallery](../examples/overview.md).

Whenever you introduce your own math, it can either be _added_ on top of our pre-defined math or _replace_ it entirely.

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
It is even possible to define a mixture of your math and other [pre-defined math](../basic/modes.md):

```yaml
config:
  init:
    extra_math: [my_new_math_1, storage_inter_cluster, my_new_math_2]
```

???+ tip
    Always remember Calliope's strict order of priority: **base math -> mode -> extra math**.

Finally, when working in an interactive Python session, you can add math as a dictionary at model instantiation:

```python
calliope.from_yaml(..., math_dict={"my_new_math_1": {...}, ...})
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

## Adding your own parameters to the math definition

The math definition contains metadata about the parameters, dimensions, and lookup tables used in defining the optimisation problem.
We use this to validate user inputs.

When you add your own math you are likely to be adding new parameters to the model.
Be sure to add your parameter metadata to the math definition as well.
This ensures that your parameters have default values attached to them and if you choose to [write your own documentation](#writing-your-own-math-documentation), your parameters will have this metadata added to their descriptions.

Entries look like this:

```yaml
dims:
  techs:
    dtype: string
    title: Technologies

parameters:
  flow_cap_max:
    default: .inf
    title: Maximum rated flow capacity.
    description: >-
        Limits `flow_cap` to a maximum.
    unit: power.
lookups:
  source_unit:
    default: absolute
    title: Source unit
    description: The unit of a given technologies source data
    one_of: [absolute, per_cap, per_area]

```

## Writing your own math documentation

You can write your model's mathematical formulation to view it in a rich-text format (as we do for our [pre-defined math](../basic/modes.md) in this documentation).
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
    We use this functionality in our [pre-defined math](../basic/modes.md).
