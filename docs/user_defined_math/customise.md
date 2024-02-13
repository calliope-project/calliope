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
