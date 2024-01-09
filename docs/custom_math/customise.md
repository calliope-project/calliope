# Introducing custom math to your model

Once you understand the [math components][math-components] and the [formulation syntax][math-syntax], you'll be ready to introduce custom math to your model.

You can find examples of custom math that we have put together in the [custom math example gallery][custom-math-example-gallery].

Whenever you introduce your own math, it will be applied on top of the [base math][base-math].
Therefore, you can include base math overrides as well as add new math.
For example, you may want to introduce a timeseries parameter to the built-in `storage_max` constraint to limit maximum storage capacity on a per-timestep basis:

```yaml
storage_max:
  equations:
    - expression: storage <= storage_cap * time_varying_parameter
```

The other elements of the `storage_max` constraints have not changed (`foreach`, `where`, ...), so we do not need to define them again when overriding the custom math.

When defining your model, you can reference any number of YAML files containing the custom math you want to add in `config.init`. The paths are relative to your main model configuration file:

```yaml
config:
  init:
    custom_math: [my_new_math_1.yaml, my_new_math_2.yaml]
```

You can also define a mixture of your custom math and the [inbuilt math][inbuilt-math]:

```yaml
config:
  init:
    custom_math: [my_new_math_1.yaml, storage_inter_cluster, my_new_math_2.md]
```

## Writing your own math documentation

You can write your model's mathematical formulation to view it in a rich-text format (as we do for our [inbuilt math][inbuilt-math] in this documentation).
To write a LaTeX, reStructuredText, or Markdown file that includes only the math valid for your model:

```python
model = calliope.Model("path/to/model.yaml")
model.build_math_documentation(include="valid")
model.write_math_documentation(filename="path/to/output/file.[tex|rst|md]")
```

You can then convert this to a PDF or HTML page using your renderer of choice.
We recommend you only use HTML as the equations can become too long for a PDF page.
