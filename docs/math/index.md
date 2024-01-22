# Inbuilt math

As of Calliope version 0.7, the math used to build optimisation problems is stored in YAML files.
The inbuilt math is a re-implementation of the formerly hardcoded math formulation in this YAML format.

The base math is _always_ applied to your model when you `build` the optimisation problem.
In addition, there are other math formulation files built in which you can load into your model.
For instance, the [inter-cluster storage][inter-cluster-storage-math] math allows you to track storage levels in technologies more accurately when you are using timeseries clustering in your model.

To load one of the built-in math files, you can reference it by name (_without_ the file extension) in your model configuration:

```yaml
config:
  init:
    custom_math: [storage_inter_cluster]
```

When solving the model in a run mode other than `plan`, some built-in custom math will be applied automatically from a file of the same name (e.g., `spores` mode custom math is stored in [math/spores.yaml](https://github.com/calliope-project/calliope/blob/main/src/calliope/math/spores.yaml)).

!!! note

    Custom math is applied in the order it appears in the `#!yaml config.init.custom_math` list.
    By default, any run mode custom math will be applied as the final step.
    If you want to apply your own custom math *after* the run mode custom math, you should add the name of the run mode explicitly to the `#!yaml config.init.custom_math` list, e.g., `#!yaml config.init.custom_math: [operate, my_custom_math.yaml]`.

If you want to introduce new constraints, decision variables, or objectives, you can do so as part of the collection of YAML files describing your model.
See the [custom math][custom-math-formulation] section for an in-depth guide to applying custom math.

The inbuilt math documentation can be explored in this section by selecting one of the options in the left-hand side table of contents.

## A guide to math documentation

If a math component's initial conditions are met (those to the left of the curly brace), it will be applied to a model.
For each objective, constraint and global expression, a number of sub-conditions then apply (those to the right of the curly brace) to decide on the specific expression to apply at a given iteration of the component dimensions.

In the expressions, terms in **bold** font are decision variables and terms in *italic* font are parameters.
A list of the decision variables is given at the end of this page.
A detailed listing of parameters along with their units and default values is given in the [model definition reference sheet][model-definition-schema].
Those parameters which are defined over time (`timesteps`) in the expressions can be defined by a user as a single, time invariant value, or as a timeseries that is [loaded from file or dataframe][loading-tabular-data-data_sources].
