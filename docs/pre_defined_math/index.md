# Pre-defined math

As of Calliope version 0.7, the math used to build optimisation problems is stored in YAML files.
The pre-defined math is a re-implementation of the formerly hardcoded math formulation in this YAML format.

The pre-defined math for your chosen run [mode](../creating/config.md#configbuildmode) is _always_ applied to your model when you `build` the optimisation problem.
We have also pre-defined some additional math, which you can _optionally_ load into your model.
For instance, the [inter-cluster storage][inter-cluster-storage-math] math allows you to track storage levels in technologies more accurately when you are using timeseries clustering in your model.

To load optional, pre-defined math on top of the base math, you can reference it by name in your model configuration:

```yaml
config:
  build:
    extra_math: [storage_inter_cluster]
```

If you are running in the `base` run mode, this will first apply all the [`base`][base-math] pre-defined math, then the [`storage_inter_cluster`][inter-cluster-storage-math] pre-defined math.
All pre-defined math YAML files can be found in [`math` directory of the Calliope source code](https://github.com/calliope-project/calliope/blob/main/src/calliope/math/storage_inter_cluster.yaml).

If you want to introduce new constraints, decision variables, or objectives, you can do so as part of the collection of YAML files describing your model.
See the [user-defined math](../user_defined_math/index.md) section for an in-depth guide to applying your own math.

The pre-defined math can be explored in this section by selecting one of the options in the left-hand-side table of contents.
