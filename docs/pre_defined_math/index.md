# Pre-defined math

As of Calliope version 0.7, the math used to build optimisation problems is stored in YAML files.
Our pre-defined math is a re-implementation of the formerly hardcoded math formulation in this YAML format.

Calliope follows a strict order of priority when applying math: **base math -> mode math -> extra math**.

* **Base math**, defined in `config.init.base_math`, is _always_ applied to any Calliope problem.
By default, it is a 'snapshot' [capacity planning problem][base-math] with perfect foresight.
* **Mode math**, activated via `config.build.mode`, is reserved for special cases that require additional processing within Calliope.
This includes [operate](mode.md#operate-mode) and [spores](mode.md#spores-mode) modes.
* **Extra math**, activated via `config.build.extra_math`, is for additional formulations that can be _optionally_ loaded on top of the base math.
For instance, the [inter-cluster storage][inter-cluster-storage-math] extra math allows you to track storage levels in technologies more accurately when you are using timeseries clustering in your model.

To load optional pre-defined math on top of the base math, you can reference it by name in your model configuration.

!!! example
    In the example below, Calliope will first apply all the [`base`][base-math] pre-defined math, then the [`storage_inter_cluster`][inter-cluster-storage-math] pre-defined math.

    ```yaml
    config:
    build:
        extra_math: [storage_inter_cluster]
    ```

All pre-defined math YAML files can be found in [`math` directory of the Calliope source code](https://github.com/calliope-project/calliope/blob/main/src/calliope/math/storage_inter_cluster.yaml).

If you want to introduce new constraints, decision variables, or objectives, you can do so as part of the collection of YAML files describing your model.
See the [user-defined math](../user_defined_math/index.md) section for an in-depth guide to applying your own math.
