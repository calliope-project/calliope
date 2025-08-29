# Creating a model

## Structuring your model directory

It makes sense to collect all files belonging to a model inside a single model directory.
The layout of that directory typically looks roughly like this (`+` denotes directories, `-` files):

```
+ example_model
    + model_definition
        - nodes.yaml
        - techs.yaml
    + data_tables
        - solar_resource.csv
        - electricity_demand.csv
    - model.yaml
    - scenarios.yaml
```

In the above example, the files `model.yaml`, `nodes.yaml` and `techs.yaml` together are the model definition.
This definition could be in one file, but it is more readable when split into multiple.
We use the above layout in the example models.

Inside the `data_tables` directory, tabular data are stored as CSV files.

!!! note
    The easiest way to create a new model is to use the `calliope new` command, which makes a copy of one of the built-in examples models:

    ```shell
    $ calliope new my_new_model
    ```

    This creates a new directory, `my_new_model`, in the current working directory.

    By default, `calliope new` uses the national-scale example model as a template.
    To use a different template, you can specify the example model to use, e.g.: `--template=urban_scale`.
